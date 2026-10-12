"""Tensor updates: periodic scalar checks, gradient safety before every step."""
import torch
from torch import nn


def tensor_batch(batch, device):
    result = {}
    for name, value in batch.items():
        dtype = torch.long if name == "destination" else torch.float32
        # CPU consumers need completed D2H copies. H2D work stays ordered on the
        # CUDA stream; pinned samples and GPU-resident replay can use it safely.
        non_blocking = torch.device(device).type == "cuda"
        result[name] = (value.to(device=device, dtype=dtype, non_blocking=non_blocking) if isinstance(value, torch.Tensor)
                        else torch.as_tensor(value, device=device, dtype=dtype))
    return result


def step_optimizer(loss, optimizer, parameters, grad_clip, check_finite=True):
    if check_finite and not torch.isfinite(loss):
        raise FloatingPointError("Non-finite learner loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(parameters), grad_clip, error_if_nonfinite=True)
    optimizer.step()


def scalar_metrics(values):
    keys = list(values)
    tensors = [torch.as_tensor(values[k]).detach() for k in keys]
    return dict(zip(keys, torch.stack(tensors).cpu().tolist()))


def compile_models(learner):
    if not hasattr(nn.Module, "compile"):
        raise RuntimeError("torch.compile requires a supported PyTorch build")
    for actor in learner.actors:
        actor.encoder.compile(dynamic=True)
    for collection in (learner.critics, learner.target_critics):
        for critic in collection:
            critic.compile(dynamic=True)
    if hasattr(learner, "target_actors"):
        for actor in learner.target_actors:
            actor.encoder.compile(dynamic=True)
