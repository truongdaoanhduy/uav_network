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
    norm = torch.nn.utils.clip_grad_norm_(list(parameters), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return norm.detach()


def scalar_metrics(values):
    keys = list(values)
    tensors = [torch.as_tensor(values[k]).detach() for k in keys]
    return dict(zip(keys, torch.stack(tensors).cpu().tolist()))


def step_actor_optimizers(losses, optimizers, parameter_groups, grad_clip, check_finite=True):
    """Independent actor gradients/clipping, one host safety check per group.

    Peers in each loss must be detached. No actor Adam step occurs until every
    actor gradient is finite, including when scalar metrics are disabled.
    """
    if check_finite and not torch.isfinite(torch.stack(losses)).all():
        raise FloatingPointError("Non-finite actor loss")
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward(losses)
    norms = [torch.nn.utils.clip_grad_norm_(list(parameters), grad_clip, error_if_nonfinite=False).detach()
             for parameters in parameter_groups]
    if not torch.isfinite(torch.stack(norms)).all():
        raise RuntimeError("Non-finite actor gradient norm; refusing all actor Adam steps")
    for optimizer in optimizers:
        optimizer.step()
    return norms


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


def critic_metrics(q1, q2, target, data):
    """Detached replay-batch diagnostics; collect only at configured intervals."""
    q1, q2, target = q1.detach(), q2.detach(), target.detach()
    td = q1 - target
    return {"q_mean": q1.mean(), "q_abs_max": q1.abs().max(), "target_mean": target.mean(),
            "target_abs_max": target.abs().max(), "td_error_abs_mean": td.abs().mean(),
            "td_error_abs_max": td.abs().max(), "twin_q_gap": (q1 - q2).abs().mean(),
            "batch_reward_mean": data["reward"].mean(),
            "batch_terminated_fraction": data["terminated"].float().mean(),
            "batch_active_uav_fraction": (data["observation"][:, :, 9] > .5).float().mean()}
