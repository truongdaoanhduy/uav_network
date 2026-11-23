"""Tensor updates: periodic scalar checks, gradient safety before every step."""
import os
import torch
from torch import nn


def _enabled(name):
    """Opt-in infrastructure optimization. Baseline remains the default."""
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes"}


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


def gradient_norm_and_clip(parameters, grad_clip, mode="global", *, error_if_nonfinite=True):
    """Apply clipping; report global norm or max tensor norm for per-parameter mode."""
    parameters = list(parameters)
    if mode == "global" and _enabled("UAV_FAST_GRAD_CLIP"):
        # Avoid a duplicate full-gradient norm reduction. PyTorch computes the
        # global pre-clipping norm and applies exactly the same clip threshold.
        # Keep error_if_nonfinite active even when metrics are disabled.
        return torch.nn.utils.clip_grad_norm_(
            parameters, grad_clip, error_if_nonfinite=error_if_nonfinite
        ).detach()
    if mode == "per_parameter" and _enabled("UAV_FAST_GRAD_CLIP"):
        # Each clip_grad_norm_ already returns that parameter's pre-clip norm;
        # do not compute the same per-tensor norm once more for diagnostics.
        # Preserve exactly the existing per-parameter clipping policy.
        norms = [torch.nn.utils.clip_grad_norm_(
                    [p], grad_clip, error_if_nonfinite=error_if_nonfinite).detach()
                 for p in parameters if p.grad is not None]
        return (torch.stack(norms).max() if norms
                else torch.tensor(0., device=parameters[0].device if parameters else "cpu"))
    gradients = [p.grad for p in parameters if p.grad is not None]
    if gradients:
        individual_norms = torch.stack([torch.linalg.vector_norm(g) for g in gradients])
        norm = torch.linalg.vector_norm(individual_norms)
    else:
        device = parameters[0].device if parameters else "cpu"
        norm = torch.tensor(0.0, device=device)
    if error_if_nonfinite and not torch.isfinite(norm):
        raise RuntimeError("Non-finite gradient norm")
    if mode == "none":
        return norm.detach()
    if mode == "global":
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=error_if_nonfinite)
    elif mode == "per_parameter":
        for parameter in parameters:
            if parameter.grad is not None:
                torch.nn.utils.clip_grad_norm_([parameter], grad_clip, error_if_nonfinite=error_if_nonfinite)
    else:
        raise ValueError(f"Unknown gradient clipping mode: {mode}")
    reported = individual_norms.max() if mode == "per_parameter" and gradients else norm
    return reported.detach()


def step_optimizer(loss, optimizer, parameters, grad_clip, check_finite=True, grad_clip_mode="global"):
    if check_finite and not torch.isfinite(loss):
        raise FloatingPointError("Non-finite learner loss")
    parameters = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = gradient_norm_and_clip(parameters, grad_clip, grad_clip_mode, error_if_nonfinite=True)
    optimizer.step()
    return norm


def scalar_metrics(values):
    keys = list(values)
    # Metrics may mix Python/CPU scalars with learner scalars on CUDA. Move each
    # detached scalar to host before stacking; this is telemetry-only and does
    # not participate in the learner graph or optimizer step.
    if _enabled("UAV_BATCH_SCALAR_METRICS") and keys:
        # A single stacked device-to-host transfer avoids K individual CUDA
        # synchronization points. Mixed CPU/GPU or non-scalar paths deliberately
        # preserve the original implementation, including output key order.
        tensors = [torch.as_tensor(values[k]).detach() for k in keys]
        if (all(t.is_cuda and t.device == tensors[0].device
                and t.shape == tensors[0].shape for t in tensors)):
            return dict(zip(keys, torch.stack(tensors).cpu().tolist()))
    tensors = [torch.as_tensor(values[k]).detach().cpu() for k in keys]
    return dict(zip(keys, torch.stack(tensors).tolist()))


def step_actor_optimizers(losses, optimizers, parameter_groups, grad_clip, check_finite=True,
                          grad_clip_mode="global"):
    """Independent actor gradients/clipping, one host safety check per group.

    Peers in each loss must be detached. No actor Adam step occurs until every
    actor gradient is finite, including when scalar metrics are disabled.
    """
    if check_finite and not torch.isfinite(torch.stack(losses)).all():
        raise FloatingPointError("Non-finite actor loss")
    parameter_groups = [list(parameters) for parameters in parameter_groups]
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward(losses)
    norms = [gradient_norm_and_clip(parameters, grad_clip, grad_clip_mode, error_if_nonfinite=False)
             for parameters in parameter_groups]
    if not torch.isfinite(torch.stack(norms)).all():
        raise RuntimeError("Non-finite actor gradient norm; refusing all actor Adam steps")
    for optimizer in optimizers:
        optimizer.step()
    return norms


def compile_models(learner):
    """Compile the whole vmap boundary; preserve original parameter/state keys."""
    from .parallel_critics import critic_values
    if not hasattr(nn.Module, "compile"):
        raise RuntimeError("torch.compile requires a supported PyTorch build")
    learner._compiled_critic_values = torch.compile(critic_values, dynamic=True, backend="inductor")
    for actor in learner.actors:
        actor.encoder.compile(dynamic=True, backend="inductor")
    if hasattr(learner, "target_actors"):
        for actor in learner.target_actors:
            actor.encoder.compile(dynamic=True, backend="inductor")


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
