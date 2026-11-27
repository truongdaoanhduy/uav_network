"""Sparse read-only MASAC logstd/entropy-head gradient diagnostics.

The observed gradients include SAC's alpha-weighted entropy objective,
so they are *not* dQ/da. No extra stochastic sample/forward/backward.
"""
from __future__ import annotations

from contextvars import ContextVar
import torch

# Read-only external gradient scaler, scoped to the standalone diagnostic.
# The user's checksum-pinned MASAC algorithm is never patched or edited.
EXTERNAL_GRADIENT_SCALE = ContextVar("uav_masac_probe_grad_scale", default=None)


class MASACGradientProbe:
    def __init__(self, actors, batch_size, agents, shared, motion_dim=3, enabled=False,
                 gradient_scale=None):
        self.actors = actors
        self.batch_size = int(batch_size)
        self.agents = int(agents)
        self.shared = bool(shared)
        # Two-axis motor-only policy is the canonical Q1-Clean action.
        # Historical experiments may append RF and an optional packet head.
        action_dim = self.actors[0].mean.out_features
        if action_dim not in (motion_dim, motion_dim + 1, motion_dim + 2):
            raise ValueError("Unsupported actor continuous head dimensionality")
        self.axes = (tuple("xyz"[:motion_dim]) +
                     (("packet",) if action_dim == motion_dim + 2 else ()) +
                     (("rf",) if action_dim > motion_dim else ()))
        self.enabled = bool(enabled)
        external = EXTERNAL_GRADIENT_SCALE.get()
        self.gradient_scale = float(gradient_scale if gradient_scale is not None else
                                    external() if callable(external) else 1.0)
        if not self.gradient_scale > 0:
            raise ValueError("Invalid gradient scale for MASAC telemetry")
        self.handles = []
        self.activations = {}
        self.head_gradients = {}
        self.param_before = {}

    @staticmethod
    def _avg(values):
        return torch.stack(values).mean(dim=0) if values else None

    @staticmethod
    def _vec_mean(value):
        v = value.detach().reshape(-1, value.shape[-1])
        return v.mean(0), v.abs().mean(0)

    def __enter__(self):
        if not self.enabled:
            return self
        for actor in self.actors:
            for label in ("mean", "log_std", "destination"):
                module = getattr(actor, label)
                self.handles.append(module.register_forward_hook(
                    lambda _mod, _args, output, label=label: self._on_forward(label, output)))
                self.handles.append(module.weight.register_hook(
                    lambda grad, label=label: self._on_weight_gradient(label, grad)))
            layers = [x for x in actor.encoder.modules() if isinstance(x, torch.nn.Linear)]
            self.handles.append(layers[0].weight.register_hook(
                lambda grad: self._on_weight_gradient("encoder_first", grad)))
        return self

    def _on_forward(self, label, output):
        if not output.requires_grad:
            return
        with torch.no_grad():
            signed, absolute = self._vec_mean(output)
            self.activations.setdefault(label, []).append((signed, absolute))
            if label == "log_std":
                self.activations.setdefault("std_clamped_low", []).append(
                    (output.detach() <= -20).float().mean())
                self.activations.setdefault("std_clamped_high", []).append(
                    (output.detach() >= 2).float().mean())
        self.handles.append(output.register_hook(
            lambda grad, label=label: self._on_output_gradient(label, grad)))

    def _on_output_gradient(self, label, grad):
        # Actor update averages per batch/agent; recover approx. per-row signal.
        scale = self.batch_size * (self.agents if self.shared else 1)
        self.head_gradients.setdefault(label, []).append(
            self._vec_mean(grad.detach() * (scale / self.gradient_scale))[1])
        return None

    def _on_weight_gradient(self, label, grad):
        self.param_before.setdefault(label, []).append(
            torch.linalg.vector_norm(grad.detach() / self.gradient_scale))
        return None

    @torch.no_grad()
    def metrics(self):
        if not self.enabled:
            return {}
        out = {}
        for label in ("mean", "log_std", "destination"):
            act = self.activations.get(label, [])
            grads = self.head_gradients.get(label, [])
            if act:
                absmean = self._avg([x[1] for x in act])
                signed = self._avg([x[0] for x in act])
                for i in range(absmean.numel()):
                    axis = f"dest_{i}" if label == "destination" else self.axes[i]
                    out[f"masac_probe/{label}_{axis}_abs_mean"] = absmean[i]
                    out[f"masac_probe/{label}_{axis}_signed_mean"] = signed[i]
            if grads:
                gradient = self._avg(grads)
                for i in range(gradient.numel()):
                    axis = f"dest_{i}" if label == "destination" else self.axes[i]
                    out[f"masac_probe/{label}_{axis}_actor_loss_gradient_abs_mean"] = gradient[i]
        for label in ("std_clamped_low", "std_clamped_high"):
            value = self._avg(self.activations.get(label, []))
            if value is not None:
                out[f"masac_probe/{label}_fraction"] = value
        for label, values in self.param_before.items():
            out[f"masac_probe/{label}_weight_grad_preclip_norm"] = self._avg(values)
        for label in ("mean", "log_std", "destination", "encoder_first"):
            post = []
            for actor in self.actors:
                if label == "encoder_first":
                    p = next(x.weight for x in actor.encoder.modules()
                             if isinstance(x, torch.nn.Linear))
                else:
                    p = getattr(actor, label).weight
                if p.grad is not None:
                    post.append(torch.linalg.vector_norm(p.grad.detach()))
            if post:
                out[f"masac_probe/{label}_weight_grad_postclip_norm"] = self._avg(post)
        out["masac_probe/forward_count"] = len(self.activations.get("mean", []))
        out["masac_probe/backward_hook_count"] = len(self.head_gradients.get("mean", []))
        return out

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False
