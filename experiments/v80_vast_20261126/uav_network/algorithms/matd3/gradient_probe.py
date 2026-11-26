"""Low-frequency, read-only attribution of MATD3 policy gradients.

Only attaches hooks to the existing actor backward on requested metric steps.
No extra actor/critic forward, backward, random draw or gradient modification.
The critic-facing gradient is d(-actor_loss)/da (scaled back from batch means),
not the gradient through the CBF, which is not part of the actor graph.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class MATD3GradientProbe:
    def __init__(self, actors, *, batch_size, num_agents, motion_dim,
                 continuous_dim, share_actor, grad_clip, grad_clip_mode="per_parameter", enabled=False):
        self.actors = actors
        self.batch_size = int(batch_size)
        self.num_agents = int(num_agents)
        self.motion_dim = int(motion_dim)
        self.continuous_dim = int(continuous_dim)
        self.share_actor = bool(share_actor)
        self.grad_clip = float(grad_clip)
        self.grad_clip_mode = grad_clip_mode
        self.enabled = bool(enabled)
        self.handles = []
        self.raw_grads = []
        self.action_grads = []
        self.action_grad_signed = []
        self.action_grad_saturation_alignment = []
        self.action_grad_saturation_support = []
        self.forward_metrics = []
        self.parameter_grads = []
        self.preclip = {}
        self.encoder_zero_rates = []
        self.encoder_grad = []
        self.internal_relu_zero_rates = []
        self.internal_relu_grad = []

    @staticmethod
    def _mean(vals):
        return torch.stack(vals).mean(dim=0) if vals else None

    @staticmethod
    def _summarize_axes(tensor):
        """One scalar per action coordinate; no host sync or graph retention."""
        return tensor.detach().abs().reshape(-1, tensor.shape[-1]).mean(0)

    def __enter__(self):
        if not self.enabled:
            return self
        for actor in self.actors:
            self.handles.append(actor.continuous.register_forward_hook(
                lambda _module, _inputs, raw, actor=actor: self._forward_raw(actor, raw)))
            # The final F.relu is outside encoder; encoder's output <=0 is
            # exactly the region where the final ReLU blocks derivatives.
            self.handles.append(actor.encoder.register_forward_hook(self._forward_encoder))
            for module in actor.encoder.modules():
                if isinstance(module, nn.ReLU):
                    self.handles.append(module.register_forward_hook(self._forward_internal_relu))
            first_linear = next(m for m in actor.encoder.modules() if isinstance(m, nn.Linear))
            last_linear = [m for m in actor.encoder.modules() if isinstance(m, nn.Linear)][-1]
            targets = {"encoder_first": first_linear.weight,
                       "encoder_last": last_linear.weight,
                       "motion_head": actor.continuous.weight,
                       "routing_head": actor.destination.weight}
            for label, parameter in targets.items():
                self.handles.append(parameter.register_hook(
                    lambda grad, label=label: self._parameter_hook(label, grad)))
        return self

    def _forward_raw(self, actor, raw):
        # Inspect the input to tanh. The backward hook below still measures
        # the gradient reaching the linear head through both transformations.
        with torch.no_grad():
            v = actor.normalize_pretanh(raw.detach())
            derivative = 1.0 - v.tanh().square()
            self.forward_metrics.append((
                self._summarize_axes(v),
                self._summarize_axes(derivative),
                v.reshape(-1, v.shape[-1]).mean(0),
                (derivative < 1e-3).float().reshape(-1, v.shape[-1]).mean(0),
            ))
        if raw.requires_grad:
            self.handles.append(raw.register_hook(self._raw_hook))

    def _forward_encoder(self, _module, _inputs, output):
        with torch.no_grad():
            self.encoder_zero_rates.append((output.detach() <= 0).float().mean())
        if output.requires_grad:
            self.handles.append(output.register_hook(self._encoder_hook))

    def _forward_internal_relu(self, _module, _inputs, output):
        with torch.no_grad():
            self.internal_relu_zero_rates.append((output.detach() == 0).float().mean())
        if output.requires_grad:
            self.handles.append(output.register_hook(self._internal_relu_hook))

    def _encoder_hook(self, grad):
        self.encoder_grad.append(grad.detach().abs().mean() * self.batch_size *
                                 (self.num_agents if self.share_actor else 1))
        return None

    def _internal_relu_hook(self, grad):
        self.internal_relu_grad.append(grad.detach().abs().mean() * self.batch_size *
                                       (self.num_agents if self.share_actor else 1))
        return None

    def _raw_hook(self, grad):
        # Scale back the batch/agent averaging in actor_loss to express a
        # per-sample signal, comparable across E and sharing configurations.
        factor = self.batch_size * (self.num_agents if self.share_actor else 1)
        self.raw_grads.append(self._summarize_axes(grad * factor))
        return None

    def watch_action(self, action):
        """Hook the action actually passed into the centralized Q1 critic."""
        if self.enabled and action.requires_grad:
            self.handles.append(action.register_hook(
                lambda grad, reference=action: self._action_hook(grad, reference)))

    def _action_hook(self, grad, action):
        factor = self.batch_size * (self.num_agents if self.share_actor else 1)
        dqda = -grad.detach() * factor
        self.action_grads.append(self._summarize_axes(dqda))
        self.action_grad_signed.append(dqda.reshape(-1, dqda.shape[-1]).mean(0))
        with torch.no_grad():
            saturated = (action.detach().abs() > .95).reshape(-1, action.shape[-1])
            aligned = ((dqda * action.detach()) > 0).reshape(-1, action.shape[-1])
            self.action_grad_saturation_alignment.append((aligned & saturated).float().sum(0))
            self.action_grad_saturation_support.append(saturated.float().sum(0))
        return None

    def _parameter_hook(self, label, grad):
        # Captures the unmodified gradient *before* per-parameter clip.
        with torch.no_grad():
            g = grad.detach()
            norm = torch.linalg.vector_norm(g)
            self.parameter_grads.append(norm)
            self.preclip.setdefault(label, []).append(norm)
            if label == "motion_head":
                self.preclip.setdefault("motion_head_rows", []).append(
                    torch.linalg.vector_norm(g, dim=1))
        return None

    def metrics(self):
        if not self.enabled:
            return {}
        out = {}
        # Canonical 2D motor policy has no RF; historical adapters append RF.
        axes = (list("xyz"[:self.motion_dim]) +
                (["packet"] if self.continuous_dim == self.motion_dim + 2 else []) +
                (["rf"] if self.continuous_dim > self.motion_dim else []))
        for group, values in (("critic_facing_dloss_da", self.action_grads),
                              ("pretanh_dloss_dz", self.raw_grads)):
            mean = self._mean(values)
            if mean is not None:
                for i, axis in enumerate(axes):
                    out[f"gradprobe/{group}_{axis}_abs_mean"] = mean[i]
        if self.action_grad_signed:
            signed = self._mean(self.action_grad_signed)
            aligned_count = torch.stack(self.action_grad_saturation_alignment).sum(0)
            saturated_count = torch.stack(self.action_grad_saturation_support).sum(0)
            for i, axis in enumerate(axes):
                out[f"gradprobe/critic_facing_dq_da_{axis}_signed_mean"] = signed[i]
                out[f"gradprobe/critic_gradient_reinforces_saturation_{axis}_fraction"] = (
                    aligned_count[i] / saturated_count[i].clamp_min(1))
                out[f"gradprobe/saturated_action_{axis}_fraction_at_backward"] = (
                    saturated_count[i] / (self.batch_size * self.num_agents))
        if self.forward_metrics:
            raw_abs = self._mean([v[0] for v in self.forward_metrics])
            derivatives = self._mean([v[1] for v in self.forward_metrics])
            raw_signed = self._mean([v[2] for v in self.forward_metrics])
            saturated = self._mean([v[3] for v in self.forward_metrics])
            for i, axis in enumerate(axes):
                out[f"gradprobe/actor_pretanh_{axis}_abs_mean_fullbatch"] = raw_abs[i]
                out[f"gradprobe/actor_pretanh_{axis}_signed_mean_fullbatch"] = raw_signed[i]
                out[f"gradprobe/actor_tanh_derivative_{axis}_mean_fullbatch"] = derivatives[i]
                out[f"gradprobe/actor_tanh_derivative_lt_1e3_{axis}_fraction_fullbatch"] = saturated[i]
        for name, values in (("encoder_final_relu_blocked_fraction_fullbatch", self.encoder_zero_rates),
                             ("encoder_final_output_grad_abs_mean", self.encoder_grad),
                             ("encoder_internal_relu_zero_fraction_fullbatch", self.internal_relu_zero_rates),
                             ("encoder_internal_relu_output_grad_abs_mean", self.internal_relu_grad)):
            value = self._mean(values)
            if value is not None:
                out[f"gradprobe/{name}"] = value
        for name in ("encoder_first", "encoder_last", "motion_head", "routing_head"):
            pre = self._mean(self.preclip.get(name, []))
            if pre is not None:
                out[f"gradprobe/{name}_grad_preclip_norm"] = pre
            post = []
            for actor in self.actors:
                linear_layers = [m for m in actor.encoder.modules() if isinstance(m, nn.Linear)]
                parameter = {"encoder_first": linear_layers[0].weight,
                             "encoder_last": linear_layers[-1].weight,
                             "motion_head": actor.continuous.weight,
                             "routing_head": actor.destination.weight}[name]
                if parameter.grad is not None:
                    post.append(parameter.grad.detach().norm())
            if post:
                out[f"gradprobe/{name}_grad_postclip_norm"] = torch.stack(post).mean()
        pre_rows = self._mean(self.preclip.get("motion_head_rows", []))
        if pre_rows is not None:
            for i, axis in enumerate(axes):
                out[f"gradprobe/motion_head_{axis}_grad_preclip_norm"] = pre_rows[i]
        post_rows = [torch.linalg.vector_norm(a.continuous.weight.grad.detach(), dim=1)
                     for a in self.actors if a.continuous.weight.grad is not None]
        if post_rows:
            mean = torch.stack(post_rows).mean(0)
            for i, axis in enumerate(axes):
                out[f"gradprobe/motion_head_{axis}_grad_postclip_norm"] = mean[i]
        if self.parameter_grads and self.grad_clip_mode == "per_parameter":
            norms = torch.stack(self.parameter_grads)
            out["gradprobe/monitored_parameter_clipped_fraction"] = (
                (norms > self.grad_clip).float().mean() if math.isfinite(self.grad_clip)
                else torch.zeros((), device=norms.device))
        out["gradprobe/action_hook_count"] = len(self.action_grads)
        out["gradprobe/raw_hook_count"] = len(self.raw_grads)
        return out

    def __exit__(self, _exc_type, _exc, _traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False
