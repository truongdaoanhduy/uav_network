"""Extracted unchanged MATD3 diagnostic methods (no optimization changes)."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from uav_network.algorithms.core.action_utils import valid_destinations
from uav_network.algorithms.core.models import _apply_activation

class MATD3DiagnosticsMixin:
    @torch.no_grad()
    def _actor_output_diagnostics(self, observations):
        """Read-only, sampled pre-tanh/route diagnostics; no policy modification.

        LayerNorm acts on hidden features, not on the final tanh output. Track
        final-layer saturation separately from an action mask that permits
        only silence. This is intentionally limited to a small sample to
        avoid material overhead with large replay batches.
        """
        probe = observations[:32]
        if self.share_actor_parameters:
            batch, n = probe.shape[:2]
            flat = probe.reshape(batch * n, -1)
            ids = torch.arange(n, device=probe.device).expand(batch, n).reshape(-1)
            actor = self.actors[0]
            hidden = _apply_activation(actor.encoder(self._actor_input(flat, ids)), actor.activation)
            raw = actor.continuous(hidden).reshape(batch, n, self.continuous_dim)
            pretanh = actor.normalize_pretanh(raw)
            logits = actor.destination(hidden).reshape(batch, n, self.destinations)
            hidden = hidden.reshape(batch, n, -1)
        else:
            hidden = torch.stack([
                _apply_activation(actor.encoder(probe[:, i]), actor.activation)
                for i, actor in enumerate(self.actors)
            ], dim=1)
            raw = torch.stack([
                actor.continuous(hidden[:, i])
                for i, actor in enumerate(self.actors)
            ], dim=1)
            logits = torch.stack([
                actor.destination(hidden[:, i])
                for i, actor in enumerate(self.actors)
            ], dim=1)
            pretanh = torch.stack([
                actor.normalize_pretanh(raw[:, i])
                for i, actor in enumerate(self.actors)
            ], dim=1)
        derivative = 1 - pretanh.tanh().square()
        motion = pretanh[..., :self.config.motion_dim]
        motion_derivative = derivative[..., :self.config.motion_dim]
        has_rf = self.continuous_dim > self.config.motion_dim
        rf_derivative = derivative[..., -1] if has_rf else torch.zeros_like(derivative[..., 0])
        mask = valid_destinations(probe, self.destinations)
        valid_count = mask.sum(-1)
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        probabilities = F.softmax(masked_logits, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        metrics = {
            "actor_motion_head_raw_abs_mean": raw[..., :self.config.motion_dim].abs().mean(),
            "actor_motion_pretanh_abs_mean": motion.abs().mean(),
            "actor_motion_pretanh_abs_max": motion.abs().max(),
            "actor_motion_tanh_derivative_mean": motion_derivative.mean(),
            "actor_motion_tanh_derivative_lt_1e3_fraction":
                (motion_derivative < 1e-3).float().mean(),
            "actor_motion_abs_action_gt_0_99_fraction":
                (motion.tanh().abs() > .99).float().mean(),
            "actor_rf_pretanh_abs_mean": (raw[..., -1].abs().mean() if has_rf else torch.zeros((), device=probe.device)),
            "actor_rf_tanh_derivative_lt_1e3_fraction": (
                (rf_derivative < 1e-3).float().mean() if has_rf else torch.zeros((), device=probe.device)),
            "actor_mask_only_silent_fraction": (valid_count == 1).float().mean(),
            "actor_mask_eligible_fraction": (valid_count > 1).float().mean(),
            "actor_destination_entropy_mean": entropy.mean(),
            "actor_destination_max_probability_mean": probabilities.max(-1).values.mean(),
            "actor_hidden_activation_zero_fraction": (hidden == 0).float().mean(),
            # Historical field retained as an alias; not a dormant-neuron rate.
            "actor_hidden_relu_zero_fraction": (hidden == 0).float().mean(),
            "actor_hidden_activation_abs_mean": hidden.abs().mean(),
        }
        for axis, label in enumerate("xyz"[:self.config.motion_dim]):
            metrics[f"actor_{label}_pretanh_abs_mean"] = pretanh[..., axis].abs().mean()
            metrics[f"actor_{label}_tanh_derivative_lt_1e3_fraction"] = (
                derivative[..., axis] < 1e-3).float().mean()
        return metrics


    @torch.no_grad()
    def _actor_head_gradient_diagnostics(self):
        """Post-clipping head gradient norms, reported only at log intervals."""
        def head_norms(extract):
            values = []
            for actor in self.actors:
                parameter = extract(actor)
                if parameter.grad is not None:
                    values.append(parameter.grad.detach().norm())
            if values:
                return torch.stack(values).mean()
            return torch.zeros((), device=self.device)

        first_linear = lambda actor: next(
            module.weight for module in actor.encoder.modules() if isinstance(module, nn.Linear)
        )
        return {
            "actor_motion_head_grad_postclip": self._sliced_continuous_grad_norm(is_motion=True),
            "actor_rf_head_grad_postclip": self._sliced_continuous_grad_norm(is_motion=False),
            "actor_destination_head_grad_postclip": head_norms(
                lambda actor: actor.destination.weight
            ),
            "actor_first_encoder_grad_postclip": head_norms(first_linear),
        }


    @torch.no_grad()
    def _sliced_continuous_grad_norm(self, *, is_motion):
        values = []
        if not is_motion and self.continuous_dim <= self.config.motion_dim:
            return torch.zeros((), device=self.device)
        for actor in self.actors:
            gradient = actor.continuous.weight.grad
            if gradient is not None:
                component = gradient[:self.config.motion_dim] if is_motion else gradient[-1:]
                values.append(component.norm())
        return torch.stack(values).mean() if values else torch.zeros((), device=self.device)
