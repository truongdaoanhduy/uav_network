"""Learner-owned AMP scalers and the two-worker compile startup boundary.

Models, optimizer state and replay remain FP32. Update clocks belong to the
algorithms; scaler counters record which attempted optimizer groups applied.
"""
from contextlib import contextmanager, nullcontext
from functools import wraps
import math
from threading import Lock

import torch

from .optimization import step_actor_optimizers, step_optimizer


_SCALER_FIELDS = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
_COUNT_FIELDS = {"attempted", "applied", "skipped"}


def _configuration(mode, groups):
    if mode not in {"off", "fp16"}:
        raise ValueError("precision mode must be off or fp16")
    groups = tuple(groups)
    if not groups or any(not isinstance(group, str) or not group for group in groups) or len(set(groups)) != len(groups):
        raise ValueError("precision groups must be distinct nonempty names")
    return groups


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_precision_state(state, mode, groups):
    """Reject an incomplete AMP checkpoint before allocating replay or workers."""
    groups = _configuration(mode, groups)
    if state is None and mode == "off":
        return
    if (not isinstance(state, dict) or state.keys() != {"mode", "scalers", "counts"}
            or state["mode"] != mode):
        raise ValueError("Precision checkpoint mode/state is missing or incompatible")
    scalers, counts = state["scalers"], state["counts"]
    expected_scalers = set(groups) if mode == "fp16" else set()
    if not isinstance(scalers, dict) or scalers.keys() != expected_scalers:
        raise ValueError("Precision checkpoint scaler groups are incomplete")
    if not isinstance(counts, dict) or counts.keys() != set(groups):
        raise ValueError("Precision checkpoint counter groups are incomplete")
    for values in counts.values():
        if (not isinstance(values, dict) or values.keys() != _COUNT_FIELDS
                or any(type(value) is not int or value < 0 for value in values.values())
                or values["attempted"] != values["applied"] + values["skipped"]):
            raise ValueError("Precision checkpoint counters are inconsistent")
    for scaler in scalers.values():
        if not isinstance(scaler, dict) or scaler.keys() != _SCALER_FIELDS:
            raise ValueError("Precision checkpoint scaler state is incomplete")
        if (not _number(scaler["scale"]) or scaler["scale"] <= 0
                or not _number(scaler["growth_factor"]) or scaler["growth_factor"] <= 1
                or not _number(scaler["backoff_factor"]) or not 0 < scaler["backoff_factor"] < 1
                or type(scaler["growth_interval"]) is not int or scaler["growth_interval"] < 1
                or type(scaler["_growth_tracker"]) is not int
                or not 0 <= scaler["_growth_tracker"] < scaler["growth_interval"]):
            raise ValueError("Precision checkpoint scaler values are invalid")


def _total_norm(gradients):
    public = getattr(torch.nn.utils, "get_total_norm", None)
    if public is not None:
        return public(gradients, error_if_nonfinite=False).detach()
    # Torch 2.5 exposes clipping but not the split norm/clip helpers.
    return (torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients]))
            if gradients else torch.tensor(0.))


class MixedPrecision:
    """Independent critic, grouped-actor and optional alpha scaler clocks.

    CPU support is for small arithmetic tests. Production training selects CUDA
    before constructing environments or replay; no parameter dtype is changed.
    """
    def __init__(self, mode: str, device, groups: tuple[str, ...]):
        self.groups = _configuration(mode, groups)
        self.mode, self.device = mode, torch.device(device)
        self.enabled = mode == "fp16"
        self.scalers = {group: torch.amp.GradScaler(self.device.type) for group in self.groups} if self.enabled else {}
        self.counts = {group: dict(attempted=0, applied=0, skipped=0) for group in self.groups}
        self.overflow_observer = None
        self.observer_errors = 0

    def autocast(self):
        # A disabled context must preserve diagnostic callers' outer autocast.
        return (torch.autocast(self.device.type, dtype=torch.float16, cache_enabled=False)
                if self.enabled else nullcontext())

    def step(self, group, loss, optimizer, parameters, grad_clip, check_finite=True):
        return self._step(group, [loss], [optimizer], [parameters], grad_clip, check_finite)[0]

    def step_actors(self, losses, optimizers, parameter_groups, grad_clip, check_finite=True):
        return self._step("actor", losses, optimizers, parameter_groups, grad_clip, check_finite)

    def _step(self, group, losses, optimizers, parameter_groups, grad_clip, check_finite):
        if group not in self.counts:
            raise ValueError(f"Unknown precision optimizer group: {group}")
        losses, optimizers = list(losses), list(optimizers)
        parameters = [list(values) for values in parameter_groups]
        if not losses or len(losses) != len(optimizers) or len(losses) != len(parameters):
            raise ValueError("Precision optimizer losses and parameter groups must correspond")
        count = self.counts[group]
        if not self.enabled:
            if group == "actor":
                norms = step_actor_optimizers(losses, optimizers, parameters, grad_clip, check_finite)
            else:
                norms = [step_optimizer(loss, optimizer, values, grad_clip, check_finite)
                         for loss, optimizer, values in zip(losses, optimizers, parameters)]
            count["attempted"] += 1
            count["applied"] += 1
            return norms
        with torch.autocast(self.device.type, enabled=False):
            # A non-finite forward loss is a model failure, not a scaling overflow.
            # Always check it: a loss can be infinite while its derivative is finite.
            if not torch.isfinite(torch.stack([loss.detach() for loss in losses])).all():
                raise FloatingPointError("Non-finite learner loss")
            scaler = self.scalers[group]
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            count["attempted"] += 1
            torch.autograd.backward([scaler.scale(loss) for loss in losses])
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
            gradients = [[p.grad for p in values if p.grad is not None] for values in parameters]
            norms = [_total_norm(values)
                     for values in gradients]
            # No actor step, including Adam clocks/moments, occurs before this
            # check has accepted every actor's independently clipped gradient.
            if torch.isfinite(torch.stack(norms)).all():
                for values, norm in zip(parameters, norms):
                    clip = getattr(torch.nn.utils, "clip_grads_with_norm_", None)
                    if clip is None:
                        torch.nn.utils.clip_grad_norm_(values, grad_clip)
                    else:
                        clip(values, grad_clip, norm)
                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()
                count["applied"] += 1
            else:
                before = scaler.get_scale()
                # unscale_ already recorded inf/nan checks for every optimizer.
                # Normal overflow therefore resets GradScaler's growth clock.
                scaler.update()
                if scaler.get_scale() >= before:
                    # An FP32 norm reduction can overflow even with finite
                    # individual gradients. Its skip needs the same backoff and
                    # growth reset, using only the public scaler state API.
                    state = scaler.state_dict()
                    state["scale"] = before * state["backoff_factor"]
                    state["_growth_tracker"] = 0
                    scaler.load_state_dict(state)
                count["skipped"] += 1
                observer = self.overflow_observer
                if observer is not None:
                    try:
                        finite = all(bool(torch.isfinite(g).all()) for values in gradients for g in values)
                        observer.on_skip({"group": group, "scale_before": before,
                            "scale_after": scaler.get_scale(), "raw_gradients_finite": finite,
                            "category": "norm_reduction_overflow" if finite else "nonfinite_gradients",
                            "attempted": count["attempted"], "skipped": count["skipped"],
                            "losses": [float(loss.detach()) for loss in losses]}, parameters)
                    except Exception:
                        self.observer_errors += 1
            return norms

    def state_dict(self):
        return {"mode": self.mode,
                "scalers": {group: scaler.state_dict() for group, scaler in self.scalers.items()},
                "counts": {group: dict(values) for group, values in self.counts.items()}}

    def load_state_dict(self, state):
        validate_precision_state(state, self.mode, self.groups)
        if state is None:
            return
        for group, scaler in self.scalers.items():
            scaler.load_state_dict(state["scalers"][group])
        self.counts = {group: dict(values) for group, values in state["counts"].items()}

    def metrics(self, groups=None):
        result = {"amp/enabled": int(self.enabled)}
        if self.observer_errors:
            result["amp/observer_errors"] = self.observer_errors
        for group in (self.groups if groups is None else groups):
            values = self.counts[group]
            result.update({f"amp/{group}/{name}": value for name, value in values.items()})
            result[f"amp/{group}/scale"] = self.scalers[group].get_scale() if self.enabled else 1.0
        return result


class CompileStartupGate:
    """Serialize initial critic/policy updates, then permit concurrent execution."""
    def __init__(self):
        self.lock, self.warmed = Lock(), set()

    @contextmanager
    def enter(self, group):
        if len(self.warmed) < 2:
            with self.lock:
                yield True
                self.warmed.add(group)
        else:
            yield False


def autocast_update(method):
    """Enter AMP in the calling worker thread; tracing gates only the update."""
    group = "critic" if method.__name__ == "update_critic" else "policy"

    @wraps(method)
    def update(self, *args, **kwargs):
        gate = getattr(self, "_compile_startup_gate", None)
        guard = gate.enter(group) if gate is not None else nullcontext()
        observer = self.precision.overflow_observer
        observe = getattr(observer, "update_context", None)
        context = observe(self, args[0] if args else kwargs["batch"], group) if observe else nullcontext()
        with guard, self.precision.autocast(), context:
            result = method(self, *args, **kwargs)
            if self.precision.enabled and isinstance(result, dict) and result:
                fp16 = (torch.is_autocast_enabled(self.device.type)
                        and torch.get_autocast_dtype(self.device.type) == torch.float16)
                owned = ("critic",) if group == "critic" else tuple(g for g in self.precision.groups if g != "critic")
                return {**result, **self.precision.metrics(owned), "amp/autocast_fp16": int(fp16)}
            return result
    return update
