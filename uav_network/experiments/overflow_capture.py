"""Bounded first-skip evidence and isolated precision replay, never a resume format."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import torch

from .evidence import json_safe


def tensor_stats(value):
    value = torch.as_tensor(value).detach()
    finite = torch.isfinite(value)
    selected = value[finite]
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "finite": bool(finite.all()), "nonfinite_count": int((~finite).sum()),
            "finite_abs_max": float(selected.abs().max()) if selected.numel() else None}


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


class OverflowCapture:
    def __init__(self, output, role):
        if not re.fullmatch(r"[a-z0-9_-]+", role):
            raise ValueError("Invalid overflow capture role")
        self.output, self.role = Path(output), role
        self.contexts, self.attempted, self.events = {}, set(), {}

    @contextmanager
    def update_context(self, learner, batch, phase):
        groups = {"critic"} if phase == "critic" else {"actor"}
        context = None
        if not groups.issubset(self.attempted):
            try:
                rng = {"cpu": torch.get_rng_state()}
                if learner.device.type == "cuda":
                    rng["cuda"] = torch.cuda.get_rng_state(learner.device)
                context = {"learner": learner, "batch": batch, "rng": rng,
                           "precision_before": deepcopy(learner.precision.state_dict())}
                self.contexts[phase] = context
            except Exception:
                learner.precision.observer_errors += 1
        try:
            yield
        finally:
            if context is not None:
                self.contexts.pop(phase, None)

    def on_skip(self, event, parameter_groups):
        group = event["group"]
        if group not in {"critic", "actor"} or group in self.attempted:
            return
        self.attempted.add(group)  # One attempt per group, including failed storage.
        context = self.contexts.get("critic" if group == "critic" else "policy")
        if context is None:
            return
        learner, batch = context["learner"], context["batch"]
        owned = {id(p) for values in parameter_groups for p in values}
        gradients = {name: tensor_stats(p.grad) for name, p in learner.named_parameters()
                     if id(p) in owned and p.grad is not None}
        event = {**event, "gradients": gradients}
        self.events[group] = event
        optimizers = {name: cpu_copy(getattr(learner, name).state_dict())
                      for name in ("critic_optimizer", "alpha_optimizer") if hasattr(learner, name)}
        if hasattr(learner, "actor_optimizers"):
            optimizers["actor_optimizers"] = [cpu_copy(o.state_dict()) for o in learner.actor_optimizers]
        payload = {"schema": 1, "purpose": "overflow_diagnostic_not_resumable",
            "event": event, "algorithm": learner.algorithm, "config": learner.config.to_dict(),
            "source_commit": os.environ.get("UAV_GIT_COMMIT", "local"),
            "source_device": str(learner.device), "role": self.role,
            "source_async": bool(learner.config.pql.enabled and learner.config.pql.mode == "async"),
            "model": cpu_copy(learner.state_dict()),
            "gradient_stage": "unscaled_before_clipping",
            "gradients": {name: cpu_copy(p.grad) for name, p in learner.named_parameters()
                          if id(p) in owned and p.grad is not None},
            "requires_grad": {k: p.requires_grad for k, p in learner.named_parameters()},
            "batch": {k: cpu_copy(torch.as_tensor(v)) for k, v in batch.items()},
            "rng": cpu_copy(context["rng"]), "precision_before": context["precision_before"],
            "optimizers": optimizers, "updates": learner.updates, "actor_updates": learner.actor_updates,
            "input_stats": {k: tensor_stats(v) for k, v in batch.items()}}
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / f"overflow-{self.role}-{group}.pt"
        temporary = path.with_suffix(".pt.tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        metadata = {k: payload[k] for k in ("purpose", "event", "algorithm", "source_commit",
                    "source_device", "role", "updates", "actor_updates", "input_stats")}
        metadata["snapshot_file"] = path.name
        metadata["snapshot_bytes"] = path.stat().st_size
        path.with_suffix(".json").write_text(json.dumps(json_safe(metadata), allow_nan=False, indent=2))


def attach_overflow_capture(learner, output, role):
    observer = OverflowCapture(output, role)
    learner.precision.overflow_observer = observer
    return observer


def replay_overflow(path, device="cpu"):
    """Replay the saved group from identical values/RNG; eager, with no environment/replay refill."""
    from ..algorithms.checkpoints import make_learner
    from ..configuration import Config, _apply
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema") != 1 or saved.get("purpose") != "overflow_diagnostic_not_resumable":
        raise ValueError("Not an overflow diagnostic snapshot")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA replay requested but unavailable; no silent CPU fallback")
    group = saved["event"]["group"]
    cases = {}
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        for mode in ("off", "fp16"):
            config = Config()
            _apply(config, saved["config"])
            config.runtime.amp_mode, config.runtime.compile_models = mode, False
            learner = make_learner(config, saved["batch"]["observation"].shape[-1],
                                   saved["batch"]["state"].shape[-1], device)
            learner.load_state_dict(saved["model"])
            learner.updates, learner.actor_updates = saved["updates"], saved["actor_updates"]
            for name, p in learner.named_parameters():
                p.requires_grad_(saved["requires_grad"][name])
            for name, state in saved["optimizers"].items():
                if name == "actor_optimizers":
                    for optimizer, item in zip(learner.actor_optimizers, state):
                        optimizer.load_state_dict(item)
                else:
                    getattr(learner, name).load_state_dict(state)
            if mode == "fp16":
                learner.precision.load_state_dict(saved["precision_before"])
            before = dict(learner.precision.counts[group])
            events, bad_gradients, hooks = [], [], []
            learner.precision.overflow_observer = SimpleNamespace(on_skip=lambda event, params: events.append(event))
            for name, p in learner.named_parameters():
                if p.requires_grad:
                    def observe(gradient, name=name):
                        if not bool(torch.isfinite(gradient).all()) and not bad_gradients:
                            bad_gradients.append(name)
                        return gradient
                    hooks.append(p.register_hook(observe))
            batch = {k: v.to(device) for k, v in saved["batch"].items()}
            torch.set_rng_state(saved["rng"]["cpu"])
            if device.type == "cuda" and "cuda" in saved["rng"]:
                torch.cuda.set_rng_state(saved["rng"]["cuda"], device)
            error, metrics = None, {}
            try:
                update = learner.update_critic if group == "critic" else learner.update_actor
                metrics = update(batch)
            except (FloatingPointError, RuntimeError) as exception:
                error = type(exception).__name__
            finally:
                for hook in hooks:
                    hook.remove()
            losses = [v for k, v in metrics.items() if k.endswith("loss")]
            cases[mode] = {"error_type": error,
                "losses_finite": error is None and bool(losses) and all(torch.isfinite(torch.tensor(v)) for v in losses),
                "skipped": learner.precision.counts[group]["skipped"] - before["skipped"],
                "applied": learner.precision.counts[group]["applied"] - before["applied"],
                "first_nonfinite_parameter_gradient": next(iter(bad_gradients), None),
                "overflow_events": events, "metrics": metrics,
                "model_finite": all(bool(torch.isfinite(p).all()) for p in learner.parameters())}
    return json_safe({"source_commit": saved["source_commit"], "source_device": saved["source_device"],
        "replay_device": str(device), "group": group, "algorithm": saved["algorithm"],
        "source_compile_models": saved["config"]["runtime"]["compile_models"],
        "compiled_forward_replayed": False, "source_async": saved.get("source_async", False),
        "rng_interleaving_replayed": not saved.get("source_async", False),
        "input_stats": saved["input_stats"], "cases": cases})
