"""Q1/A* 4-Tier Telemetry: W&B Online (Resume-Safe) + Hugging Face Forensic & Incremental Backup Store (Streamfix v3).

Key Q1 paper telemetry fixes:
- Zero-delivery hygiene: `mean_aoi_s`, `mean_aoi_iqm_s`, `p95_aoi_s`, `multihop_delivery_ratio_pct`,
  and `energy_kj_per_delivered_target` are `None` (JSON `null`) when 0 targets are delivered.
- No `1e-8` relative delta blow-up in `sim_to_net_gap`: `delta_rel_pct_*` is `None` when baseline == 0.
- No degenerate 0-width 95% CI when `n < 2`.
- Strict RFC 8259 JSON (`allow_nan=False`), dynamic task key from `config["world"]`, and resume-safe
  append-mode `train_history.jsonl.gz` + W&B `resume="allow"`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from uav_network.experiments.dual_fidelity_eval import EpisodeKPI, aggregate_kpis


def iqm_stratified_bootstrap_ci(
    values: np.ndarray, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 42
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute IQM and 95% Stratified Bootstrap CI (matches Google DeepMind rliable)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None, None

    x = np.atleast_2d(arr)
    s_seeds, t_tasks = x.shape
    flat = np.sort(x.ravel())
    k = int(0.25 * flat.size)
    trimmed = flat[k : flat.size - k] if flat.size >= 4 else flat
    point_iqm = float(trimmed.mean())
    if flat.size < 2:
        return point_iqm, None, None

    rng = np.random.default_rng(seed)
    if s_seeds > 1:
        idx = rng.integers(0, s_seeds, size=(n_resamples, s_seeds, t_tasks))
        boot = np.take_along_axis(x[None, :, :], idx, axis=1).reshape(n_resamples, s_seeds * t_tasks)
    else:
        idx = rng.integers(0, t_tasks, size=(n_resamples, t_tasks))
        boot = x[0][idx]

    boot.sort(axis=1)
    k_b = int(0.25 * boot.shape[1])
    boot_trimmed = boot[:, k_b : boot.shape[1] - k_b] if boot.shape[1] >= 4 else boot
    boot_iqm = boot_trimmed.mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot_iqm, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return point_iqm, float(ci_lo), float(ci_hi)


def summarize_eval_episodes(episodes: List[EpisodeKPI], seed: int = 42) -> Dict[str, Optional[float]]:
    """Compute Mean, Std, Median, IQM, 95% CI, and IEEE Q1 network KPIs."""
    agg = aggregate_kpis(episodes)
    returns = np.array([ep.episode_return for ep in episodes], dtype=np.float64)
    deliv_pct = np.array([100.0 * ep.targets_delivered / max(ep.total_targets, 1) for ep in episodes], dtype=np.float64)
    aoi_vals = np.array(
        [float(ep.mean_aoi_s) for ep in episodes if ep.targets_delivered > 0 and ep.mean_aoi_s is not None],
        dtype=np.float64,
    )

    ret_iqm, ret_lo, ret_hi = iqm_stratified_bootstrap_ci(returns, seed=seed)
    del_iqm, del_lo, del_hi = iqm_stratified_bootstrap_ci(deliv_pct, seed=seed)
    aoi_iqm, aoi_lo, aoi_hi = iqm_stratified_bootstrap_ci(aoi_vals, seed=seed)

    agg.update({
        "return_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else None,
        "return_median": float(np.median(returns)),
        "return_iqm": ret_iqm,
        "return_iqm_ci_low": ret_lo,
        "return_iqm_ci_high": ret_hi,
        "delivery_ratio_iqm_pct": del_iqm,
        "delivery_ratio_ci_low_pct": del_lo,
        "delivery_ratio_ci_high_pct": del_hi,
        "mean_aoi_iqm_s": aoi_iqm,
        "mean_aoi_ci_low_s": aoi_lo,
        "mean_aoi_ci_high_s": aoi_hi,
    })
    return agg


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class SlottedTelemetryLogger:
    """Unified W&B Online (9 prefixes, resume-safe) + Hugging Face Forensic & Incremental Backup logger."""

    def __init__(
        self,
        output_dir: str | Path,
        config: Dict[str, Any],
        algorithm: str,
        seed: int,
        wandb_project: str = "uav_search_target",
        wandb_entity: Optional[str] = None,
        wandb_mode: str = "disabled",
        hf_repo_id: Optional[str] = None,
        run_name: Optional[str] = None,
        wandb_run_id: Optional[str] = None,
        resume: bool = False,
        gpu_warp_verified: bool = False,
        vendor_uavnetsim_verified: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.algorithm = algorithm
        self.seed = int(seed)
        self.hf_repo_id = hf_repo_id
        self.gpu_warp_verified = bool(gpu_warp_verified)
        self.vendor_uavnetsim_verified = bool(vendor_uavnetsim_verified)

        w_cfg = config.get("world", {})
        self._task_key = (
            f"{int(w_cfg.get('num_uavs', 6))}uav_"
            f"{int(w_cfg.get('num_obstacles', 22))}bldg_"
            f"{int(w_cfg.get('num_targets', 25))}tgt"
        )
        self._slotted_env_key = "warp_slotted_fp32" if self.gpu_warp_verified else "numpy_slotted_reference"
        self._csma_env_key = "uavnetsim_full_csma" if self.vendor_uavnetsim_verified else "csma_event_proxy"
        self._eval_prefix = "eval_slotted" if self.gpu_warp_verified else "eval_reference_slotted"
        self._test_slot_prefix = "test_slotted" if self.gpu_warp_verified else "test_reference_slotted"
        self._test_csma_prefix = "test_full_uavnetsim" if self.vendor_uavnetsim_verified else "test_reference_csma_proxy"

        self.PREFIXES = (
            "train/*",
            "reward/*",
            "network/*",
            "safety/*",
            "learner/*",
            "sys/*",
            f"{self._eval_prefix}/*",
            f"{self._test_slot_prefix}/*",
            f"{self._test_csma_prefix}/*",
            "sim_to_net_gap/*",
        )

        prov_path = self.output_dir / "provenance.json"
        prior_prov: Dict[str, Any] = {}
        if resume and prov_path.exists():
            try:
                prior_prov = json.loads(prov_path.read_text(encoding="utf-8"))
            except Exception:
                prior_prov = {}

        utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_name or prior_prov.get("run_id") or f"slotted_{algorithm}_s{seed}_{utc}_{uuid.uuid4().hex[:8]}"
        self.wandb_run_id = wandb_run_id or prior_prov.get("wandb_run_id") or uuid.uuid4().hex[:12]

        hist_path = self.output_dir / "train_history.jsonl.gz"
        open_mode = "at" if (resume and hist_path.exists()) else "wt"
        self._train_log = gzip.open(hist_path, open_mode, encoding="utf-8")
        self._train_log_count = 0
        self._wandb = None
        self._run = None

        me_path = self.output_dir / "marl_eval_results.json"
        if resume and me_path.exists():
            try:
                self._marl_eval = json.loads(me_path.read_text(encoding="utf-8"))
            except Exception:
                self._marl_eval = {}
        else:
            self._marl_eval = {}

        for ek in (self._slotted_env_key, self._csma_env_key):
            self._marl_eval.setdefault(ek, {}).setdefault(self._task_key, {}).setdefault(
                f"vanilla_{algorithm}", {}
            ).setdefault(f"seed_{seed}", {})

        if wandb_mode != "disabled":
            try:
                import wandb
                self._wandb = wandb
                tags = [
                    "warp-slotted-fp32" if self.gpu_warp_verified else "numpy-slotted-reference",
                    "uavnetsim-csma" if self.vendor_uavnetsim_verified else "csma-proxy",
                    "ds-aoi",
                    f"vanilla-{algorithm}",
                    "shared-onehot",
                ]
                self._run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    id=self.wandb_run_id,
                    resume="allow",
                    name=self.run_id,
                    config=config,
                    mode=wandb_mode,
                    tags=tags,
                )
                wandb.define_metric("global_step")
                for pref in self.PREFIXES:
                    wandb.define_metric(pref, step_metric="global_step")
            except ImportError as exc:
                self._train_log.close()
                raise RuntimeError(
                    "W&B requested but wandb is not installed; install it or use --wandb-mode disabled"
                ) from exc

        self._write_provenance()

    def _write_provenance(self) -> None:
        prov = {
            "run_id": self.run_id,
            "wandb_run_id": self.wandb_run_id,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "gpu_warp_verified": self.gpu_warp_verified,
            "vendor_uavnetsim_verified": self.vendor_uavnetsim_verified,
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
        (self.output_dir / "provenance.json").write_text(json.dumps(prov, indent=2, allow_nan=False), encoding="utf-8")
        (self.output_dir / "config_resolved.json").write_text(json.dumps(self.config, indent=2, allow_nan=False), encoding="utf-8")

    def log_train_step(self, global_step: int, scalars: Dict[str, float]) -> None:
        payload = {"global_step": int(global_step), **{k: v for k, v in scalars.items() if v is not None}}
        if self._train_log is None:
            raise RuntimeError("Training logger already closed")
        self._train_log.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        self._train_log_count += 1
        if self._train_log_count % 5 == 0:
            self._train_log.flush()
        if self._wandb is not None:
            self._wandb.log(payload)

    def log_periodic_eval(self, global_step: int, episodes: List[EpisodeKPI]) -> Dict[str, Optional[float]]:
        summary = summarize_eval_episodes(episodes, seed=self.seed)
        if self._wandb is not None:
            payload: Dict[str, Any] = {"global_step": int(global_step)}
            for k, v in summary.items():
                if v is not None and math.isfinite(v):
                    payload[f"{self._eval_prefix}/{k}"] = v
            self._wandb.log(payload)

        node = self._marl_eval[self._slotted_env_key][self._task_key][f"vanilla_{self.algorithm}"][f"seed_{self.seed}"]
        node[f"step_{int(global_step)}"] = {
            "episode_return": [ep.episode_return for ep in episodes],
            "delivery_ratio_pct": [100.0 * ep.targets_delivered / max(ep.total_targets, 1) for ep in episodes],
            "multihop_ratio_pct": [
                (100.0 * ep.multihop_deliveries / ep.targets_delivered) if ep.targets_delivered > 0 else None
                for ep in episodes
            ],
            "mean_aoi_s": [ep.mean_aoi_s for ep in episodes],
            "p95_aoi_s": [ep.p95_aoi_s for ep in episodes],
        }
        (self.output_dir / "marl_eval_results.json").write_text(
            json.dumps(self._marl_eval, indent=2, allow_nan=False), encoding="utf-8"
        )
        return summary

    def upload_checkpoint_snapshot(self, max_retries: int = 3) -> Dict[str, Any]:
        """Incremental mid-run backup to Hugging Face Dataset so Kaggle 12h timeout never loses progress."""
        if self._train_log is not None:
            self._train_log.flush()
        if not (self.hf_repo_id and os.environ.get("HF_TOKEN")):
            return {"hf_uploaded": False, "hf_note": "HF_TOKEN or hf_repo_id not configured"}
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        last_err = None
        for attempt in range(max_retries):
            try:
                api.upload_folder(
                    folder_path=str(self.output_dir),
                    path_in_repo=f"runs/{self.run_id}",
                    repo_id=self.hf_repo_id,
                    repo_type="dataset",
                )
                return {"hf_uploaded": True, "hf_path": f"runs/{self.run_id}"}
            except Exception as exc:
                last_err = str(exc)
                time.sleep(2.0 * (attempt + 1))
        return {"hf_uploaded": False, "hf_note": f"upload failed after {max_retries} attempts: {last_err}"}

    def log_final_dual_fidelity_test(
        self,
        global_step: int,
        slotted_eps: List[EpisodeKPI],
        csma_eps: List[EpisodeKPI],
    ) -> Dict[str, Any]:
        slot_sum = summarize_eval_episodes(slotted_eps, seed=self.seed)
        csma_sum = summarize_eval_episodes(csma_eps, seed=self.seed)

        gap_keys = (
            "return_iqm",
            "delivery_ratio_iqm_pct",
            "multihop_delivery_ratio_pct",
            "mean_aoi_iqm_s",
            "p95_aoi_s",
            "mean_mac_collisions",
            "mean_propulsion_kj",
            "energy_kj_per_delivered_target",
        )
        gap: Dict[str, Optional[float]] = {}
        for key in gap_keys:
            s_val, c_val = slot_sum.get(key), csma_sum.get(key)
            if s_val is None or c_val is None or not (math.isfinite(s_val) and math.isfinite(c_val)):
                gap[f"delta_abs_{key}"] = None
                gap[f"delta_rel_pct_{key}"] = None
            else:
                delta_abs = float(c_val - s_val)
                gap[f"delta_abs_{key}"] = delta_abs
                gap[f"delta_rel_pct_{key}"] = (100.0 * delta_abs / abs(s_val)) if abs(s_val) > 1e-9 else None

        for env_key, eps in ((self._slotted_env_key, slotted_eps), (self._csma_env_key, csma_eps)):
            node = self._marl_eval[env_key][self._task_key][f"vanilla_{self.algorithm}"][f"seed_{self.seed}"]
            node["absolute_metrics"] = {
                "episode_return": [ep.episode_return for ep in eps],
                "delivery_ratio_pct": [100.0 * ep.targets_delivered / max(ep.total_targets, 1) for ep in eps],
                "multihop_ratio_pct": [
                    (100.0 * ep.multihop_deliveries / ep.targets_delivered) if ep.targets_delivered > 0 else None
                    for ep in eps
                ],
                "mean_aoi_s": [ep.mean_aoi_s for ep in eps],
                "p95_aoi_s": [ep.p95_aoi_s for ep in eps],
            }

        slot_gz = "test_slotted_episodes.jsonl.gz" if self.gpu_warp_verified else "test_reference_slotted_episodes.jsonl.gz"
        csma_gz = "test_full_uavnetsim_episodes.jsonl.gz" if self.vendor_uavnetsim_verified else "test_reference_csma_proxy_episodes.jsonl.gz"
        self._write_episodes_gz(self.output_dir / slot_gz, slotted_eps)
        self._write_episodes_gz(self.output_dir / csma_gz, csma_eps)
        (self.output_dir / "marl_eval_results.json").write_text(
            json.dumps(self._marl_eval, indent=2, allow_nan=False), encoding="utf-8"
        )
        full_report = {
            self._test_slot_prefix: slot_sum,
            self._test_csma_prefix: csma_sum,
            "sim_to_net_gap": gap,
            "gpu_warp_verified": self.gpu_warp_verified,
            "vendor_uavnetsim_verified": self.vendor_uavnetsim_verified,
        }
        if not (self.gpu_warp_verified and self.vendor_uavnetsim_verified):
            full_report["fidelity_warning"] = (
                "Reference/proxy comparison mode — not both GPU Warp and vendored UavNetSim."
            )
        (self.output_dir / "summary_metrics.json").write_text(
            json.dumps(full_report, indent=2, allow_nan=False), encoding="utf-8"
        )

        if self._wandb is not None:
            payload: Dict[str, Any] = {"global_step": int(global_step)}
            for k, v in slot_sum.items():
                if v is not None and math.isfinite(v):
                    payload[f"{self._test_slot_prefix}/{k}"] = v
            for k, v in csma_sum.items():
                if v is not None and math.isfinite(v):
                    payload[f"{self._test_csma_prefix}/{k}"] = v
            for k, v in gap.items():
                if v is not None and math.isfinite(v):
                    payload[f"sim_to_net_gap/{k}"] = v
            col_a = "Warp_Slotted_M4" if self.gpu_warp_verified else "NumPy_Slotted_Reference"
            col_b = "Full_UavNetSim_CSMA" if self.vendor_uavnetsim_verified else "Event_Driven_CSMA_Proxy"
            table = self._wandb.Table(columns=["Metric", col_a, col_b, "Delta_Abs", "Delta_Rel_Pct"])
            for k in gap_keys:
                table.add_data(k, slot_sum.get(k), csma_sum.get(k), gap.get(f"delta_abs_{k}"), gap.get(f"delta_rel_pct_{k}"))
            payload["sim_to_net_gap/comparison_table"] = table
            self._wandb.log(payload)

        return full_report

    @staticmethod
    def _write_episodes_gz(path: Path, episodes: List[EpisodeKPI]) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as gz:
            for idx, ep in enumerate(episodes):
                row = {"episode_index": idx, **asdict(ep)}
                gz.write(json.dumps(row, allow_nan=False) + "\n")

    def write_sha256_manifest_and_upload_hf(self) -> Dict[str, Any]:
        """Generate SHA256 manifest of all forensic files and push to Hugging Face if configured."""
        if self._train_log is not None:
            self._train_log.close()
            self._train_log = None
        files = sorted(p for p in self.output_dir.rglob("*") if p.is_file() and p.name != "manifest_sha256.json")
        manifest = {
            "run_id": self.run_id,
            "files": {str(p.relative_to(self.output_dir)): _sha256_file(p) for p in files},
        }
        manifest_path = self.output_dir / "manifest_sha256.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")

        uploaded = False
        hf_note = "no HF repo requested"
        if self.hf_repo_id and os.environ.get("HF_TOKEN"):
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.upload_folder(
                folder_path=str(self.output_dir),
                path_in_repo=f"runs/{self.run_id}",
                repo_id=self.hf_repo_id,
                repo_type="dataset",
            )
            uploaded = True
            hf_note = "uploaded"
        elif self.hf_repo_id:
            hf_note = "HF_TOKEN missing"

        return {
            "manifest_path": str(manifest_path),
            "file_count": len(files),
            "hf_uploaded": uploaded,
            "hf_note": hf_note,
        }

    def close(self) -> None:
        if self._train_log is not None:
            self._train_log.close()
            self._train_log = None
        if self._run is not None:
            self._run.finish()
            self._run = None
