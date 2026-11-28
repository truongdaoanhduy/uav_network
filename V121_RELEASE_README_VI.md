# UAV V121 — RELEASE README (VI)

Ngày: 2026-11-28. Phiên bản nền là ZIP V120. Phát hành **toàn bộ mã nguồn kịch bản**, 30 baseline files byte-identical, config active, test, log/CSV/JSON, registry 82 conflict và SHA256. Không có training hoặc thay đổi learner.

**CẢNH BÁO:** `SCENARIO_FROZEN=false`. Đây là bản sửa/kiểm toán kỹ thuật của **Phase1**, không phải chứng thực toàn bộ khoa học. Không sử dụng số liệu đã sửa sai đơn vị ở V120 làm kết luận mới. Không gọi custom Warp DCF surrogate là UavNetSim upstream SimPy.

## Quan trọng nhất — tài liệu để mở đầu

1. `docs/V121_FULL_SCENARIO_AUDIT_AND_OUTSTANDING_GATES_VI.md` — full tình trạng từng module, math DS, action/reward, provenance, 10 lỗi mới, scientific limits, runbook.
2. `outputs/v121_scenario_audit/V121_CONFLICT_REGISTRY_C01_C82.csv` và `.json` — C01–C72 lịch sử **giữ nguyên**, overlay evidence V121 và C73–C82 mới.
3. `outputs/v121_scenario_audit/V120_to_V121_source.diff` và `V121_CHANGED_FILES_AND_HASHES.json` — diff toàn bộ source vs V120.
4. `outputs/v121_scenario_audit/cuda_uninterrupted_{base,rf}_post_tie.json` cùng `.jsonl`, `cuda_*_post_tie.stdout.log` — 2/2 CUDA PASS hậu sửa.
5. `outputs/v121_scenario_audit/{shield_mixed_cuda,shield_post_tie_cuda}.json` và `*.steps.jsonl`: 16 divergent→0 trên 900 world-step common-prestate.
6. `outputs/v121_scenario_audit/calibrations/v121_corrected_ds_45runs_300steps.json` và `v121_corrected_rf_45runs_300steps.json`; 90 CPU runs, actual command in m/s.
7. `outputs/v121_scenario_audit/constructive_witness/summary.json` với per-world trace; 10/10 mission oracle-validation witness (NOT learned policy).
8. `outputs/v121_scenario_audit/pytest_release_final.stdout.log`, `source_manifest_release.log`, `policy_release.log` — kiểm tra release.
9. `V119_RELEASE_README_VI.md`, `docs/V120_ALL_72_CONFLICTS_ROOT_CAUSE_AND_SOLUTIONS_VI.md` và `outputs/v119_reaudit/`, `outputs/v120_conflict_resolution/` — lịch sử bảo toàn.

### Trạng thái khoa học

C04/C23/C41/C66 (correlation, DS calibration) **OPEN**; C07/C33/C65 (SimPy equivalence) **NO_STRICT_PARITY_CLAIM**; C12/C13/C32 RF & energy model **SYNTHETIC / sensitivity**; C18/C31/ C81 full scenario scientific freeze **OPEN**; shield-only parity is not CPU/GPU full-state parity. Canonical fixed TX power 0.1 W, continuous XY-only, DS only, pure-GCS application data, multi-hop allowed, no Bayes runtime, no hybrid actions. The empirical witness targets are known to the validation planner and are not a baseline MARL result.

### Reproduction

```bash
export PYTHONPATH=research/zhang_xue/src:.
python scripts/maintenance/verify_source_manifests.py --check
python scripts/maintenance/check_project_policy.py --scope full
python -m pytest -q tests/ research/zhang_xue/tests/
python scripts/maintenance/v121_build_conflict_registry.py
python scripts/gpu/v121_uninterrupted_cuda_gate.py --device cuda:0 --envs 4096 --steps 900 --motion random --seed 44 --output outputs/recheck_base.json
python scripts/gpu/v121_uninterrupted_cuda_gate.py --device cuda:0 --envs 4096 --steps 900 --motion random --seed 44 --noise-power-w 4e-11 --output outputs/recheck_rf.json
```

All commands should be run at project root. CUDA requires compatible NVIDIA GPU, Warp, CUDA runtime and NumPy. A reproducible run uses its own log/source hash. Do not run training before Phase1 explicit scientific-scope review.

### Input provenance, download and integrity

V120 source ZIP SHA256: `faa2fa23c12ad10976759dbf374dcd194f4077f59df1f0682226c3022595ef00`. Original V120 release ZIP is nested as `input/V120_ORIGINAL_VERIFIED.zip` at the root of this release archive (outside `uav_network_V121/`), enabling rollback/source comparison. On extraction verify `RELEASE_SHA256.txt` and `SOURCE_SHA256_MANIFEST.json` before further changes. No Git push or mutation of the project master branch was performed.
