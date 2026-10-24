# Native vector null telemetry / E2E smoke fix — 2026-10-24

## Real cloud symptom

Real mini GPU->controller E2E smoke `bjkjbhm/uavnet-train-20261024-104832-b89da3` (source `75ce367`) trained MASAC through the fourth requested episode, then failed strict episode JSON serialization. The retained learner metrics and parameters were finite. `continuation.json` reported a secondary ValueError because the learner process had already failed.

The learner log failed at `json.dumps(record, allow_nan=False)` with undefined conditional telemetry represented as NaN. The failed smoke is retained as evidence and is not overwritten.

## Confirmed root cause

`configs/kaggle-smoke.yaml` uses the native backend with Gymnasium vector workers. Gymnasium converts mixed vector-info values like `[1.5, None]` to a floating array `[1.5, nan]`. When one completed worker has a packet/delivery sample and another completed worker has none, `unpack_info()` therefore returned NaN instead of the native simulator's intended `None`.

The exact failing episode had no TX/delivery and finite learner state:

- network stat count = 0, sum = 0, but mean/min/max = NaN after vectorization
- report delivery_count = 0, but delivery_hops_mean / delivered_report_age_mean_s / first_discovery_to_delivery_mean_s = NaN
- optional report timestamp/age slots can likewise be coerced to NaN in mixed batches

Direct native `network_diagnostics()` and `report_diagnostics()` correctly return None for these undefined values. The bug was the vector-info boundary, not PHY/MAC, policy learning, or strict JSON.

## Minimal fix

`uav_network/environment/vector.py` now restores only schema-defined undefined conditionals:

- stat mean/min/max -> None only when count == 0
- delivery conditional means -> None only when delivery_count == 0
- known optional report timestamp/age list NaNs -> None

A positive-count NaN is intentionally left unchanged so strict JSON continues to expose genuine numerical faults. `allow_nan=False` is unchanged.

## Verification

RED: `test_unpack_info_restores_null_conditionals_after_mixed_vectorization` failed on the old behavior (`nan is not None`).

GREEN:

- focused vector/training/handoff/controller suite: 50 passed, 17 subtests
- deterministic local native smoke, 2 env x 4 episodes x 40 steps: completed episode 4 without serialization failure
- full suite: 716 passed, 74 skipped, 8 warnings, 1261 subtests in 267.11s

Warnings are the existing W&B/Pydantic deprecations and Gymnasium pending-step close warning.

## Next gate

Run a new uniquely named real Kaggle E2E smoke from this fixed immutable source. Completion requires GPU training, controller handoff, final CPU evaluation/visualization, W&B publication, and immutable HF publication. Preserve the failed smoke above and all four live 1 KiB production lineages.
