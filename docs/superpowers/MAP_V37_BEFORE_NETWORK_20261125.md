# Zhang–Xue v3.7 — map-first validation and freeze checklist

## User-approved work order

1. Finish and test **2D map / sensing / Dempster–Shafer / motion** (this branch).
2. Finish and test **UavNetSim-v1 ↔ vectorized Warp network**, including hidden terminals (Gate 5).
3. **Only then** run MATD3/MASAC training, then evaluation/checkpoints/W&B/HF.

**Do not launch Kaggle training now**. No network changes were made by this map-only patch.

## Exact scenario and newly explicit camera parameter

Current scientific v3.2 world preserved: 1000×1000m, 20×20 cells, 50m cell
side, six UAVs at fixed 100m altitude, 10 targets, 2 no-fly circles
(radius 50m), 2 radio blackout diamonds (Manhattan radius 3 cells),
200 steps, `dt=1s`, `Vmax=50m/s`, min separation 50m,
`Pd=0.85`, `Pf=0.03`, `m_f=m_e=0.8`,
DS threshold `0.95`.

**Change from the v3.5 pilot:** The previous 1-cell sensor was
physically inconsistent with the 90-degree nadir camera inherited from
the existing UAV research design. We now explicitly configure
`camera_fov_degrees=90` and sense all ground-cell **centers** inside
a circular nadir footprint at altitude 100m:

```math
r_{FOV}=h\tan\left(\frac{\theta}{2}\right)
      =100\tan(45^\circ)=100\;\mathrm{m}.
```

No ray/height occlusion is claimed: 2D no-fly circles block motion, not
camera rays; blackout diamonds affect radio only. This 90-degree field
was **not an explicit field of original v3.2 YAML** and must be treated
as an intentional new scenario assumption / scientifically reported
change, not silently relabeled an exact reproduction.

Per timestep, for each visible cell `c` and UAV `u`, one independent
Bernoulli detection uses `Pd` if ground truth contains a target,
else `Pf`; one positive or negative evidence count is added.
Hidden ground truth is never serialized to Actor/Critic observations.
`uint16` CPU counts and bounded INT32 Warp counts (cap 65535) retain
headroom for episodes. Dynamic sensing correlation is NOT modeled.

## Warp map backend implemented

- `uav_network/environment/zhang_xue_map_geometry.py`: one-source
  deterministic scene generation for CPU and batch resets; returns
  `positions[E,6,2]`, `target_mask[E,400]`,
  `no_fly_zones[E,2,3]`, `blackout_zones[E,2,2]`.
  Network objects are never allocated during batch sampling.
- `uav_network/environment/gpu/zhang_xue_map_warp.py`:
  actual Warp 1.18.0 **CPU** kernels with independent env/user/cell
  threads, optional exact supplied RNG uniforms for verification, or
  device-local Warp seeded RNG. Persistent `own_pos/own_neg/m_f/m_u`
  arrays `[E,6,400]`; team-only `m_f/m_u` over sums of the **own**
  evidence, computed separately for reward/diagnostic.
  Near-equal counts evaluated in stable FP32 anchored log-space DS.
- `uav_network/environment/gpu/zhang_xue_motion_warp.py`:
  separate Warp CPU/GPU kernel for 1km boundary, no-fly **whole
  segment** collision, and inter-UAV minimum separation. One Warp
  thread per env resolves UAVs in fixed ID order like the original
  CPU environment, preserving deterministic collision priority.
- `uav_network/environment/zhang_xue_coop.py` now calls shared
  scene generation and senses the full footprint, rather than only
  the center cell. The old 800m/25T Warp environment stays untouched.

## Reproducible evidence and assertions

Run from repository root (no Kaggle or trainer):

```bash
python -m pytest -q \
  tests/test_zhang_xue_v37_map_first.py \
  tests/test_zhang_xue_v37_scene_batch.py \
  tests/test_zhang_xue_v37_warp_map.py \
  tests/test_zhang_xue_v37_warp_motion.py \
  tests/test_zhang_xue_production_integration.py
```

Observed: **29/29 pass** on local Ubuntu Warp CPU.

- Scene seed parity: same positions/targets/no-fly/blackout bitwise
  between CPU and batch across six known seeds; 1024 independent seeds
  deterministic, 10 targets each.
- Camera: all and only 50m ground-cell centers inside 100m footprint.
- DS: 16 worlds × 6 UAV × 400 cells, four sequential steps, exact
  evidence counts and local/team DS masses compared with independent
  NumPy formulas at absolute 2e-6.
- Sensor statistics: independent seeded Warp RNG across 1024 worlds,
  measured conditional `Pd` and `Pf` within stated statistical
  tolerances. Clean `reset()` restores unknown masses to 1.
- Motion: 64 random seeded scenes from real CPU environment,
  vectorized Warp motions match CPU positions/velocities and rejection
  counters, plus border/obstacle-edge cases.
- Map-only scalability smoke: 4096 environments, one Warp **CPU**
  sensing+DS step, ~0.196s including sync; this is NOT GPU throughput
  and does not include actor, MAC or replay. Driver shows
  `torch.cuda.is_available()==False` on local machine.

## Full batch map-flow run (additional verification)

The executable scripts/verify_zhang_xue_map_v37.py runs deterministic
scene batch reset, Warp motion safety/boundaries and per-UAV camera/DS
sensing. Actual E4096 x 5 map-step run on Warp CPU (not a trainer):
20,480 env-steps, all 10 targets per world, 1,543,223 evidence
readings, mean distinct-cell coverage 17.7956543%, 13,048 safety
rejections, zero boundary rejections; scene initialization 2.813s,
map motion+sensing+DS 0.992s. No CUDA driver. Evidence JSON:
docs/superpowers/evidence/zhang_xue_v37_4096_map_fullflow_cpu_20261125.json

Map-only tests 29/29 plus one standalone full batch map-flow test
passed. Scoped project regressions: 183 passed, 10 skipped with known
pre-existing historical source-fingerprint assertion excluded.

## Important remaining limits (do not call full training ready)

This is a *tested map component*, **not yet a single fused 4096-env
actor→motion→map→network→reward pipeline**. CPU scene batch generation
remains at reset; actual Warp map sensing+motion are independent
callable kernels. Need integrate with peer packet/ACK fusion and
real CSMA surrogate without leaking truth. No Warp GPU execution or
full episode convergence has been measured.

The existing v3.6 Gate 5C still fails: collision-free mini reference
reports PDR100% on hidden-terminal stress where real UavNetSim-v1
reports 0%; do not equate verified Warp PHY/CCA/backoff *pieces*
with full network parity.

## Rollback

Branch: `experiment/zhang-xue-v37-warp-map-first-20261125`
from v3.6 Gate5F SHA `89c8e938db53806ef714acffccc9bd050dc6f944`.
Commit and push this map milestone before any more network changes.
No main merge or destructive branch operations.
