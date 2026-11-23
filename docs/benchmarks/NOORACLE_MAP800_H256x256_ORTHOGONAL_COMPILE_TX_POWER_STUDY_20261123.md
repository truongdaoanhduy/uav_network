# NoOracle adaptive altitude | map800 × 800, H256×256, Orthogonal, `torch.compile`

Created 2026-11-23 from inspected and verified source commit `f14d525c432fbac2349897888bf7e72ebe29ff7f` on a **new isolated Git worktree/branch**, without changing historical training runs `18d756ef` or `e079eace`.

## User-requested changes

| Dimension | Previous audited v10 MAP1000 H256×512 | New MAP800 H256×256 |
|---|---:|---:|
| World | 1000×1000 m | **800×800 m** |
| Bayesian cell grid | 40×40 (25m cells) | **40×40 (20m cells)** |
| Coarse map | 5×5 | 5×5 (unchanged) |
| Obstacles, targets, UAV | 25, 35, 6 | 25, 35, 6 |
| Obstacle radius | 30–70 m | **24–56 m** (scales 0.8 with map) |
| Obstacle placement clearance | 10 m | **8 m** (scales 0.8 with map) |
| MATD3 actor/critic hidden | 256×512 | **256×256** |
| Initialization | PyTorch defaults | **Orthogonal** (both Actor, TwinCritic) |
| `runtime.compile_models` | false | **true** (`torch.compile`, Inductor) |
| Camera altitude | 0–150m | 0–150m unchanged |
| Confirmation | positive yes, posterior≥.99, positive LR≥4 | unchanged |
| Network max TX | 0.4 W | **0.4 W baseline**; optional separate 0.8W ablation |
| Training | E4096 / U4 / B8192 / H256×512, 20k episodes | E4096 / U4 / B8192 / **H256×256**, 20k episodes specified but NOT launched |
| Hybrid replay | R1M, CPU75% GPU25% | unchanged |
| Validation/Test config | 100 episodes/checkpoint, 1000 held-out test episodes | unchanged, NOT run on Kaggle for this new model |

**Why cell size changes:** Changing only map size to 800m while retaining 25m cells gives 32×32 belief grid, which violates `belief_coarse_cells=5` divisibility in the actual config validator. To retain the existing observation shape and coarse 5×5 cells, we use 20m cells across 800m (40×40). This is a scientific choice; 800×800 changes the physical area and target density, so old and new task scores are NOT directly apples-to-apples.

**Why obstacle radius/clearance change:** The literal 800m scene with original 30–70m radii and 10m placement clearance failed 16/64 scene seeds with Warp error105 after the configured generator exhausted 4 certified attempts. Scaling obstacle radii to24–56m and clearance8m yielded **1152/1152 valid scene generations** sampled across training, validation and held-out test seed banks, with 35 target cells unique and all25 obstacles retained. We did **not** shorten or fake episodes, or disable collision/occlusion.

## Evidence and provenance

- Added config `configs/matd3-nooracle-adaptive-altitude-lr4-map800-h256x256-orthogonal-compile-20261123.yaml`. Historical configs unchanged.
- Orthogonal initializer exists in `uav_network/algorithms/models.py:initialize_matd3_orthogonal`; applies sqrt2 to ReLU hidden Linear layers, .01 to Actor heads, 1.0 to last Critic layers, zero Linear biases, copies into target Actor/Critic. Used by `MATD3.__init__` when `matd3.initialization=orthogonal`.
- `torch.compile` is invoked from `uav_network/experiments/training.py` when `runtime.compile_models=true`: wraps `critic_values` with PyTorch Inductor and compiles Actor and target Actor `encoder`s. This does **not** compile Warp physics/PHY/MAC and does not guarantee training speed improvement. Actual Inductor compilation + backprop + optimizer, checkpoint and held-out 2-episode CPU evaluation passed on installed PyTorch2.7.1. T4×2 results remain pending.
- Self-contained source-pinned Kaggle T4×2 **bounded pilot**: `notebooks/MATD3-NOORACLE-LR4-MAP800-H256x256-ORTHOGONAL-COMPILE-E4096-U4-B8192-T4X2-BOUNDED-PILOT-20261123.ipynb`, 64 vector steps, 4096 worlds, warmup131072 joint env transitions, batch8192, U4, hybrid replay 75%CPU/25%GPU, seed44. No W&B/HF credentials, no full training or publication. The notebook's pinned source archive was extracted and actual compiled CPU train from unpacked files passed. Future CUDA T4×2 execution requires running the notebook on Kaggle.
- Extra physical scenario stress: real Warp CPU 16 worlds ×1500 steps, **0 internal errors**, 502 true target discoveries and399 deliveries across all worlds, 195 false-confirmed cell hypotheses. These are random, altitude-targeted test actions **NOT trained policy** and must never be used as paper results.
- Selected regression batches: **76 passed/3 skipped** (map, compile, orthodox init, privacy and source archive, checkpoint schema) plus **78 passed/7 warnings** (older train/eval/W&B-HF handoff and network). Total **154 passed, 3 skipped**, zero failures. Not the complete test suite.

## TX power decision: baseline 0.4W, controlled ablation 0.8W

Source: actual `uav_network/communication/link_budget.py` urban NLOS model, 2.4GHz, 22MHz bandwidth, 7dB noise figure, threshold6dB, LOS exponent2, NLOS exponent3 plus12dB. Required transmit power ignoring concurrent interference:

| Link distance | LOS | NLOS urban |
|---:|---:|---:|
| 100m | 0.000177W | 0.2803W |
| 125m | 0.000276W | ~0.5475W |
| 250m | 0.001105W | 4.380W |
| 500m | 0.004421W | 35.04W |

At LOS up to 500m, the existing0.4W cap already exceeds ideal receive power needs. For short obstructed links around125m, max0.8W may enable signals blocked by0.4W. But at NLOS250m even0.8W does not overcome nominal losses. Doubling TX max gives +3dB link budget, but increases battery TX energy and CSMA interference when multiple UAVs transmit concurrently. For 1024-byte report at 2Mbps and PA efficiency0.35, model estimates one-send reference energy ~0.00538J at0.4W and ~0.01033J at0.8W, excluding relays/ACK/retries; flight energy still dominates in this setting. A smaller physical map may make routing easier even without TX changes.

Keep principal training run unchanged at **0.4W**. Separate opt-in ablation `configs/matd3-nooracle-adaptive-altitude-lr4-map800-h256x256-orthogonal-compile-tx08-ablation-20261123.yaml` differs in scientific settings **only** by `network.tx_power_max_w:0.8`; run identity/output path unique. Measure true GCS Delivery%, packet PDR, NLOS failure, radio energy, collision/interference and mean return under identical seed sets before selecting 0.8W.

## Flow and known limits

- Science pipeline unchanged otherwise: reset map -> local observation/Actor actions -> XYZ safety + battery -> camera FoV/Pd/Pf/occlusion -> Bayesian 2D belief -> positive LR≥4 plus posterior≥.99 -> cell-ID report -> Warp PHY/MAC/GCS relay -> reward -> replay MATD3 updates -> snapshots. A checkpoint observation version10 is stored and validated.
- `tracking.deferred_gpu_eval_enabled=true` and `runtime.eval_episodes=1000` in production YAML. The **bounded pilot deliberately disables tracking, GPU Validation and Final Test**. The earlier one-session 8192-world-per-GPU Eval/Test scheduling is not automatically invoked by this pilot. Do not claim a complete 20k Train/Val/Test run or full 8192-GPU certification.
- Preserve old source / separate configuration and output directory. Because map, grid cell width, obstacle geometry, network H and orthogonal scheme changed, **train from scratch**; do not reuse e079eace, 18d756ef or v10 MAP1000 Actor weights.
- On Kaggle choose **T4×2** explicitly. `torch.compile` first-run JIT cost may be significant and GPU VRAM/throughput unverified; compare timed steady-state with same science before claiming a speedup.
