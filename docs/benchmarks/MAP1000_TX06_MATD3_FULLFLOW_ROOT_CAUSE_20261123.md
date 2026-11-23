# MAP1000 / TX0.6 / MATD3 — End-to-end functional audit & speed root cause

**2026-11-23, source SHA** `fa27380bf0118b2c244af3513cdcb8aedc681728`; reproduction uses the connected Ubuntu sandbox but **this Ubuntu has no NVIDIA driver/GPU**, so all reported NEW measurements are CPU-only. Do not interpret CPU kernel percentages as actual T4 GPU utilization or a successful Kaggle full 20,000-episode job. Existing run checkpoints and data are preserved.

## Public verifiable evidence

- [18d756ef baseline W&B run](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/18d756ef) and exact captured YAML `configs/reference-18d756ef-wandb-resolved-20261121.yaml`.
- [e079eace NoOracle V9 W&B run](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/e079eace).
- [602a1e82 NoOracle MAP800 Posterior-only TX08 W&B run](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/602a1e82).
- [Hugging Face per-run diagnostic archive](https://huggingface.co/datasets/duy95/uav-search-diagnostics/tree/main/runs) and [immutable Actor snapshots](https://huggingface.co/datasets/duy95/uav-search-diagnostics/tree/main/deferred-gpu-eval).
- [WarpDrive JMLR 23(316), 2022](https://www.jmlr.org/papers/v23/22-0185.html); GPU resident end-to-end MARL, not a throughput prediction for this rich PHY/MAC simulator.
- [EnvPool NeurIPS 2022 Datasets & Benchmarks](https://papers.neurips.cc/paper_files/paper/2022/hash/8caaf08e49ddbad6694fae067442ee21-Abstract-Datasets_and_Benchmarks.html); async environment scheduling and variance-aware collector throughput.
- [BenchMARL JMLR 25(217), 2024](https://www.jmlr.org/papers/v25/23-1612.html); control scientific configs when comparing performance and policy quality.
- [FastTD3 GitHub source](https://github.com/younggyoseo/FastTD3); entire replay on GPU to avoid CPU-GPU transfer, but this often requires much more VRAM. Its humanoid task, GPUs and throughput are **not** an apples-to-apples benchmark for six-UAV UavNetSim. Do not claim the 2025 FastTD3 preprint is already a verified Q1 journal paper.
- [PyTorch official pinned-memory & non_blocking copy tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html); manual `.pin_memory()` on the main thread can be blocking and slower; benchmark on real CUDA hardware.

## 1. W&B measured training speed vs completion time

| Run | E / U / B | Transitions | `train/joint_transitions_per_s` | W&B `_runtime` | Estimated training time from total steps/final rate |
|---|---|---:|---:|---:|---:|
| `18d756ef` | 4096 / 4 / 8192 | 27,549,696 | **4,599.758/s** | 8,577s (all phases) | ~1.66h |
| `e079eace` | 4096 / 4 / 8192 | 29,392,896 | 1,981.912/s | 38,744s | ~4.12h |
| `602a1e82` | 4096 / 4 / 8192 | 30,720,000 | **2,493.174/s** | 18,931s | ~3.42h |

**New run is already ~26% faster than V9**, but ~46% lower throughput than 18d baseline. This cannot be attributed solely to architecture: 18d H256×512/default init/compile off, new H256×256/orthogonal/compile on; new NoOracle cell-ID reports, posterior-only and learned TX, target/PHY activity, occlusion and mission diagnostics are different. For *training-only* seconds, the estimates assume the W&B final effective throughput is representative; don't confuse W&B `_runtime` with just optimizer.

`602a1e82` W&B host timing counters: `collector_wait_seconds` 6,881.264s, `learner_seconds` 3,034.034s, `replay_sample_host_seconds` 1,996.615s, `optimizer_update_host_seconds` 1,023.075s. Collector and learner can overlap asynchronously, so these intervals are not exclusive additive CPU/GPU critical-path measurements. Replay sampling alone comprises about **65.8% of learner-side host time**, hence merits pinned-memory, H2D and GPU scatter A/B.

## 2. Definitive environmental kernel location — measured on CPU, not T4

`python -m scripts.profile_nooracle_warp_phases_20261122 --config configs/matd3-nooracle-posterior-only-map1000-h256x256-orthogonal-compile-tx06-fastgrid-20261123.yaml --device cpu --envs 128 --warmup 10 --steps 100`

| Scenario | CPU env TPS | Dominant kernels |
|---|---:|---|
| MAP1000/TX06 full UavNetSim | **1430.89** | `packet_step`: **61.25%** of kernel time, Bayesian `sense_bayesian_indexed`:18.89%, `observe`:15.11% |
| MAP1000/TX06 **network disabled (diagnostic only)** | 3483.50 | Bayesian sense47.41%, observations37.57% |
| Original 18d configuration on current code | 1413.79 | `observe`:67.39%, sensing22.71%, `packet_step`:5.88% |

Do **not** disable networking in training: it changes the scientific task, packet semantics and reward. The diagnostic shows that the **event-driven CSMA/CA UavNetSim is currently the leading CPU kernel cost**, but CPU speeds and kernel % cannot be directly extrapolated to T4×2. The original 18d and new config have equal map1000/grid25/U6/targets35/obstacles25/E4096 and replay75/25, but the NoOracle cell-report mode reserves **140 slots per world vs baseline35**. MAP1000 posterior-only can create many false-positive cell hypotheses; this changes TX/ACK/queue contention and makes `packet_step` a highly variable per-world GPU thread with while-event-loop and heap/queue operations.

For NoOracle, `uav_network/communication/gpu/mac.py:packet_step` is `@wp.kernel` with `e=wp.tid()` (one thread per simulated world); its `while s.event_count[e]>0` can run differing queue lengths/packet collision/retry/event schedules. GPU warp divergence, register pressure and waiting for stragglers are plausible **hypotheses** until measured with `wp.timing_begin` or Nsight on T4. Treat CPU profile + source as evidence of dynamic scheduler overhead, not as proof of GPU bottleneck percent.

## 3. Actual MAP1000/TX06 Warp full-horizon mission stress (no training)

Executed 16 real Warp CPU worlds×1500 ticks each with scripted altitude-targeted actions, complete full UavNetSim and Bayes, seeded400..415, no internal errors; report `/home/aduy/Downloads/MAP1000_POSTERIORONLY_TX06_16WORLD_FULL_HORIZON_STRESS_20261123.json`.

- 535 of 560 true targets detected across worlds; 442 true targets delivered; 393 false-confirmed cells in total.
- World maximum observed altitude150m, 140 report slot capacity, max used hypotheses77, avg58 per world. No error115/116 in these seeds.
- The false confirmations are a **scientific quality risk** from Bayesian positive-only threshold0.99 with high-altitude Pf as large as0.30; not evidence of code bug or truth leak. In the simulator, ground truth is used to generate the simulated sensor sample and for diagnostics/metrics, NOT to gate the NoOracle report ID. Suppressing these reports would change the user's explicit science assumptions and must be tested/declared as a new ablation, not secretly inserted as a speed improvement.

## 4. Complete CPU Train -> all Actor snapshot Val -> Best -> heldout Test path

New reproducible script `scripts/audit_map1000_tx06_cpu_fullflow_20261123.py` executed against trusted source, no credential access or remote upload. Output `/home/aduy/Downloads/MAP1000_TX06_CPU_FULLFLOW_SMOKE_20261123/fullflow_report.json`.

- **Real MATD3+Warp** CPU train6 short 12-step episodes (2 worlds, H32×32, B16, one update/step), retained7 snapshots at cadence6 short-smoke vector steps.
- Same two fixed Validation seeds for EACH of seven Actor snapshots, exact science config (but bounded training settings); best selected by mission success/delivery/discovery/return. Independently seeded heldout Test2 completed. All required files written, no runtime errors.
- **Not a 20k E4096 GPU fullflow.** Short episodes result discovery/delivery0, which is not predictive of a trained final model. Full Kaggle T4x2 run was NOT launched from this Ubuntu because there is no NVIDIA driver/active Kaggle job tool connection available in this session.

## 5. Fix candidates — correctness first; no unverified speed claims

**Candidate A, sparse 140-slot prefix:** separate local branch `experiment/map1000-tx06-fastgrid-sparse-report-prefix-20261123`. Exact bitwise hashes across **167 complete WorldState arrays at 13 snapshots including one reset** in 16 worlds×1500 steps were identical. However 16-world full-horizon CPU old46.22s vs new48.50s (slower); 128-world×100 CPU quick benchmark old1430.89 vs new1462.21 TPS (+2.2%). No stable CPU speed gain. **Not promoted into production**; requires T4 GPU profiling and benchmark to accept.

**Candidate B, one-time H2D indices per B8192 replay sample:** separate local branch `experiment/map1000-tx06-replay-index-cache-20261123`. Tests22/22 passed including uniform replay RNG/roundtrip. Full B8192 MAP1000 real field samples (145.3 MB per minibatch including all eight stored fields) had matching complete SHA256 tensors and RNG/seen state. Alternating CPU A/B median old98.76ms vs new101.08ms (**-2.3%** on CPU); not evidence of GPU benefit. On actual CUDA, eliminating 8 repeated row-index transfers per batch may help, but no GPU profiling here. **Not promoted into production**.

Existing safe FASTGRID production improvements (already in current notebook): **only periodic Actor snapshots every100 vector steps** without extra training metric-best policies, `UAV_FAST_GRAD_CLIP=true`, `UAV_BATCH_SCALAR_METRICS=true`, sparse `UAV_CUDA_EVENT_PROFILE=true`, deep HF forensic logs retained (`UAV_FAST_TRAINING=false`). Old 602a1e82 manifest contains 254 snapshots = 76 periodic +178 extra offgrid; periodic-only reduces number of policies requiring Val100 from254 to around76 (~70.1% fewer evaluated Actor policies) without reducing the 100 heldout scenarios per periodic Actor. This is an **evaluation cost win**, not a validated training TPS speedup.

## 6. Ranked safe engineering actions to approach 18d speed, WITH actual T4 validation

1. **Collect T4 collector kernel timing first.** Run `python -m scripts.profile_nooracle_warp_phases_20261122 --config configs/matd3-nooracle-posterior-only-map1000-h256x256-orthogonal-compile-tx06-fastgrid-20261123.yaml --device cuda --envs 4096 --warmup 20 --steps 250`, on SAME T4 GPU, comparing current config and 18d. Report per-kernel ms and GPU utilization. Do not use CPU profiler proportions for CUDA optimization decisions. `packet_step` per-world event loop and packet-count distributions are the first candidate for strict event-preserving optimization.
2. **Make realistic equivalent traffic benchmarks.** Use identical traffic patterns for PHY/MAC optimizations, log per-step event_count, generated ACKs, queue retries, contested channels, total reports and real GCS delivery, and require equality of all simulator state arrays/seeds/rewards after any change; optimization cannot drop events, RF power, delay or falsify confirmation.
3. **Replay A/B on real T4 learner GPU:** current `.pin_memory().to(cuda, non_blocking=True)` per field vs preallocated bounded `UAV_HYBRID_PINNED_STAGING=1` (already in source), then candidate cached index transfers; verify bitwise minibatch, timing, `torch.cuda.max_memory_allocated`, H2D bandwidth and loss/Actor weight parity, on B8192 E4096 and fixed 75/25 split. PyTorch warns manual main-thread `.pin_memory()` may block and be slower; don't enable staged without stable benefit. Network bandwidth is not automatically a bottleneck explanation by itself; replay CPU `index_select` can also dominate.
4. **Control experiments:** identical E4096/U4/B8192, seed44, 40x40 map, target distribution, NoOracle and TX06; 3 paired warm GPU runs per change after warmup131072 transitions, report **environment TPS** and code commit, not W&B `runtime` alone. Re-measure policy discovery, delivery, AUC, false-confirmation and return so faster throughput is not bought by breaking accuracy.
5. **Review original 18d slow/fast mismatch fairly:** original uses oracle target-ID indexing and can have less packet pressure due lower confirmation/delivery; its ~4600TPS may be **unattainable for a more expensive simulation without actual algorithmic acceleration**, so choose target ~3500–4000 TPS as a *goal*, not a demonstrated result. Avoid disabling UavNetSim, truth/NoOracle safeguards, lowering 4096 envs, U4 or B8192 or shrinking time horizon to claim convergence speed.

## Verification summary

| Layer | Current result |
|---|---|
| Whole MAP1000 CPU mission 16×1500 | PASS no simulator errors, 535 targets found, 442 delivered, 393 false confirmed cells |
| CPU end-to-end Train->ActorSnapshots->All-Val->Best->Heldout-Test | PASS (bounded 6/2/2 episodes), including rerun regression with new output path |
| Matd3 MAP1000 TX06 notebook original | 22 checked tests PASS, mandatory Kaggle preflight files retained and verified |
| Sparse prefix 16×1500 bitwise trajectory parity | PASS, but no stable speedup, NOT merged |
| Replay B8192 all-fields+RNG parity | PASS, but CPU2.3% regression, NOT merged |
| Actual T4×2 production full 20k E4096/U4/B8192 Val100/Test1000 | **NOT EXECUTED** (no CUDA/Kaggle execution connection) |
| Actual GPU throughput >=4000TPS | **NOT MEASURED** |

### User-facing files

- `scripts/audit_map1000_tx06_cpu_fullflow_20261123.py`: rerunnable CPU bounded full-flow check; use `--output /tmp/a_fresh_directory` (refuses to overwrite prior output).
- `/home/aduy/Downloads/MAP1000_TX06_CPU_FULLFLOW_SMOKE_20261123/fullflow_report.json`.
- `/home/aduy/Downloads/MAP1000_POSTERIORONLY_TX06_16WORLD_FULL_HORIZON_STRESS_20261123.json`.
- `/home/aduy/Downloads/MAP1000_TX06_BEFORE_CPU_WARP_PROFILE_20261123.log` and paired baseline/radio-disabled profiles.
- `/home/aduy/Downloads/MAP1000_TX06_REPLAY_INDEX_CACHE_CPU_AB_20261123.json`.
- Main Kaggle notebook remains the previous SHA-verified `notebooks/MATD3-NOORACLE-POSTERIORONLY-MAP1000x1000-TX06-FASTGRID-T4X2-E4096-U4-B8192-20261123.ipynb` (766,497 bytes, unchanged).

## Final regression after audit script added

`23 passed` across source-bundle Kaggle bootstrap, mandatory preflight, posterior-only MAP1000/TX0.6 science tests, FASTGRID periodic-only, real CPU bounded train/Val/Test, and fake W&B/HF logging; full detailed CI log in `/home/aduy/Downloads/MAP1000_TX06_FULLFLOW_ROOTCAUSE_REGRESSION_20261123.log`. Run environment has **no NVIDIA driver**; CUDA-only tests cannot be treated as executed.
