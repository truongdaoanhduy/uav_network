> **Current 2026-11-21 update (supersedes historical verification rule below):** M2 now defaults to Zhang 2024 **fixed-BPA DS (mf=me=.8), per-UAV MAX source-count fusion and immediate m(F)>=.95 confirmation** with independent sensor (`rho=0`). SPRT is opt-in ablation only; Native MATD3 pilot checkpoint semantics were bumped to v2. This file otherwise preserves prior pilot provenance. See `docs/methods/M2_ZHANG_2024_DS_THRESHOLD_FLOW_20261121.md`.

# MATD3-first M2 Target Belief Map 2D: scientific scope and preflight

Date: 2026-11-21. New task: train MATD3 before MASAC, preserving target search, cooperative verification and real packet-level GCS reporting. The original M2 target-only DS map has 1600 cells (40×40 cells ×25m) in a **1000m ×1000m** field; there are 6 flying 3D UAVs, 35 static ground targets and 25 3D physical obstacles for CBF collision safety and sensor occlusion. There is **no obstacle/3D belief map**.

## Not full M2 Warp E4096

This is a **single Native CPU M2+UavNetSim environment with a CUDA:0 MATD3 neural learner** on Kaggle T4 GPU (T4×2 accelerator may provide two GPUs, but only GPU0 runs the learner). The separate Warp E4096 F/E/U projector is numerically validated on T4×2 but is *not* integrated into an equivalent full M2 GPU environment. Do not compare this pilot's throughput directly with E4096 Warp or call it GPU-environment training. MASAC has already passed unit/CPU learner checks but is deliberately not launched until this MATD3 pilot is examined.

## Frozen pilot parameters

- Map 1000m; 6 UAV; 35 targets; 25 obstacles; 1500s mission (dt=1s); XY/Z action Actor-controlled, 5 continuous channels (XYZ, payload kind, TX power), categorical next-hop destination [silent, ordered UAV peers, GCS]; Actor does not get target truth.
- Native UavNetSim, 1024-byte payload, 500m peer/GCS admission, actual MAC/PHY losses and energy, source-provenanced 2D target DS map, 0.55 correlated sensor, candidate threshold0.95, cooperative SPRT α=β=0.05, bidirectional GCS map delta.
- **8 observed target cells/UAV/second**, explicit bounded sensing capacity vs prior physically unbounded all-FoV-cell sampling; coverage/uncertainty rewards and reporting otherwise same.
- MATD3 MLP [256,256], shared Actor/Critic, Actor LayerNorm, PyTorch default init, Adam3e-4, delayed actor (2), twin critics, target smoothing, replay 20,000, batch128, warmup1,024 joint transitions, one learner update per 8 new world transitions, resulting replay draw/transition ratio B/U=128/8=16. This matches the ratio of earlier E4096/U8/B8192 runs but **not** their sample independence or episode budgets.
- Seed44 training scenarios vary by episode (`44+ep-1`); fixed held-out evaluation seeds 2000000044 and 2000000045 after 0,10,20,… completed episodes. Max planned episodes120, within 10.5h train+evaluation time cap. Run can stop earlier without establishing convergence.
- One ground-truth evaluation-only return per step; log training per completed episode and periodic fixed-seed validation by **environment transitions** X-axis. Report return, verified correct GCS delivery recall, delivery AUC, false reports, GCS 2D map coverage, AUPRC, Brier, uncertainty, energy, MAC received packets, packet selection/ACK, Actor action saturation/Z/category/GCS-vs-silent, critic/actor loss/grad norm.
- Policy checkpoints use **M2-specific observation-schema version**, not legacy MATD3.load; inference-only, **NOT exact off-policy training resumes** (replay/Adam RNG not captured). Output CSVs and learning-curve PNG are Kaggle notebook outputs; no W&B keys are embedded or transmitted.

## New blocking bug discovered by 1500-step deep preflight and TDD fix

**Reproduced at about simulator step 400**, source `uav_network/communication/m2_network.py::M2PacketStore.publish`: source-only `_wire_bytes(events)` reserved 128 nominal header bytes, but a growing 4-digit packet serial ID can make the complete serialized envelope+16-byte safety margin 1025B when the advertised batch was exactly 1024B. This raised `ValueError("evidence envelope exceeds packet payload budget")` and would crash Kaggle training deep into an episode.

Added failing test `test_evidence_packetizer_handles_exact_1024_boundary_with_complete_envelope` with serial ID 5000. Changed `M2PacketStore.publish` to greedily build batches using the exact per-message `M2Envelope` `_envelope_wire_size()+16`; when an event no longer fits, it commits the previous packet and starts a new one. Every packet remains within 1024B and sensor events are ingested exactly once. The original packetization helpers remain available for legacy tests.

**Full M2 1500-step scenario after fix:** Passed without exception, no NaN in Actor observations, rewards, target mass maps or mass normalization; battery monotonically decreases; GCS events monotone and cannot exceed observed sources; real UavNetSim network active. Random-action seed44 final GCS target coverage **18.375%**, **9 verified correct target reports**, **1 false**, **552 GCS evidence events**, **2,099 MAC received packets**, with depleted battery by mission end. These random-policy results are a *logic stress test* only, NOT MATD3 performance or evidence of convergence.

## Additional issues investigated and mitigated

- MATD3 training/replay ABI verified with 5-continuous-channel actions, observation255 ×6, state1530, correct post-step terminal transitions and actual Native radios.
- Legacy `MATD3.load` validates legacy observation geometry and would reject M2 255D snapshots. Implemented `load_m2_policy_checkpoint` with explicit M2 5×5 F/E/U+verification schema and M2-config source; save/reload tests verify **bit-for-bit state_dict**. Do not feed these into the old inference loader.
- Random-policy ACK/drop/packet opportunity metrics added to train CSV to distinguish failure of learning from failure of radio connectivity; Actor motion saturation and Z action logged to diagnose tanh collapse documented in historical MATD3 training.
- Dedicated 2-episode small-horizon CPU MATD3 test genuinely creates two M2 environments, applies critic and delayed actor updates, evaluates at seed 2000000044, loads M2 checkpoints, verifies schema/no oracle columns and writes CSVs.
- Full project test suite retains historical failures unrelated to M2; exact comparisons need clean parent and same pinned UavNetSim dependency. Pilot-specific tests must pass before Kaggle upload.

## Remaining science/engineering limitations

1. Not a full Warp E4096 M2 rollout: event/radio simulation is CPU Native and global UavNetSim disallows concurrent instances inside one Python process. Training only 1 environment means temporal sample correlation differs from E4096 and early curves may be noisy.
2. Current Actor packet type uses a discretized scalar continuous action, not a separate categorical Gumbel branch. Its learning effectiveness is unproven; do not call it a converged policy-selection model until evaluation improves over no-learning baseline.
3. The CBF safety layer uses true simulated physical obstacles (not an observed obstacle map), so do not claim obstacle mapping or learned perception-only safety.
4. Pending packet buffer sizes are bounded, but historical DS event maps remain unbounded across one 1500-step episode. This is acceptable for the one-world pilot and **not scalable to E4096 without redesign**.
5. Fixed heldout evaluation seed count2 is a pilot and cannot support statistical significance, even if learning curves look smoother; later compare multiple training seeds and episodic confidence intervals.
6. Network outage and protocol losses can cause GCS metrics to plateau despite improved target observations. The new metrics are required to diagnose this; do not silently optimize reward to hide it.

## Run (on Kaggle)

The private pinned source dataset is `longlonthiu/m2-target-belief-2d-native-matd3-src-20261121` (no credentials in ZIP; pinned UavNetSim revision included). The private MATD3 notebook uses CUDA:0 only and writes JSON, CSV, PNG and `final_policy.pt` under `/kaggle/working/m2_matd3_native_gpu_pilot/`.

Source script: `scripts/research/train_m2_matd3_native_gpu_20261121.py`. CPU 1500-step preflight: `scripts/research/verify_m2_native_1500step_flow_20261121.py`. The user authorized MATD3-first training with Kaggle account1; do not start a parallel MASAC training run until MATD3 is evaluated.
