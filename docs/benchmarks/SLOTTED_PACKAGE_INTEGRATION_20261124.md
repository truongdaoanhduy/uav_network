# UAV Warp Slotted Vanilla Package: Port and Verification (2026-11-24)

## Provenance and separation

- The public `main` branch was restored **without force push** to the tracked source tree of `a8da8e9e1912dcd0f599ae108b1758824ac24faf` in commit `a65fcf2ebbdd224e2c906c9f3bcf88be452b9c91`.
- The former `main` tree is saved on `backup/main-before-source-restore-20261124` (`65fe29a0a27afd3bc95bda184144f72dc84ab6ca`).
- The original 9 text files from `uav_warp_slotted_vanilla_baselines_package.zip` are preserved *byte-for-byte* as a distinct commit `a963187ae528616df226bd923d3323400768665b` on the experiment branch; generated `__pycache__` and `.pyc` were not imported.
- Source package ZIP SHA-256: `a61ec47f5b58141b6b43ec44d8ca0aefc56b99f6c06c9dfc7b50f3b489402261`.
- Full earlier source-and-logs ZIP SHA-256: `0b93d3289342d97d944cf03adefe660962f8e31d296eee2a5f9f61547f174789`.

## Scientific scenario: NEW baseline, not the old MAP1000 one

The uploaded slotted design targets an 800x800m map, a 20x20 DS grid, 6 fixed-50m-altitude UAVs, 25 static targets, 22 tall buildings, T=550s, fixed TX 0.1W, 4 MAC subslots, shared plain-MLP MATD3/MASAC and a 300k FP16 replay buffer. These values are different from the corrected MAP1000/no-oracle experiment and must not be compared as the same MDP.

## Essential repair differences from uploaded source

1. **Warp compilation:** replaced nonexistent `wp.vec6i` with a declared 6-element `wp.types.vector(length=6, dtype=wp.int32)` type. Confirmed all three supplied kernel definitions compile and launch on CPU in Warp 1.18.
2. **NoOracle:** the DS kernel now enqueues observed **cell hypotheses** rather than reading hidden `target_id` to decide whether a report may exist. A positive measurement and DS thresholds are required; a full queue does not permanently mark the cell as reported. False hypotheses must be able to enter the network.
3. **Scoring boundary:** the MAC kernel deduplicates by **400 cell IDs**, and simulator ground truth is consulted only **after GCS reception** to count true GCS arrivals and false deliveries separately. `step_aoi_score` is for true reports only; the new `step_false_gcs_deliv` output exposes false reports. This changes the MAC signature; callers must allocate its buffers.
4. **Slotted CCA:** on same-slot contention, hearing a lower-index active transmitter defers the node to a subsequent subslot where possible; hidden transmitters may collide. This is a slotted CCA *approximation*, not bit-exact UavNetSim full CSMA/CA.
5. **Queue admission:** a received peer packet is not removed from the sender's queue when the destination's buffer is full; it retries instead.

## Tests executed on the local host

Environment: NVIDIA Warp 1.18.0, PyTorch 2.7.1+cu126, local `cpu` backend (this session's CUDA driver is unavailable). A separate ignored copy of pinned UavNetSim revision `04daafb815eb377409b40b285574eeb62b9a8d58` is required for the existing source-bundle tests; it is not committed or published by this branch.

```bash
cd /home/aduy/Documents/NCKH/uav_network_warp_slotted_20261124
/home/aduy/Documents/NCKH/uav_network/.venv/bin/python -m pytest -q \
  tests/test_map1000_reward_belief_actor_fix_20261123.py \
  tests/test_nooracle_adaptive_privacy_final_audit_20261123.py \
  tests/test_redesign_package.py \
  tests/test_warp_slotted_functional_20261124.py
```

Run the new package's CPU probes independently:

```bash
/home/aduy/Documents/NCKH/uav_network/.venv/bin/python -m pytest -q \
  tests/test_redesign_package.py tests/test_warp_slotted_functional_20261124.py
```

Checks include CPU JIT+launch for PHY, DS and MAC, hypotheses originating from empty cells, false/true GCS delivery credit and deduplication, and an identical-seed CCA vs hidden-terminal comparison.

## NOT complete / do not claim

- The uploaded ZIP contains **building blocks only**, not a wired training-ready 800m/550s GPU environment. `uav_network/experiments/training.py` is still the prior MAP1000 pipeline; these new YAMLs are **not routed into it**.
- Motion, 2.5D occlusion of camera sensing, scene generation and placement, Dec-POMDP observations/masks, CUDA Graph orchestration (motion -> PHY -> DS -> MAC -> reward -> autoreset), full 550-step mission training, and Kaggle notebook integration need new integration code.
- The supplied `dual_fidelity_eval.py` aggregates values from two user-provided evaluators. It **does not itself implement** a Full UavNetSim CSMA/CA evaluator or prove sim-to-eval parity.
- No actual T4 CUDA launch, GPU profiling, 20M-step train, five-seed validation or held-out evaluation has been run. Neither 0% warp divergence nor a throughput uplift can be claimed from CPU tests.
- Paper attribution of the DS formula and rotary-wing energy constants was not independently verified here.

Keep this work on the experimental branch until the real full-flow training/eval simulator and Kaggle T4 validation are implemented and verified. Do not reuse historical MAP1000 checkpoints for the new MDP.
