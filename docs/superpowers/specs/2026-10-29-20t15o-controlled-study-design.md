# 20-target / 15-obstacle controlled 2D study
Date: 2026-10-29. User approved implementing the five audit recommendations and a 20/15 scenario. Existing worktree: /home/aduy/Documents/NCKH/uav_network-public-hf-20261022, branch feat/cpu-replay-20261022, base a470a4022166d94a656b81efa51eebd715ae3c57.

## Goal
Keep decentralized target search -> report custody/relay -> GCS under finite radio range and battery; make a controlled, easier 2D study before future Bayesian altitude sensing / XYZ.
Do not retrofit new science into existing checkpoint lineages. No cancellation or automatic mass submission of cloud jobs.

## Approved approach and implementation choices
Use the existing native/Warp environment and MASAC/MATD3 BASE/PQL flows. Update the canonical 20k profile to 6 UAV, 1000 m, 20 targets, 15 obstacles. Preserve the previous 35/25 profile as a clearly named historical config.
Use network.power_control=link_budget, network.link_budget_margin_db=3.0, with RF power clamp(required nominal power * 10^(3/10), minimum, maximum) computed for the selected local receiver after motion. Geometry/link-budget inputs are the same local-link assumptions already used for admission/beacons. This is not clairvoyant end-to-end routing or a guarantee of SINR under contention.
Default new field power_control=learned for old saved configurations. The actor still learns movement and receiver/silence. Retain the legacy 3-slot action ABI but canonicalize slot 2 to zero for controller-powered policies; critics must ignore arbitrary replay slot-2 values and MASAC must exclude its log-probability and entropy target contribution. No learned power gradient or target noise influences executed RF power. Shared and independent actor paths plus PQL inference copies must agree.

## Controlled variants
All run 20/15, FP32, 20k episodes, E4096 B8192 R1M, identical reward and physical settings unless the variant explicitly changes it.
1. baseline: link-budget power 3 dB; buffer 3072 B, TTL 300 s, 277200 J, coverage off.
2. learned_power: only change power_control to learned.
3. buffer16: only buffer_bytes=16384.
4. buffer32: only buffer_bytes=32768.
5. ttl600: only report_ttl_s=600.
6. coverage8: only sensor.coverage_grid_cells=8.
7. battery2x: only energy.battery_capacity_j=554400.
These are individual comparisons against baseline, not a factorial combination.

## Coverage memory
An optional 8x8 per-UAV coarse binary map, updated only by that UAV's active FOV and LOS, persists until its world is reset. No target truth, peer memory or remote delivery receipt enters actor memory. Add it before the existing destination-mask tail; critic state includes all agents' memory. Use sensor.coverage_grid_cells=0 by default, so legacy observation dimensions/semantics remain unchanged.
Native and Warp must agree. Selective reset must clear only reset worlds, exact checkpoints include memory, and CPU evaluation reconstructs the saved setting.
Keep observation schema version 3 as for existing optional link-state observation; configuration and dimension guards identify the added feature. Do not silently restore new memory/power settings into old checkpoints. Normalize only missing newly added fields to their legacy defaults for old exact environment snapshots; missing memory array may be defaulted only when coverage is disabled.

## Evaluation and seeds
Reuse the existing single-worker asynchronous CPU evaluator: evaluation_interval=0, evaluation_interval_steps=3686400, periodic_evaluation_episodes=5, periodic_evaluation_seed=1000000044; timeout 900 s. Final CPU evaluation uses 30 episodes, seed 2000000044. Validation and test banks remain disjoint.
Prepare a study manifest for independent training seeds [44,100044,200044], algorithms MASAC/MATD3, modes BASE/PQL. This yields 84 explicitly labeled cases when all seven variants are requested. Preparing this matrix must not silently submit 84 paid/quota-bound jobs.
CLI study supports plan/config materialization, sequential local execution of selected cases (existing train/evaluate, no new scheduler), and summary of completed final evaluations. Use sample std across independent training-seed means and expose completed/missing cases, precision, scenario, and unequal interaction budgets. Never call repeated eval episodes independent training runs. A focused smoke uses all four algorithm/mode pairs and multiple seeds, small counts/horizon only for verification, never claimed as production learning.

## Feasibility and future sensing
Run the privileged feasibility controller for full 1800-step 20/15 cases on at least 5 layouts and expose success/failure honestly. A failure of the oracle is not a proof of impossibility.
Read-only reference: /home/aduy/Documents/NCKH/uav_search_target/uav_marl/world/sensing.py and configs/task/uav_search.yaml. Later phase: altitude-dependent Pd/Pf [0.99,0.9,0.8,0.7]/[0.01,0.1,0.2,0.3] at [0,50,100,150] m, Bayes, posterior >=0.99, altitude <=50 m plus positive observation before confirmation, then XYZ dynamics/LOS/energy/observation/action changes together. No 3D or probabilistic sensing implemented in this phase.

## Verification
RED->GREEN behavioral tests for real power selected/energy, learner masking/entropy/replay invariance, native/Warp local coverage + occlusion + partial reset + exact resume, real study configs/CLI/multi-seed aggregation, and asynchronous evaluator scheduling with the new profile.
Run project unittest baseline and full pytest suite after implementation; CPU Warp parity is required, real CUDA only if hardware exists. Run compileall, diff checks, credential-value scan without printing secrets, and an independent whole-change review. Preserve the preexisting uncommitted CURRENT_TASK.md edits, add a scoped continuation record, commit and push authorized changes per AGENTS.md.
