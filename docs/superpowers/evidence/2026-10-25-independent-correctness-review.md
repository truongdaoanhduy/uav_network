# Final independent correctness review, 2026-10-25

Reviewed read-only connected worktree /home/aduy/Documents/NCKH/uav_network-public-hf-20261022, base 4eb45ed9dd28567f8e1aeb8545570453b02d0927 through committed 7409a91 plus observation-version working changes (subsequently committed unchanged by root as 2f4b5a8). No source/index/branch mutations or external service writes. No subagents. AGENTS.md and top docs/superpowers/CURRENT_TASK.md read. No independent full suite duplicated.

## Findings

No confirmed Critical defect in inspected paths.

Important (scientific observability, not model calculation): CPU evaluation curves use an absent checkpoint axis. experiments/tracking.py:191-193 defines evaluation/* and eval/* with evaluation/checkpoint_global_step, but cloud/visualize.py:287-294 emits global_step=episode and never the checkpoint axis. experiments/dashboard.py:111 independently forces the same checkpoint axis. A completed CPU run therefore displays missing/zero/misleading x coordinates despite valid episode records and correct result.protocol.training_global_step. Minimal fix: explicitly define the five CPU per-episode metrics (return, search_rate_percent, delivery_rate_percent, success, episodes_completed) against evaluation/episodes_completed; teach panel_axis the same exact names; retain periodic aggregate evaluation on checkpoint_global_step. Log the actual training checkpoint step separately as provenance. Root's proposed fix matches this recommendation. Add assertions on emitted metric definitions, per-episode logs and authored panel x names; retain existing async evaluation checkpoint-axis test. Existing historical W&B rows are not automatically repaired by a source fix.

Minor (provenance introduced by version metadata): experiments/tracking.py:139 advertises current OBSERVATION_VERSION before cloud/visualize.py:296-297 loads publication-only assets. load_publication_assets intentionally does not load/re-evaluate a policy; controller.py:220 upgrades recovery source to controller source. Republishing historical media with upgraded source can therefore label an old observation protocol as version 3 in W&B config. Regular evaluate/load_policy correctly rejects old checkpoints. Keep historical media publishable, but override evaluation observation provenance from result.protocol.scenario.observation_version; missing historical field must be unknown/historical, not current 3. This path is not an executed old-checkpoint training/evaluation bypass.

## Main change assessment

- Native ReportStore.discover and Warp sense_reports_at no longer consult remote delivered[target] when creating/recovering a locally seen report. Capacity and recovery state remain local; expiry retains original TTL and fresh sighting permits regeneration.
- Both GCS transfer implementations still deduplicate unique deliveries before incrementing delivery counters, delays, hops and first_delivery. Multiple sources can consume resources delivering duplicates, which is consistent with no global acknowledgement beacon.
- Current observation version is saved for full/policy checkpoints and evaluation-only actor snapshots. Public evaluation routes through load_policy/load_evaluation_snapshot; BASE restore routes through fixed validation and load_policy; PQL restore routes through PQL validation and load_policy. Cloud automatic continuation preflight uses the same validators. Internal in-memory actor/state_dict snapshots share one running semantics version and are not external old-checkpoint bypasses.
- Same-current-version restore remains allowed. Root reports full suite 734 passed, 77 skipped, 1269 subtests, exit 0; reviewer did not independently rerun it. CUDA report tests are root-provided committed evidence, not hardware executions by reviewer.
- Actor inference reads only per-agent local observation. Central critics use privileged state/joint actions. Finite mission deadline is represented in observation and treated as termination consistently; replay stores terminal next observation before reset. Memoryless actors restrict the policy class without invalidating the Dec-POMDP formulation.
- Ideal local radio beacons/admission/power metadata, global team reward, coupled motion/safety transitions, and central critic privileged information are modeling assumptions, not automatically actor leakage. No requirement to invent packet-level control traffic was imposed.

## Actual coverage

Structural pass: AST-parsed and inventoried entry points for all 100 owned uav_network Python modules and all 96 tests. This establishes syntax and available entry points ONLY, not a full line-level audit. Inspected the entire source diff from base plus uncommitted version guard and changed scenario test. Read both complete primary comparison-20k-hybrid75-1kb.yaml and retained comparison-20k-hybrid75-1m.yaml profiles; only listed the remaining YAML files.

Detailed source reads (file bodies, some retrieved in multiple chunks):
- algorithms: action_utils.py, checkpoints.py, hybrid_replay.py, masac.py, matd3.py, models.py, replay.py.
- communication: backend.py, energy.py, link_budget.py, reports.py, routing.py; gpu/mac.py, gpu/reports.py.
- environment: env.py, observations.py, vector.py; gpu/observations.py.
- experiments: continuation.py, environment_checkpoints.py, evaluation.py, evaluation_snapshots.py, fixed_checkpoints.py, pql_checkpoints.py, pql_replay.py, pql_system.py, pql_workers.py, training.py, comparison.py.
- cloud: visualize.py.

Targeted reads/searches, not entire bodies: environment/gpu/vector.py snapshot/restore/action/step functions (127-255); cloud/remote.py prepare_checkpoints and learner command; cloud/continuation.py next_stage/build_continuation_spec; cloud/controller.py _cpu_complete/_cpu_step/controller_step; experiments/tracking.py metadata and metric definitions; experiments/dashboard.py panel_axis and uses; experiments/async_evaluation.py checkpoint-axis logging; experiments/comparison_metrics.py semantic hash diff. New tests/test_checkpoint_observation_version.py and tests/test_decentralized_observations.py fully inspected; test_scenario_v3.py changed portion; visualization-tracking/diagnostics-axis tests searched for metric assertions.

Remaining module bodies are NOT claimed deeply reviewed: algorithm optimization/parallel_critics/precision; cloud archive/client/cli/credentials/execution_check/kernel/media_publish/pipeline_status and unlisted portions of targeted cloud files; communication diagnostics/telemetry/upstream; GPU diagnostics/events/metrics/phy/power/rng/streaming/telemetry; configuration.py; environment avoidance/energy/geometry/motion/sensing/world and GPU geometry/motion/state/validation/world; experiment utility/benchmark/probe/diagnostics/feasibility/resource/rendering/telemetry/store modules, pql_training.py, and unlisted portions of targeted experiment files. Init modules and CLI entrypoints only structurally inventoried. Most existing tests only structurally inventoried. Root must combine with its own reads for any full-owned-code audit claim.

## Declined to judge / limits

- Did not independently claim fresh live W&B/Kaggle state or public media hashes; those are root's timestamped read evidence. Did not mutate running services.
- Did not claim native/Warp equivalence for all PHY/motion events, CUDA correctness from a CPU host, or end-to-end cloud service behavior from unit tests. Root owns additional GPU PHY/motion review and prior real CUDA evidence.
- Did not claim asynchronous PQL bit-for-bit trajectory equivalence after resume; the code explicitly labels asynchronous scheduling nondeterministic. Version compatibility is distinct from schedule determinism.
- Did not infer optimal policies, statistical MASAC/PQL superiority, or generalization from one seed and 30 evaluations; compare_evaluations explicitly scopes uncertainty to conditional scenarios.
- Did not judge external vendored simulator internals or service implementations as owned audited code.
- Did not label nominal local channel/admission information, team reward, or safety coupling violations absent a contradictory modeling contract.
- Did not claim absence of defects in structurally inventoried but unread bodies. This review is complete for the listed substantive paths, not an independently exhaustive line review of all owned code.
