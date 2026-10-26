# Current task: CUDA scenario-first fixes and joint storage cleanup — ACTIVE

Updated 2026-10-26T03:35:45.108376+00:00. No 20k production cohort was requested or launched.

- **Legacy provenance repair is COMPLETE for all six proven families:** **187/187** original W&B versions were recreated as manifest-only artifacts in their original collection/version identities, covering the same **193 immutable HF files**. Global fresh closure proves exact collection counts, all source runs still finished, all config/scientific-summary hashes unchanged, all scalar/final checkpoint pointers direct, all 193 HF size/SHA metadata unchanged, and the one stale `e0c7b655` fallback provenance key repaired. Evidence: `evidence/2026-10-26-legacy-provenance-canary.md` and `evidence/2026-10-26-legacy-provenance-family-repair.md`.
- **CUDA v3 is terminal PASS on latest committed HEAD `4c5b8862`**: 2xTeslaT4, scenario 280 passed; production-shaped primary probe pass; `FOUR_CUDA_LEARNERS_PASS`; full CUDA repository suite **834 passed, 21 skipped, 37 warnings, 2432 subtests, 0 failed**. The v2 14-failure cluster is fully explained by CPU-only maintenance/visualization tests being run on a CUDA-visible host; the test-only classification fix is validated on real T4. Evidence: `evidence/2026-10-26-cuda-v2-full-suite-triage.md` and `evidence/2026-10-26-cuda-v3-storage-legacy-progress.md`.
- Fresh post-fix CPU suite remains **774 passed, 81 skipped, 8 warnings, 1285 subtests, exit 0** under `HF_HUB_OFFLINE=1 WANDB_MODE=disabled`. Production archive/visualization CPU-only guards are unchanged.
- Periodic-policy cleanup: original 3-version canary remains successful. Bounded **batch 01** additionally retired exactly 10 proven-safe `01c9edab` periodic versions and 10 matching LFS objects, removing **1,653,111,278 B**, with `delete_aliases=False` and `rewrite_history=False`. Verification-only recovery after a local receipt-formatting TypeError proves 1203 current immutable tuples, all 864 retained tuples, 72 full-state manifests, 193 legacy paths, and synthetic-test paths remain valid. **339 candidates remain untouched by this batch and require fresh closure.**
- Legacy provenance is now repaired and re-verified: immutable HF revision `446dced...` still contains exactly **193 files / 37,418,273,432 B**, while the six original W&B collections again expose the proven **187 original version identities** as tiny manifests. The archive payload was not rewritten.
- Current order: additional periodic-policy cleanup only from a new fresh closure -> final evidence/storage totals/Git. No production training launch.

- Source fixes committed/pushed: `7223f97` aligns obstacle tolerance, prevents generated clearance disks from disconnecting free space, excludes the invisible small-camera annulus, adds `world.generation_version=2` to protocol/load guards, and preserves unknown/original generation on historical media publication. `9f60ce9` binds all5CPU Warp fixture masks explicitly to CPU. Observation_version stays3; reward, battery and primary1KiB workload/learner settings remain unchanged. Corrected training must start episode0 under the new generator protocol.
- Confirmed legacy impossibility: primary native seed302,target6 has an optimistic nearest reachable visible camera point131.250575m vs100m FOV. Frozen geometry and conservative independent proof are in evidence/2026-10-26-unobservable-*. Boundary RED8fail observed before Sandbox reset; generation RED13fail and historical-publication RED9fail logs persisted. Full source review accepted the construction/guards after the historical-publication fix.
- New test-isolation bug fixed in `823bf16`: two real-render handoff tests reached liveHFpublisher via local credentials fallback, changing six synthetic visualization paths (3LFSobjects321685B). RED2fail with a guard before credential lookup; mock only external publisher, keep real train/load/render/evaluation and guard. GREEN31pass9subtests; independent review accepted. Pollution is preserved pending exact dependency proof, not silently deleted. Future local pytest uses HF_HUB_OFFLINE=1 and WANDB_MODE=disabled.
- Fresh final CPU suite on this exact test/source tree: **774passed,81skipped,8warnings,1284subtests,exit0,211.15s**. Evidence `2026-10-26-cpu-offline-final.log`. Prior full result774/81/1282/422.28s is superseded by this final offline run. All21YAMLprofiles validate; generator smoke105native+42WarpCPU scenes across all21profiles passes. Native primary1030seed scan also passes. Primary profile SHA256503cc6fb34133b7ffb0a43e3c9238b07a5e587958a49d68c45d3f491f711e952.
- Kaggle diagnostic lineage: v1 terminalERROR at scenario stage from the fixed CPU-reset-mask device bug; v2 is now terminal and superseded by the fresh receipt summarized above. Preserve v1/v2 receipts as evidence; do not relaunch stale commits.
- W&B42rawlogs completed (oldcanary+remaining41): raw1,619,600,956B losslessly archived to PRIVATE HFdataset `duy95/uav-run-evidence-archive` (gzip223,401,167B). Remaining41 immutable commit `db2f530f76c970b5dae638830f6608031ac25ca6`; canary `291477fec36fe18ec1f9d8c9360fd703cba0fa0b`. Fresh forced-download/decompress MD5/SHA/size proof before per-file removal; source scalar archive mappings persist. All14scientific records, otherfiles and20history/eventversions unchanged;42rawnames absent. Fresh01:12UTC W&Bstorage settled1,862,931,994B, exact41-file decrease1,616,620,635B.315runs228finished65failed22crashed. Five heavy protected artifacts retained.
- HF two historical-only healthcheck LFSobjects24,117,248B removed with rewrite_history=False; both heads/all original commitIDs/headtrees/all1216manifesttuples preserved;3retained archived checkpoints force-restored/hash-verified. Completion receipt outputs/.../storage-rebuild/hf-test-canary/completion.json.
- Periodic-policy proposal originally identified352obsolete unaliased versions from18strict-completed producers. The authorized THREE-version canary is complete/successful as summarized above; 349 candidates remain only as candidates and require a fresh closure before each bounded batch. Exact receipts are in `outputs/.../storage-rebuild/policy-canary-three/`.
- Historical remote provenance defect is repaired: all187 proven archive mappings across6legacy families now point to recreated same-collection manifest refs, and the193 immutable HF payload files remain protected. Durable read-only proof, per-family receipts and global verification are under outputs/.../legacy-provenance-repair/. Do not rewrite those HF payloads/history.
- Next: commit/push this triage, run latest-HEAD CUDA v3, then continue only freshly re-proved storage candidates; finish provenance-repair proof and consolidate receipts. No production launch.

---

# Current task: Kaggle CUDA scenario-first audit and joint storage cleanup — IN PROGRESS

Updated 2026-10-25T23:43:10.030922+07:00. User explicitly requested real Kaggle CUDA testing, W&B/HF cleanup, scenario logic/modeling before algorithms, and minimal verified fixes.

- Plan: `plans/2026-10-25-cuda-scenario-storage-audit.md`. Existing linked worktree and branch `feat/cpu-replay-20261022` retained; source under test is `dc63f27c12e298c3f495fff8e1c4e84e7f191369`.
- Kaggle diagnostic `duytrngoanh/uav-cuda-scenario-algorithm-audit-20261025`, version 1, account `account_01`, quota 24.88 h before launch; actual latest status `running`. Submission is not test completion or hardware proof. Fetch `/kaggle/working/cuda-audit-receipt.json`, stage logs/XML and probe results before claims. Runner has no account/W&B/HF credentials and does not modify production jobs/stores.
- Required order: scenario regressions -> primary 8-world1800-step and4096-world16-step probes plus privileged feasibility controls -> four production-shape CUDA execution and small train/save/load/eval probes -> full repository CUDA suite. Failures stop subsequent stages. Timeout10800s; no production training budget was launched.
- Evidence runner/launch files: `evidence/2026-10-25-cuda-scenario-*` and `2026-10-25-cuda-learner-probe.py`. Check actual hardware from receipt (requested T4x2 is not proof).
- Storage inventories are read-only so far. HF main checkpoint repo is about426GB and live manifests also use a second public checkpoint repo about203GB. All retained immutable revision/path/hash references must survive cleanup; never globally squash/rewrite history.
- W&B proposed first batch:42 old raw episodes.jsonl/diagnostics.jsonl/output.log files,1,619,600,956B,from14 pre-Oct18 successfully full-budget-complete sources. Before any removal, losslessly compress to a private HF archive, force-download/decompress/hash, persist source mappings and recheck terminal state. Retain W&B history/events/metrics/config/summary and protected failed/crashed/final.pt artifacts.
- Additional obsolete intermediate policy and historical completed-lineage resume retirement is still being proved jointly. Do not treat inventory candidate lists as approved deletion receipts. Five standalone HF test objects are being checked separately.
- Local diagnostic progress/inventories: `/tmp/uav-cuda-audit-20261025/`; recover remote job and committed runner if scratch vanishes. Next: retrieve CUDA stage results, investigate scenario findings first, then algorithms, perform only verified cleanup batches, update evidence and commit/push actual state.

---

# Current task: W&B cleanup and runtime re-verification — COMPLETE

Updated 2026-10-25 23:22 +0700. Tested source: `c4d5225eb34104cfa8e13444ae4f5fac0963f75d`; local HEAD and a fresh remote ref matched. The commit containing this record changes docs/evidence only.

- Resolved the handoff uncertainty: bulk `zlh6haz3` finished 34/34 with zero errors; canary `vxc2hc2e` finished 1/1. All 35 originals are absent, all replacements and producer mappings exist, and all immutable HF size/SHA256 metadata match. Three sampled old refs resolved and passed forced fresh-download checksum checks. No duplicate archive or deletion was performed.
- Complete model inventory: 132 collections / 1,149 versions. Archived binaries: 313,266,608 B; replacement manifests: 15,521 B; net logical decrease: 313,251,087 B. Current model logical total: 54,219,565 B. Five protected heavy artifacts remain (one failed source, two crashed sources, two final.pt schemas); no remaining heavy version is in scope.
- Fresh W&B at 2026-10-25T23:20:42+07:00: 315 runs (228 finished, 65 failed, 22 crashed), zero active. Project storageBytes=3,482,532,950; entity=3,485,826,151. Model storage of 54.22 MB is distinct from total project storage of about 3.48 GB. Run/history/event/metric/config data were retained.
- No source changes since the prior owned-code audit. Re-read critical actor/critic and native/Warp locality/report paths; no new confirmed leakage or source defect. AST: 100 source / 97 test / 8 script modules; 21 YAML profiles validate; compileall passes.
- Fresh full suite: **740 passed, 77 skipped, 8 warnings, 1,274 subtests passed in 532.46 s; exit 0**.
- Fresh runtime smoke: 4/4 pass. MASAC/MATD3 BASE/PQL each trained two episodes / 80 steps with networking, then saved, loaded, checked finite parameters and evaluated one episode using seed 2000000044. BASE: 73 updates; PQL: 392 critic updates; MATD3 PQL: 196 actor updates. This checks runtime wiring, not learning quality.
- CUDA is unavailable here; the 77 skips were not executed. The prior 78-pass T4 diagnostic is historical, limited-scope evidence. No production run/controller was started, restarted, cancelled or deleted. Primary 1 KiB scientific config and acceleration flags are unchanged.
- Evidence: `evidence/2026-10-25-cleanup-runtime-reverification.md` and its linked JSON/full-suite/CLI logs.
- Next action: no cleanup or source fix remains under this verified scope. If requested, do a bounded real-T4 smoke or a fresh corrected four-way episode-0 cohort after live checks. Historical scientific policies are not corrected canonical results. Keep the five protected artifacts and histories.

---

# Current task: full owned-code correctness audit — COMPLETE

Updated 2026-10-25 17:50 Asia/Ho_Chi_Minh. User requested a fresh start-to-finish audit of config, scenario/environment, Dec-POMDP/CTDE, communication, MASAC/MATD3/PQL, replay/checkpoint, training/evaluation/tracking and cloud lifecycle. Plan: `plans/2026-10-25-full-code-correctness-audit.md`; evidence: `evidence/2026-10-25-full-code-correctness-audit.md`.

- Inventory:100 owned source modules,97 tests,8 scripts AST/compile clean;21/21 YAML load+validate. Primary 1KiB profile remains report/payload1024B,buffer3072B,env4096,batch8192,replay1M,CPU75%,35 targets,25 obstacles,horizon1800,seed44,final eval30/seed2000000044. No algorithm/reward/profile retuning and no cloud job mutation.
- Dec-POMDP re-audit: no new actor leakage found. MASAC/MATD3 execution consumes only per-agent observation; privileged state stays critic-only. Out-of-range link metadata and remote-GCS-report leakage remain fixed; observation_version3 guard remains enforced. Oracle helpers are confined to feasibility/relay audit.
- Algorithm/system audit: SAC/TD3 core target/update structures are internally consistent with project hybrid extensions. PQL path correctly labels itself `pql-inspired`; it is MASAC/MATD3 plus asynchronous collector/critic/policy scheduling, not a claim of canonical PQL. Replay/optimizer/RNG/environment resume gates remain present.
- **New confirmed defect fixed TDD:** `OnlineTracker.complete()` previously defaulted missing `episode_budget_complete` to True and missing stop reason to `episode_budget`, allowing a legacy/future incomplete result to be mislabeled as full-budget completion. RED reproduced the false True; minimal fix now fails closed: missing -> completion False, stop_reason `unknown`, session_status `session_completion_unknown`. Current BASE/PQL trainers already provide explicit fields, so verified historical20k runs are not changed by this finding. Focused GREEN3passed; lifecycle regression86passed+38subtests.
- Rejected findings: MATD3 validation is present (initial suspicion disproved); Ruff F821 in legacy benchmark is a closure false positive and direct MASAC/MATD3 benchmark behavior passes. Legacy CPU visualization fallback1episode/seed10044 remains backward compatibility only; canonical current path passes explicit30/2000000044.
- Methodological/reproducibility risks, not source bugs: memoryless decentralized actor restricts the Dec-POMDP policy class; state-based centralized critic may have partial-observability bias/variance; single training seed limits algorithm-level inference; Python deps are range-pinned rather than lockfile-pinned; UavNetSim internals are external and not Git-tracked here (upstream commit is pinned). Sensor is intentionally exact deterministic FOV+occlusion (no Pd/Pf). Propulsion is the documented uncalibrated `140+0.5v^2+2a^2 W` surrogate; with277.2kJ it gives ~1980s stationary,1459s@10m/s,1098s@15m/s,815s@20m/s before acceleration/radio, so the1800s mission is intrinsically battery-tight under sustained motion.
- Pre-fix whole suite in this audit:739passed77skipped8warnings1273subtests,exit0. Fresh post-fix authoritative suite: **740 passed,77 skipped,8 warnings,1274 subtests,exit0,285.53s**. `git diff --check` clean. Audit is complete for owned tracked source; next scientific step is a fresh episode-0 corrected cohort only when explicitly requested.

---

# Current task: W&B / 1 KiB / Dec-POMDP correctness audit — COMPLETE

Updated 2026-10-25, Asia/Ho_Chi_Minh. Latest user request was to verify live runs and report size, audit V3 Dec-POMDP and owned code, then fix confirmed defects. Full report with explicit coverage limits: [2026-10-25 audit](evidence/2026-10-25-wandb-decpomdp-audit.md).

### 17:24 Dec-POMDP re-verification after user request

- Re-read the actor-observation, report-custody and native/Warp observation paths from the connected worktree and independently checked the pushed branch via GitHub. MASAC/MATD3 execution still feeds each actor only `observation[:, i]`; privileged `state` remains critic-only during CTDE training.
- Confirmed the two previously identified leakage classes remain fixed: out-of-range peer/GCS link metadata are zeroed, and local report sensing/recovery does not consult remote GCS `delivered` state. GCS delivery dedup remains only in transfer/accounting.
- Focused verification with the active repo on this host: `PYTHONPATH=.:tests python3 -m pytest -q tests/test_decentralized_observations.py tests/test_receiver_admission.py tests/test_checkpoint_observation_version.py tests/test_training_audit_regressions.py` -> **31 passed, 3 skipped, 6 subtests passed**. The skips are CUDA-only on this host; the earlier real-T4 evidence remains the hardware check.
- No new confirmed Dec-POMDP violation was found, so no production source change was justified. Keep the minimal existing fixes and do not invent a new observation/control model. Next scientific step remains a fresh episode-0 four-way cohort only when explicitly requested.

## Verified code and tests

- Worktree `/home/aduy/Documents/NCKH/uav_network-public-hf-20261022`, branch `feat/cpu-replay-20261022`.
- Final tested source: `66a4b7025a46c8249ff79760a3211b0e02075b4d`, pushed and independently read via GitHub. This final record commit changes docs only.
- `c05806a4a1ff772d7062a0850342df4853f686b0`: native/Warp camera sensing no longer consults remote GCS delivered flags to create/recover local report custody. Unique GCS delivery dedup remains.
- `2f4b5a8`: persist/validate observation_version3 in policy, actor snapshot, BASE/PQL resume; old/missing semantics rejected. Scenario protocol identity includes observation version.
- `66a4b70`: CPU per-episode SDK/panel axes fixed; actual checkpoint-step and saved observation provenance recorded. Historical publication-only recovery retains unknown/older source versions instead of relabeling them as current.
- Final full pytest: **739 passed, 77 skipped, 8 existing dependency/cleanup warnings, 1270 subtests, exit0**, 203.25s. Real T4x2 diagnostic at c05806a: **78 passed, 0 skipped, 12 subtests, exit0**. CPU reporting focused36passed. Real W&B0.30 offline metric-definition check passed; production online guard unchanged.
- Final structural/config check:100 source modules,97 test modules AST-parsed and inventoried;21 YAML profiles loaded/validated. Detailed review of scientific/training/cloud paths plus root PHY/motion/config/precision reads; do not describe this as a proof of every utility/dependency or universal simulator equivalence.
- MASAC/MATD3 learner files and primary1KiB profile are unchanged from initial4eb45ed source. No acceleration flags/reward/hyperparameter retuning.

## Live result and scientific status

- Latest W&B snapshot16:21:55 Vietnam time25/10/2026: **zero Running runs**.
- MASAC BASE `bb1adf51`:20,000 episodes,35,840,000 transitions, episode_budget_complete=true, stop_reason=episode_budget. CPU `44656f8c`:30/30; Kaggle complete; production _cpu_complete=true; W&B/HF publication ok; fresh media sizes/hashes match.
- MASAC PQL `12582cc7`:20,000 episodes,35,139,584 transitions, same strict completion/publication gates. CPU `831a115e`:30/30 with fresh asset hashes verified.
- BASE/PQL CPU final evaluation seed is **2000000044**, matching config/job; old handoff seed10044 was stale. BASE search74.095%,delivery25.238%,success0/30; PQL search77.714%,delivery33.619%,success0/30.
- Historical MATD3 BASE `767cb8da` and PQL `30a40bf7`: strict20k completed, report/payload1024 B,buffer3072 B; already tagged obsolete-pre-decpomdp-fix (pre6263156). New MATD3 cases were not launched; fresh account14/15 owned lists empty.
- **All existing trained policies predate the new c05806a local-report fix**, including completed MASAC cases. They are historical, not a canonical corrected local-sensing cohort. Six MASAC root/child/CPU runs now have historical-pre-local-report-fix tags; original scientific outcomes/source/config/history retained. CPU summary checkpoint steps corrected from0 to verified35,840,000 /35,139,584.
- Primary report/payload1024 B,buffer3072 B. Config `configs/comparison-20k-hybrid75-1kb.yaml`; env4096,batch8192,replay1M,CPU75%,targets35,obstacles25,horizon1800,seed44. Map1000m XY and fixed altitude100m in this repo. Legacy1MB profiles retained as history/stress tests.
- Legacy local automation6adb1a0b96dc8191b26ad5ed86545aac was privately checked and is disabled; no automation mutation performed. Existing cloud controllers/data were preserved. Only a bounded T4 correctness diagnostic was submitted today; it completed.

## Next authorized work

No replacement20k cohort or merge was requested during this correctness audit. Worktree and feature branch are preserved and synchronized. If the user requests training next, reconcile live jobs/accounts again, use the published corrected source and observation_version3, and launch a matched four-way cohort from **episode0**. Never warm-start the corrected cohort from any historical checkpoint above. Retain 1KiB shape, auto-continuation and final30-episode CPU protocol unless the user changes the spec.

Independent review findings were fixed in one TDD pass; no confirmed minor left deferred. Historical media remain publishable with their original/unknown protocol version, while old policies need pinned historical code for evaluation. Detailed evidence and limitations are linked in the report. Older sections below are historical context, superseded by this record.

---

## 2026-10-24 Dec-POMDP local-observation leakage fix — VERIFIED

- Root cause: `link_state_features` exposed required-power / progress-to-GCS / receiver-buffer features for peers and GCS outside the local radio range; GPU observation kernel mirrored the same behavior. Actor therefore received derived information about unreachable nodes even though peer position slots and destination masks were range-limited.
- Fix: preserve the observation shape but zero all 3 link-state features unless the source is active and the destination is locally radio-observable (active peer within `peer_range_m`, or GCS within `gcs_range_m`). Centralized critic state remains unchanged. Policy-selected next-hop routing remains unchanged.
- TDD RED: CPU regression failed with out-of-range features `[[0,0.1915,1],[0,0.2915,1]]`; detached pre-fix GPU worktree failed with `[0.370332,-0.315631,0.5]` for an unreachable peer.
- GREEN: focused observation/native parity `24 passed, 1 skipped, 4 subtests`; full suite `717 passed, 74 skipped, 8 warnings, 1265 subtests` in 217.05s.
- Scientific consequence: checkpoints trained before this fix used the old observation semantics. Do not resume them under the fixed observation and call the result a single canonical Dec-POMDP lineage; a clean canonical cohort must start from episode 0 after this fix.

# Current task: full training-path audit and measured fixes

Updated 2026-10-24 10:55 Asia/Ho_Chi_Minh.

## Cloud independence implementation checkpoint

Plan: plans/2026-10-24-sandbox-independent-flow.md. Added private CPU controller generation/state/heartbeat, terminal-only continuation/quota decisions, adoption/no-overwrite, independent session handoff. CLI auto_continue seeds it and retains GPU receipt if seed fails. CPU visualization timeout39600 and resume completed evaluation episode/seed records; publication-only recovery reads saved Kaggle assets. Keep all30configured evaluation episodes; require W&B and immutableHF delivery before pipeline complete. Regression RED confirmed on pre-fix 44fdd43; focused controller/continuation56passed38subtests; final fullsuite707passed74skipped1250subtests; fix source pushed at7222fd47. Real cloud smoke is currently blocked pre-push by Kaggle private-dataset download timeouts reproduced on spare accounts35 and27; no phantom child/kernel was created and all five production continuations remain running. Re-run real E2E when that service path responds, then deploy controllers. Do not yet claim Sandbox can close.

Fresh10:47 W&B: recoveredPQLaa859075 now actually trains global7,053,312 above restored7,045,120; recovered500k01763049 at14,049,280 above13,914,112. Three healthy1M children preserved and progressing: MASACBASE9,269,248, MATBASE9,527,296, MATPQL9,850,880. No new full-budget completion claim.

## Latest user priority: Sandbox-independent full flow — ACTIVE

User repeated that once W&B shows Running, Sandbox/their computer will be turned off; training continuation and final CPU visualization must still happen. Implement a CPU-only Kaggle controller per lineage; do not rely on the local ChatGPT/Sandbox watcher. Design: specs/2026-10-24-sandbox-independent-flow-design.md. Reuse existing GPU self-chain, act only after terminal parent, wait same-account quota, adopt actual children, chain bounded CPU controller sessions, and verify final CPU visualization. Preserve all five current GPU jobs. New design builder check, test-first implementation, full suite, real labelled end-to-end cloud smoke, and deployment to five lineages required before claiming unattended readiness.

Previous Task7 source at0c80b7d: CPUfullsuite680passed74skip1235subtests; allcorefixescommitted. Acceleration smoke v2 bjkjbhm/uav-task7-acceleration-smoke-20261024 pinned0c80b7d COMPLETE; compile-only and fused-only preflights passed, realCUDAprobe1passed14warnings22.66s; v1compile-only/fused-only passed, staleCUDAcritic-dtypeassertion corrected to existing AMP contract and rerun. Netcache5fcb272 rejected+1.30% below2% and reverted. Kernel-blockgeometry hypothesis (4096worlds/default256threads gives16blocks on40SM T4) is unimplemented backlog; current user priority is cloud independence.

Current production: healthy1MBASE MASACa8ac9ffc/MATD3 7a57e2a5/PQLMATD3bdb9a503 preserved; same-account21PQLc02 aa859075 and account28BASE500kc02 01763049 submitted on source4cd9358, timeout39600, rootprefix1066/7045120 and4745/13914112. Their bootstrap is RUNNING; fresh training progress still needs verification. Recovery evidence: evidence/2026-10-24-task7-recovery-submissions.json.

All99ownedPython modules,8scripts,20YAMLconfigs audited by root/read-only domain agents. Final Task7 report/verification consolidation remains; do not redo audits or restart production. Watcher6adb1a0b96dc8191b26ad5ed86545aac was paused02:19 for missingSandbox, resumed10:03 with latestlineages; only configuration enabled verified, no new scheduledrun. New cloud controllers replace this dependency.

## Resume here — Task 7 IN PROGRESS

New user instruction after Task 6: audit all owned scenario/algorithm/runtime/cloud code and fresh online W&B metrics/logs, explain why 20,000 episodes × <=1,800 steps cannot finish within 12h, and implement justified fixes. This authorizes investigation and minimal verified fixes; proceed sequentially, preserving healthy production training and scientific settings. Task 6 remains completed/reverted at commit6062744.

- Worktree `/home/aduy/Documents/NCKH/uav_network-public-hf-20261022`, branch `feat/cpu-replay-20261022`, starting HEAD `6062744b6626fcc73f55fab8b7f321da74a5f1d9`.
- Read-only audits split into scenario, algorithms/replay/runtime, native network, cloud/lifecycle; root owns GPU packet path and integration. No agents may edit or touch production jobs.
- Fresh W&B 01:39 Oct24: BASE MASAC650/6,983,680 and MATD3 0/3,825,664 finished `time_budget`; both Kaggle parents COMPLETE. PQL1M child3939/7,331,840 running; MATPQL0/3,465,216 has time_budget recorded. Full20k completion conditions unchanged.
- Downloaded actual root continuation control files under `/tmp/uav-task7-terminal/account_32` and `account_33`. Both children are **RUNNING** on their same accounts: `jwekfjjwoigjo/uavnet-train-20261023-073957-c01`, `lihuhi/uavnet-train-20261023-074258-316e25-c01`; both pin source337608c2, verified prefixes650/6,983,680 and0/3,825,664. No new child W&B run yet at snapshot: preflight/restore is plausible, do not duplicate or restart them.
- Actual runtime budget: max_train_seconds36000 (10h), submitted ceiling39600 (11h), wrap-up reserve1020s for artifact_timeout900; a nominal12h allowance is not12h of training.
- Native/GPU IDLE/LISTEN semantics reviewed. LISTEN already coalesces up to nearest other event; broad skipping changes insertion ties, diagnostics or link snapshots. Candidate hypothesis: retain all logical events and reuse already-valid pair gain inside frozen-geometry packet_step instead of repeated displacement/sqrt checks. Scoped candidate now implemented after RED test (missing busy_in_step). Focused propagation/native packet/physics/retry/admission suite:79 passed,1 skipped,14 subtests in23.62s. Generic busy unchanged. T4 benefit gate pending; unpromoted.
- Proven ancillary issue for explicitly budgeted jobs: CPU-visualization builder accepts time_budget/unfinished episodes outside auto_continue; legacy partial previews without a declared target are intentionally supported by existing tests, so keep that path; compile/fused-only launches skip acceleration preflight. These do not explain primary collector slowdown. Fix one at a time with a failing test, not bundled speculative changes.
- Next: finish minimal design/test/benchmark plan, write performance gate before candidate code, smoke CPU, benchmark pinned baseline/candidate on T4 account35 after quota check. Keep candidate only if measured gain exceeds noise and all initialized state hashes/native oracle match. Then address concrete lifecycle gates with test-first changes, full suite, review, commit/push and update audit coverage/evidence. Do not promise completion in12h without measured full-training throughput.

### New steering and live recovery at09:30

- Healthy1M children: MASACBASEa8ac9ffc4096/8,630,272; MATBASE7a57e2a54096/8,818,688; MATPQLbdb9a5034096/9,162,752. Preserve all three.
- MASACPQLc01bc18a145 failed4096/8,908,800 during final full-resume torch.save: basic_ios iostream error / unexpectedpos3030776896. Confirmed save I/O failure; disk pressure strongly supported by last5.11GBfree versus parent full-resume15.52GB and old66f4d40 retaining the staged input. No current CUDA-OOM evidence in that log. Top-level continuationValueError is secondary, job-result empty.
- Rootfc14508c:v3 descriptor downloaded: immutableHF4672be4ad3ea3d13a419e7593960d3fb01955420, resume.pt15,522,939,889B SHA8d26e1e9671157180c4b048c0b95a515bded6ead06526d6c6c8a468b64fecbed, prefix1066/7,045,120. Need inspect child artifact descriptors/Kaggle output for any newer valid full state before rollback; never label policy-only4096 as exact resume. Failedc01 must remain preserved; recovery uses unusedc02 sameaccount21 and tested4cd9358 source/controller, explicitly override source because helper preserves parentcommit.
- 500kchild30e43dda finishedtime_budget4745/13,914,112. Plannedc02failedonlyinsufficientquota0.41h; sameaccount28 waiting, artifactv3. Do not switch accounts. Verify immutablefullmanifest and ownedchildstate before future quota-resume.
- /tmp from overnight session was cleared; committed/home worktree persisted at346632c plus D WIP. Recover records from git/evidence, do not rerun completed remote training or completed T4 AB.

### Task7 ledger

- Task7A: RED missing scoped Warp CCA helper; GREEN79passed/1skipped, CPUfixture2worlds dt.01 errors0. Candidate5fcb272 unpromoted; T4same-account35 quota26.7h available, kernel bjkjbhm/uav-frozen-gain-t4-ab-20261024 v1 launched. B,C,C,B,B,C, full140arrays,39CUDAtests; wait for actual result.
- Task7B RED:6 unfinished/inconsistent-budget cases accepted beforefix; GREEN:37passed,35subtests in8.18s, exit0 (HF_HUB_OFFLINE=1 WANDB_MODE=offline). Legacy no-budget preview still passes.
- Task7B Ruling: guard explicitly budgeted final handoffs, retain existing no-budget legacy partial previews — existing CPUHandoffTests deliberately require partial visualization; cost if wrong is legacy no-budget output still being mistaken for full training, so retain training_status labels and document scope. No production job changed.

- Task7C RED: compile-only/fused-only skipped preflight; BASE preflight allocation ignored hybrid75 (actual allocation-expression CPU reproduction). GREEN:50passed,2skipped,12subtests in7.54s, exit0. Bootstrap now gates any requested acceleration; BASE uses production make_replay and reports cold/hot bytes. No flags enabled by this fix.
- Task7C Ruling: align preflight replay with production while fixing independent-flag gating — otherwise expanded preflight could falsely OOM on all-GPU replay; cost if wrong is a misleading diagnostic, so bounded real two-T4 smoke remains required, not a claim of canonical accelerated1M success.

- Task7A FINAL: T4v1 COMPLETE, baseline39.11293086 vs candidate39.62140881t/s, **+1.300025% reproducible**, all140arrays match6trials,39CUDAoracle tests passed37.222s. Below predeclared2% gate; REVERTED allthreecandidatefiles byte-equal6062744. Do not call it noise/no gain; it is too small for the acceptance gate. No further micro-cache candidate this task; event-loop architecture needs separate measured redesign.
- Task7D RED actor-grad failure reproduced from actual BASEprobe context on CPU BF16; fixed cache_enabled=False. PQLprobe actual report expression failed on missing HybridMeasuredReplay.data; storage formatter now reports cold/hot tiers and preserves plain keys. GREEN65passed1skipped in8.44s, exit0. Real CUDA probe/preflight smoke pending; fullsuite after netrevert pending.

- Final review Important topology finding FIXED: expanded acceleration guard must support non-dual-T4 CUDA. RED one-board P100 facade failed oldguard; GREEN34passed16subtests9.97s after generic usable-CUDA guard, BASE/PQL role resolution and device_count loops. Load real test consumers before patching constructors to avoid import-time mock capture. No re-review; fresh fullsuite and2T4smoke next.
- Fullsuite before topologyfix:679passed74skipped8warnings1234subtests498.67s, exit0. Postfix suite pending.
- Recovery submissions actual at09:50: PQL1Maccount21c02 aa859075 and BASE500kaccount28c02 01763049, sameaccounts, source/controller4cd9358, both timeout39600 after freshquota30h. Prefixes1066/7,045,120 and4745/13,914,112; immutableHFsize/SHA verified. See evidence/2026-10-24-task7-recovery-submissions.json. Submitted is not yet verified training progress; checkKaggle/W&B.

- Post-review fullsuite:680passed74skipped8warnings1235subtests486.84s, exit0. Actual T4smokev1: compile-onlyhybrid and fused-onlyhybrid both PASSED (6collections,4critic/4actorupdates;3compiledgraphs/54kernels andactualfusedAdam). CUDAprobe applied critic/actor/alpha updates successfully, then a stale test assertion incorrectly required FP16criticoutputs; existing production evidence gate/tests explicitly allow FP32critic reductions with FP16actors/completedcriticcalls. Corrected that CUDA-only assertion to the existing contract; no productionprecision change. Preservev1error/evidence; rerunv2 pinnedsource beforecompletion. CPUfocused40passed1skip4subtests6.27s.

## Previous Task 6 record

**Task 6: receiver buffer cache — COMPLETE, NEGATIVE RESULT, REVERTED.**
Worktree: `/home/aduy/Documents/NCKH/uav_network-public-hf-20261022`.
Branch: `feat/cpu-replay-20261022`. Candidate: `cb7c5d6c0c0ba50640c75bea83238f422ea4d902`; baseline: `4cd9358fe0d7cc851bc95ed1c6d5bd231666d310`.
Candidate is in history, but its two-file change has been reverted. Both files match 4cd9358 byte-for-byte. Do not recreate the cache. Primary peer-1MB gate: baseline median 39.17247858, candidate 39.16647574 transitions/s (-0.015324%, within variability); all 140 initialized arrays match across all six trials. No measured benefit, so reject this candidate. Reviewer confirmed the conclusion and byte equality.

All Task 6 gates are complete:

- Full suite after revert: **673 passed, 74 skipped, 8 warnings, 1222 subtests passed in 254.25s**, exit 0. `/tmp/uav-task6-after-revert-full-20261024.log` and `.exit`.
- T4 peer-1MB version 1: `bjkjbhm/uav-receiver-cache-peer1m-20261024`; all 140 arrays/trial include MAC RNG, pending events/packets and energy; errors=0.
- T4 control version 2: `bjkjbhm/uav-receiver-cache-t4-comparison-20261024`; active1KB peer medians 787.74247218 / 787.55057233 (-0.02436%); active1MB GCS 38.49941272 / 38.48308302 (-0.04242%); 24 array hashes/trial match, errors=0. The GCS control alone is not a peer-benefit test. V1 failed before simulation on invalid device config; only V2 was accepted.
- Existing native-vs-Warp packet oracle on candidate/T4 `cuda:0`: **38 passed in 12.333s**, exit 0. CPU suite and CUDA oracle scopes differ; do not describe local CUDA skips as having run.
- Evidence: `evidence/2026-10-24-task6-final-verification.json`, `evidence/2026-10-24-t4-receiver-cache-peer1m.json`, `evidence/2026-10-24-t4-receiver-cache-control.json`, `evidence/2026-10-24-t4-receiver-cache-cuda-oracle.log`.

**Next action:** commit/push the final evidence/revert on this branch; the resulting HEAD is the final task commit (do not amend the record just to embed its own SHA). Then stop Task 6. Wait for the user's next task instruction before implementing another candidate. The leading next hypothesis is exact MAC IDLE/LISTEN polling; it requires profiling and native parity, not relaxed timing. Learner AMP/compile/fused is a separate controlled candidate. Preserve all healthy production jobs.

## Fresh verification and root-cause evidence

- Full suite at candidate cb7c5d6: **674 passed, 74 skipped, 8 warnings, 1221 subtests passed in 243.11s**, exit 0. Log: `/tmp/uav-task6-full-20261024.log`.
- Task 4 diagnostic is now COMPLETE: `bjkjbhm/uav-component-isolation-20261023-160905`, baseline 4cd9358, 4096 worlds, two T4 visible. Saved in `evidence/2026-10-24-t4-component-isolation.json`.
- Matched T4 fixtures: active 1KB/dynamic 9726.67 transitions/s; active 1MB/dynamic 38.74995; active 1MB/reject 38.79999. Thus 1MB is ~251x slower in this diagnostic, while changing motion changes active-1MB throughput ~0.13%. Network-disabled dynamic is 15843.01 transitions/s. These are simulator diagnostics, not full-training rates.
- Additional completed instrumented probe: `bjkjbhm/uav-task6-dispatch-probe-20261023-160547`, diagnostic commit cbc7253, wider sensor FOV, 4096 worlds, 2 warmup + 1 timed step. Cumulative dispatches across all 3 steps: 4,735,399,723; IDLE + LISTEN account for 97.814%. The timed third step is 118.12 s. This identifies microsecond MAC polling as a profiling target; dispatch-count share is not runtime share, and these counts are not a production-run measurement. See `evidence/2026-10-24-t4-dispatch-probe.json`.
- GPU0 runs all 4096 environments, GPU1 runs learner/replay. Cumulative production timing supports a major collector bottleneck. **Phase-specific exception:** between 00:30 and 00:43, MASAC BASE 1M learner time grew 741.14 s while collector wait grew only 0.082 s over 750.18 s; it was learner-limited in that window. MATD3 BASE collector wait grew 723.76 s/765.18 s; MATD3 PQL collector grew 737.87 s/741.29 s. AMP/compile/fused Adam may help learner-limited phases, but do not remove packet-event cost.
- Corrected cache-cost interpretation: installed Warp 1.18 BoolOp code generation short-circuits `and`/`or`. In production 1MB/3MB, each source holds at most three reports, so the old `bytes_used` call is guarded by report-present/nonexpired/unknown checks; a naive 35x35 target-loop count overstates actual production scans. Measure the real gain.
- Fresh acceleration-config audit: all four canonical 1M W&B runtime configs explicitly show amp_mode=off, compile_models=false, fused_optimizers=false. Code supports them; do not claim they are enabled. Future learner acceleration needs a separate controlled numerical/performance gate; preserve active runs and BASE/PQL fairness.
- Tasks 1–3 were completed in the prior audit; Task 5 queue-count candidate was reverted after no measured speedup. Framefree remains unpromoted because its T4 benchmark lost to Gen1. This is not a claim that every module's audit is complete.

## Production cohort: read-only monitoring

Latest snapshot: **01:22 ngày 24/10/2026 UTC+7**, `evidence/2026-10-24-performance-latest-cohort.json`. Detailed diagnosis/ETA: `evidence/2026-10-24-runtime-root-cause-report.md`. Earlier timing-phase snapshot preserved in `evidence/2026-10-24-performance-cohort-0043.json`.

| Case | Canonical root W&B | Current child | Latest completed episodes / global step |
| --- | --- | --- | --- |
| MASAC BASE 1M | 63faeffb | not observed yet | 650 / 6,983,680; time_budget, W&B finished, Kaggle wrapping up |
| MASAC PQL 1M | fc14508c | bc18a145 c01 running | 3815 / 7,294,976 |
| MATD3 BASE 1M | 2dcbf36c | not observed yet | 0 / 3,825,664; time_budget, W&B/Kaggle wrapping up |
| MATD3 PQL 1M | 278634f6 | none yet | 0 / 3,379,200; running |
| MASAC BASE 500k | b9d73bd3 | 30e43dda c01 running | 4096 / 11,083,776 |

Root Kaggle refs: MASAC BASE `jwekfjjwoigjo/uavnet-train-20261023-073957-9c4cd8` (account_32); MASAC PQL `ewkljfelweow/uavnet-train-20261023-051420-773792` (account_21); MATD3 BASE `lihuhi/uavnet-train-20261023-074258-316e25` (account_33); MATD3 PQL `kjhiguyf/uavnet-train-20261023-074845-a0387e` (account_34). The 500k root/child use account_28, `ekjfewflwejl/uavnet-train-20261023-051422-cbe381` / `uavnet-train-20261023-051422-c01`.

The two existing children have meaningful progress beyond verified parent prefixes; preserve them. PQL parent ended 1066 / 7,045,120; 500k parent ended 4096 / 10,309,632. Old roots/children inherit infrastructure commit 66f4d40. A finished W&B parent may still have a running Kaggle notebook saving/uploading a full checkpoint and scheduling its child. Recheck actual owned state and immutable manifest before recovery; do not duplicate a healthy wrap-up or child.

## Recovery watch restored at 01:10 Asia/Ho_Chi_Minh

Automation `6adb1a0b96dc8191b26ad5ed86545aac` (UAV 1M Cohort Recovery) was disabled by the prior assistant at 21:22 Oct 23 after Sandbox Resource not found. Current Sandbox, W&B/Kaggle and GitHub reads succeeded. Re-enabled the existing hourly watch with recovery controller/source pinned to tested 4cd9358 in an isolated checkout, preserving current healthy children and excluding experimental performance commits. The update response confirms is_enabled=true; next_run_time was null and no new scheduled execution has been verified. Do not describe it as successfully executing yet. Evidence: `evidence/2026-10-24-recovery-watch-restored.json`.

## 2026-10-24 user-approved report-size decision

- HARP-LAWN Table II uses UavNetSim-v1 with a 1024-byte data-packet payload; it does not define a separate 1 MB report.
- User explicitly approved switching the primary future training workload to 1 KiB after this verification.
- New primary config: `configs/comparison-20k-hybrid75-1kb.yaml` with report=1024 B, payload=1024 B, buffer=3072 B. Preserve env4096, 20k episodes, 1800 steps, replay1M, batch8192, CPU replay75, warp_packet.
- Keep existing 1 MB runs/config immutable as historical/stress-test evidence. Do not continue a 1 MB lineage into a 1 KiB lineage; new 1 KiB training must start a fresh lineage from episode 0.
- Existing T4 isolation evidence: active1KB 9726.67 transitions/s vs active1MB 38.75 transitions/s (~251x slower for 1 MB under forced active traffic). This is a simulator diagnostic, not a full-training ratio.
- Evidence: `docs/superpowers/evidence/2026-10-24-harp-lawn-1kb-primary-profile.md`.


## 2026-10-24 1 KiB learner acceleration benchmark — COMPLETE

- Spare-account production-shape MASAC benchmark completed on Kaggle T4 x2: `bjkjbhm/uav-1kb-learner-accel-20261024` v7. Runner commit `89440b0`; production code under test is parent `fbbb556`.
- Exact benchmark shape: report/payload 1024 B, env4096, batch8192, replay1M, CPU75/GPU25 hybrid replay filled to 1,000,000 rows. Timed `sample + update`, 20 updates, two interleaved repetitions/case.
- Baseline median 566.02 ms/update. AMP-only 534.07 ms (1.060x), compile-only 534.69 ms (1.059x), fused-only 577.57 ms (0.980x), all-three 353.78 ms (**1.600x**, 37.50% learner-time reduction).
- Combined cases had AMP critic/actor/alpha 23/23 applied, 0 skipped; 3 compiled graphs / 88 Inductor kernels; all Adam groups fused; parameters finite. Full-combination GPU1 peak allocated ~5.55 GB vs baseline ~6.62 GB.
- Ruling: combined AMP+compile+fused passes the throughput/execution gate; fused-only does not. Do **not** mutate current non-AMP 1 KiB production lineages. Acceleration may be used only for a fresh matched cohort/future experiment from episode 0.
- Evidence: `evidence/2026-10-24-1kb-learner-acceleration-result.json` and `.md`.
- Next: fresh-check the four existing 1 KiB production lineages. Preserve them if healthy; only start an accelerated matched cohort if the user later chooses to replace/compare them rather than silently changing an in-flight experiment.


## 2026-10-24 matched acceleration across BASE/PQL — COMPLETE

- Kaggle T4x2 matched benchmark v8 completed from runner `b42a9a7`, no production mutation.
- MATD3 BASE: 158.82 -> 139.29 ms/update, **1.140x**.
- MASAC PQL: 2407.09 -> 2453.36 transitions/s, **1.019x** with meaningful run-to-run variation; not a robust promotion.
- MATD3 PQL: 3509.38 -> 3272.39 transitions/s, **0.932x**; combined AMP+compile+fused is slower.
- Ruling: do not globally enable the three acceleration flags. Keep current four 1 KiB production lineages unchanged. MASAC BASE's prior 1.600x result remains path-specific.
- Evidence: `evidence/2026-10-24-1kb-matched-acceleration-result.json` and `.md`.
- Next task: PQL-specific one-variable isolation (AMP-only, compile-only, fused-only; then only justified combinations) on spare T4x2, preserving production.


## 2026-10-24 PQL one-variable acceleration isolation — COMPLETE

- Kaggle T4x2 diagnostic `bjkjbhm/uav-1kb-learner-accel-20261024` completed from runner `1b2ddc9`; no production mutation.
- MASAC PQL: AMP-only **1.074x**, compile-only **0.884x**, fused-only **1.011x** vs baseline.
- MATD3 PQL: AMP-only **1.018x**, compile-only **0.919x**, fused-only **0.997x** vs baseline.
- Ruling: reject compile for PQL; do not promote fused alone; AMP is the only single variable that improved both paths.
- Evidence: `evidence/2026-10-24-1kb-pql-acceleration-isolation-result.json` and `.md`.
- Next controlled task: compare **AMP+fused (compile off)** against AMP-only and baseline on spare T4x2. Preserve live production lineages.


## 2026-10-24 PQL AMP+fused combination check — COMPLETE

- Kaggle T4x2 diagnostic `bjkjbhm/uav-1kb-learner-accel-20261024` v11 completed from runner `c8ca18a`; no production mutation; compile stayed off.
- MASAC PQL: baseline 2219.19, AMP-only 2439.91 (**1.099x**), AMP+fused 2356.10 (**1.062x**, only **0.966x** vs AMP-only).
- MATD3 PQL: baseline 3509.02, AMP-only 3457.48 (**0.985x**), AMP+fused 3500.10 (**0.997x**).
- Ruling: reject fused addition for MASAC PQL; no robust PQL-wide acceleration policy. Preserve current non-AMP production lineages. MASAC PQL AMP-only remains the only path-specific candidate for a future fresh cohort.
- Evidence: `evidence/2026-10-24-1kb-pql-amp-fused-combo-result.json` and `.md`.
- Next operational task: fresh-check all four primary 1 KiB production lineages and advance completed 20k cases to validation/CPU visualization without restarting or mutating active training.


## 2026-10-24 native vector null telemetry E2E blocker — FIXED LOCALLY

- Real E2E smoke `bjkjbhm/uavnet-train-20261024-104832-b89da3` trained 4 requested episodes then failed strict JSON because Gymnasium mixed vector-info coerced schema `None` values to `NaN`; learner metrics/parameters were finite. The controller failure was secondary.
- Root cause reproduced: mixed `[number, None]` becomes `[number, NaN]`; zero-count network stats and zero-delivery report conditionals reached `episodes.jsonl` as NaN.
- Minimal fix in `environment/vector.py`: restore None only for zero-denominator stat fields, zero-delivery conditional report means, and known optional report timestamp/age lists. Positive-count NaN remains exposed; `allow_nan=False` stays strict.
- TDD RED confirmed old failure; focused suite 50 passed +17 subtests; local 4-episode smoke completes; full suite **716 passed, 74 skipped, 1261 subtests**.
- Evidence: `evidence/2026-10-24-native-vector-null-and-e2e-smoke-fix.md`.
- Next: commit/push immutable fix, then run a new real GPU->controller->CPU visualize E2E smoke. Do not overwrite the failed smoke and do not mutate the four live 1 KiB production lineages.


## 2026-10-24 cloud E2E smoke v3 — PASS

- Fresh smoke from fixed source `fa4b7ec`: GPU `kjhihgiugy/uavnet-train-20261024-112613-05bf43`, W&B `710c151e`, controller `kjhihgiugy/uav-flow-99b4b012fb3b7e-0000`, CPU `...-cpu`.
- GPU completed 4/4 episodes with `episode_budget_complete=true`, `stop_reason=episode_budget`.
- CPU evaluation completed on CPU for seed 10044; production `_cpu_complete()` returned true.
- W&B publication `ok`; HF publication `ok` at immutable revision `cf1c73e5b9eb621d5f0e197155ef36d594e04bdb`; video/image/metrics all have hashes and immutable URLs.
- The NaN blocker is gone: schema-defined undefined telemetry is JSON null in the real CPU result.
- Evidence: `evidence/2026-10-24-cloud-e2e-smoke-v3-pass.md`.
- Next operational task: fresh-check all four primary 1 KiB production lineages; advance only cases that satisfy the 20k completion gate to validation/CPU visualization. Preserve active/incomplete lineages.


## 2026-10-24 primary 1 KiB MATD3 finalization — COMPLETE

- MATD3 BASE W&B `767cb8da`: 20000 episodes, `episode_budget_complete=true`, `stop_reason=episode_budget`; CPU eval 30 episodes complete; `_cpu_complete=true`; W&B/HF publication ok; HF revision `24dccbd57963e7eec1c434cf3de92f6ad4751ac3`.
- MATD3 PQL W&B `30a40bf7`: 20000 episodes, `episode_budget_complete=true`, `stop_reason=episode_budget`; CPU eval 30 episodes complete; `_cpu_complete=true`; W&B/HF publication ok; HF revision `fee4f14953727d0be0d1da1c3c2c86d2c840da60`.
- Existing controllers advanced both cases automatically; no duplicate validation job was submitted.
- Evidence: `evidence/2026-10-24-1kb-matd3-20k-finalization.md`.
- Next: preserve and monitor MASAC BASE/PQL until their 20k gate; reconcile any non-canonical duplicate lineages before spending more quota.


## 2026-10-24 MASAC live state + duplicate cleanup

- MASAC BASE `7f985331`: root/controller running, global_step 12,181,504; 1 KiB/env4096/20k target; not yet at completion gate.
- MASAC PQL `e0bfbcaf`: root/controller running, global_step 13,352,960; 1 KiB/env4096/20k target; not yet at completion gate.
- Preserve both; do not restart, change precision flags, or submit early validation. Existing controllers own final handoff.
- New Sandbox alias remap: resolve owner before mutation. Current: duytrngoanh->account_01, vuliu123456->account_02, libiu123->account_03, nhtbrg->account_04.
- Removed redundant MATD3 PQL root `bbfb7e35` (`nhtbrg/...085710-ecbd3f`) and its controller after proving it duplicated the already-finalized seed44 canonical experiment; source difference was only cloud preflight `torch.cuda.init()`.
- Evidence: `evidence/2026-10-24-1kb-masac-live-and-duplicate-cleanup.md`.
- Current operational action: let the two canonical MASAC GPU roots/controllers continue; only advance after the strict 20k completion gate.

## Operating constraints

- Keep healthy production jobs running. Apply performance changes only to controlled diagnostics/future launches.
- Scientific invariants for the new primary 1 KiB line: seed/RNG, reward, motion, PHY, CSMA/backoff/collision/retry/ACK order and timestamps, report custody/TTL/admission, energy, observations/state, exact continuation, 35 targets, 25 obstacles, 1800-step horizon and 20k episode target. Report/payload are 1024 B and buffer is 3072 B. Existing 1 MB lineages remain immutable stress-test/history and must not be mixed with the new 1 KiB lineage.
- Training completion requires episodes_completed >= 20000, episode_budget_complete=true, stop_reason=episode_budget. A time_budget segment is not final evaluation; zero completed episodes during a synchronized first wave is not a stuck counter.
- Continuation stays on the same account/lineage. Verify immutable HF full-resume/checkpoint/prefix before recovery; inspect actual owned children before submitting. A private-kernel permission denial is ambiguous until same-account owned-list lookup. Preserve an old-code child with meaningful progress; use b32d18e-or-newer controller only for concrete failed/absent/unsafe recovery.
- Credentials: trusted `load_credentials` helper, `/home/aduy/Documents/NCKH/uav_network/key.txt`; never print secrets. CUDA benchmarks use Kaggle because this host has no CUDA.
- Earlier handoff/replay design and completed work remain in `specs/2026-10-23-training-handoff-replay-design.md`, `plans/2026-10-23-training-handoff-replay.md`, and Git history of this record. For operational recovery, read the applicable continuation code/tests and newest live evidence before acting.

## 2026-10-24 1 KiB cohort launched and verified

- Primary 1 KiB cohort is now live from episode 0 on source `0c9fbc5b93acfd3008f9802950570849bc42997c`:
  - MASAC BASE account_01: Kaggle `duytrngoanh/uavnet-train-20261024-073119-a53a63`, W&B `7f985331`.
  - MASAC PQL account_03: Kaggle `vuliu123456/uavnet-train-20261024-074226-5b2381`, W&B `e0bfbcaf`.
  - MATD3 BASE account_06: Kaggle `libiu123/uavnet-train-20261024-074821-e57c46`, W&B `767cb8da`.
  - MATD3 PQL account_09: Kaggle `nhtbrg/uavnet-train-20261024-080451-eda89d`, W&B `30a40bf7`.
- All four GPU kernels and their four CPU cloud controllers were freshly health-checked running with `unattended_ready=true`.
- One accidental newer MATD3 BASE duplicate caused by an ambiguous 502 submit response (`a379f8bd` + controller `libiu123/uav-flow-10ea62cf6a6c06-0000`) was deleted after confirming the older `767cb8da` had more progress. The retained lineage stayed running.
- 62 s live W&B rates: MASAC BASE 792.45, MASAC PQL 1122.64, MATD3 BASE 3367.93, MATD3 PQL 2443.40 global steps/s. Versus canonical 1 MB root averages this is about 4.13x, 5.79x, 32.28x, 25.56x respectively.
- BASE bottleneck has shifted to learner: MASAC learner 2223.96 s vs collector-wait 30.32 s; MATD3 learner 999.95 s vs collector-wait 83.17 s. Do not optimize packet path first now.
- Rough early 20k ETA using historical 1 KiB episode/step relation: ~11.87 h MASAC BASE, 8.77 h MASAC PQL, 3.30 h MATD3 BASE, 4.53 h MATD3 PQL. Treat as early estimates, not guarantees.
- Evidence: `docs/superpowers/evidence/2026-10-24-1kb-cohort-launch-and-live-throughput.md`.
- Next controlled task: benchmark learner-side AMP/compile/fused on a spare account without mutating these four production lineages.