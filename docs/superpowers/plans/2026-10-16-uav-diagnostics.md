# UAV diagnostics and verified four-account launch
Date: 2026-10-16. Starting HEAD: 241f706890dd8de9fb70c3b160225c2ca8d87399.
Workspace: /home/aduy/Documents/NCKH/uav_network-cpu-flow-20261014 (existing linked worktree, branch pql-async-20261014).
User instruction: implement the missing diagnostics/evaluation identified in the audit, then launch MASAC BASE, MASAC PQL, MATD3 BASE, MATD3 PQL on four other Kaggle accounts and explain changes.

## Binding scope and decisions
- Same task/scenario: homogeneous6UAV,35targets25obstacles,1800steps,25000episodes,seed44,4096envs, replay500000,batch8192,8updates/collection. Preserve current urban LOS/NLOS/SINR/CSMA/ACK model, motion/reward/observation/action semantics and1KiB report/3KiB buffer. Model extensions (fading/radio-hardware terms/1MB workload) require calibrated scenario choices and are not inferred from a telemetry request.
- Complete actual-attempt transmission power/link/failure/relay/report/battery telemetry with honest denominators, native/Warp consistency and zero RNG consumption.
- No online packet-by-packet traces: fixed-size counters/histograms, scalar summaries. Detailed first-discovery/first-delivery timestamps can live in episode/evaluation JSON.
- Preserve legacy metric keys. Undefined conditional delay/hops/power use None plus explicit counts; never fake0 as valid measurement.
- Periodic checkpoint evaluation on fixed held-out mini-bank5episodes, async CPU worker, cadence7372800joint transitions (one1800step vector horizon); final CPUvisualization/eval30episodes already exists and stays enabled. Log source checkpoint global step and n; do not mutate learner RNG or block collection for the full eval.
- Cohort/time-based aggregates for comparisons; historical rolling20 may remain for compatibility but must not trigger a training-regression claim for neighboring workers at the same global step. Show episode/evaluation source clearly.
- W&B online mainproject truongdaoanhduy-uit/uav_search_target. New profile/group/tag distinguishes diagnostics version. Large checkpoints remainHF. No retrain/stop/delete existing jobs.
- Fresh CPU tests plus meaningful native/Warp packet parity, deterministic reset/rollout, BASE/PQL training/eval smoke. No claim of localCUDA verification without CUDA.
- Push tested commit required for Kaggle pinned-source launch; user explicitly authorized launching the new code. Choose accounts with verified live quota and no current GPUjob, different from current run accounts; initial candidates13–16.

## Task1: actual native/Warp network and mission telemetry
Existing communication/environment modules own capture; shared small pure-Python summary helper allowed for parity.
Return additions in info['network']['diagnostics'] (scalars plus fixed hist lists) and info['energy']/info['report_diagnostics'].
Required groups:
1 DATA/ACK attempts, successes/failures and DATA retries; observed TXpower mean/min/max + histogram and means split peer/GCS; count actual emissions including failed attempts (not just selected actions), ACK power separate.
2 On actual DATA PHY decode: SNR/SINR/RSSI and3Ddistance summary/count, LOS/NLOS counts+successes, peer/GCS counters. Mutually exclusive primary failure weak_signal(noise-onlySNR<threshold) vs interference(SNRpasses,SINRfails); ACK failures separate. Record overlap/collision counts with documented definitions.
3 Queuefull, packet lifetime, retrylimit, energydenial and CCA/backoff counters at actual decision sites, not inferred from outcome.
4 Successful custody peer transfers; direct unique GCS report count, relayed unique GCS report count, deliveryhop histogram (tailbinallowed); counts sum correctly with existing first-delivery semantics.
5 Action opportunity counts separate active/report-bearing/receiver-eligible from silent/no-opportunity. Explicitly define pre/postmotion observation timing.
6 Per-UAV cumulative propulsion/RF/total energy and battery values; targetfirstdiscovery/firstdelivery timestamps, firstdiscovery-to-GCS delay, unique discovered-undelivered with/withoutcustody, bufferpeak/oldestage. Reuse current record layout where present; no observation shape changes.
7 Lightweight calibration command/test or documented callable sweep demonstrates NLOS100m0.1Wfailure vs0.4Wsuccess, LOSnormal, and >500madmission rejection; seeded reproduci bility.
Testing: write new purposeful failing tests first, verify red, implement, verifygreen and applicable existingtests. Resetalltelemetry; no stale bleed on partial vector reset. Saved collector snapshots must carry newstate arrays. Native/Warp CPU compare meaningful counters. Capture overhead stays bounded O(envs*(N+T+constantbins)); no unbounded traces.
Ownership: communication/*,environment/*,associated tests; do not edit experiments/tracking/eval/configuration unless needed for state-compatible toggle, coordinate first.
Commit task1 code and tests when done; no push. Write test evidence/report to /tmp/uav_telemetry_20261016/task1-report.md.

## Task2: tracking/aggregation, asynchronous eval, deployment profile
Consume task1 scalar/hist contract; flatten diagnostics into descriptive W&Bkeys and include count/availability. Log per-UAV battery/energy, conditional action rates, pooled energyperGCSdelivery and raw denominators; deliverednone ->None.
Aggregate across complete vector/time cohorts with n, sumrawcounters for pooled PDR; distinguish delivery/success percentages and snapshotprogress from completedoutcomes. Avoid within-same-step return_drop diagnoses. Final_* compatibility allhistory; add explicit trailing/evaluationmetrics and dashboard columns/sections with labels.
Implement CPU async fixedcheckpoint evaluation manager reused by BASE/PQL. Cadence on globalinteractionsteps once perthreshold, onependingjob, unique immutable checkpoint, seed44 preserved, maxqueued1/backpressure status. Make evaluation failure/timeout visible without failing otherwisehealthytraining; drain/close workers cleanly. Respect cloud timebudget; no unbounded processwait. Mini-bank5samevalidation scenariosseed1000000044..1000000048; final30testseeds2000000044..2000000073unchanged anddisjoint. Addtracking.periodic_evaluation_seed/periodic_evaluation_episodesandevaluation_interval_steps. Updateevaluation_protocoltoaccuratelyrecordvalidationprovenance; noanimation. Add source_global_step/checkpoint/count/protocol and pooledcomparisonmetrics to online logs. Definition of evaluation axes must not confuse uploadtime with checkpoint interactiontime.
Review BASE/PQL replay telemetry and expose available sampleage/unsampled eviction/policy lag symmetrically where practical; never fabricate unsupported values.
New configs/comparison-25k-diagnostics.yaml: retain matched training parameters; enable newdiagnostics and periodic eval; main W&Bproject destination; groupuav-network-base-pql-25k-seed44-t35-o25-diagnostics-v1; tagdiagnostics-v1, eval5, finaleval30.
Tests: async lifecycle + immutable checkpoint, failure/timeout, same-step aggregate stability, meaningful BASE/PQL tinytrain+evalsmoke, configmatchesfourcases, optionalcounts/None semantics. Baselinefullsuite and fullsuiteafterintegration.
Ownership: experiments/*, configuration.py,newprofile, associatedtests/docs. Do not alter task1physics without routing a reviewedfix to its implementer.
Commit task2, no push. Report /tmp/uav_telemetry_20261016/task2-report.md.

## Task3: review/verify/publish/launch (controller)
Independent taskreview of eachdiff, then wholechange review. Fix substantive findings and rerun coveringtests. Baseline/fulltests mustbe read, not just agentclaims. Update docs changelog with units/denominators/performancecosts.
Check cleanGit,pushcommit, freshlivequota+kernels. Bootstrap privatecredentials via existinglauncher, secret-safe; verify no duplicateconfigon sameaccounts. Launch fourcases using newprofile and4otheraccounts with auto-continue andcpu-visualize. Read eachmanifest, Kagglestatus and W&B firstprogress/config/sourcecommit. If startupfails, diagnoseandfix onlythefailedcase; do not relaunch healthyjobs.
Final response: additions, tests/limits, fourcase/account/kernel/runID/links and liveverifiedstatus.

## Deployment transport regression
New private credential datasets are created successfully but whole-dataset ZIP download times out. Exactnamed-file download -f succeeds and contentwasverified(account13). ExtendTask2ownershipto cloud/client.py: downloadselectedfilenameevenwhenonlyonefile, preservingprivateowner/exactfileset/contentchecks. Addpurposefulregressiontest. Do not bypasscredentialverification.
