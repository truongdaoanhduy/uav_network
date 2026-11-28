# V124 — Conflict C51–C99 (tiếp phần C01–C50)

### C51 · WRAPPER_SCENARIO_SCOPE · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** The V115 wrapper accepted DS-only but arbitrary scenario e.g. RF stress + PEER enabled and could silently claim canonical base. Direct invocation of legacy runner still defaults V77/hybrid.
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** GPU dry-run and real launch via wrapper; grep provenance recorded in result.json (Gates: G08_TRAINING_POLICY_CONTRACT)

### C52 · STALE_ACTIVE_CURRENT_TASK_BAYES · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** In V115 current-task still called v91 Bayesian scenario active even though v103 DS YAML is active. Future maintenance/agents could run Bayes by mistake.
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Run test_v116_current_task_ds_contract.py with manifest validation (Gates: không trực tiếp)

### C53 · V116_GPU_ENV_NOT_READY · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Accessible Vast NVIDIA RTX3060 12GiB and V113 core sources match V116 hashes; available Python environment does not import torch/warp, so cannot claim fresh V116 CUDA gate.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** Post-V113 4096x900 four stress modes, safety faults, packet counters, time, VRAM (Gates: G05_CPU_CUDA_WHOLE_WORLD)

### C54 · ALTITUDE_ACTION_NOT_IMPLEMENTED · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Adding a third continuous v_z head to current fixed-altitude 50m model would be dead; manual act_dim=-3 without physics redesign is invalid.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** Vertical sensitivity and nontrivial action effect; same-fixed-seed ablation (Gates: G08_TRAINING_POLICY_CONTRACT)

### C55 · POST_V113_FULL_STRESS_ABSENT · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** V112 four CUDA 4096x900 stress tests predate V113 safety terminal patch C37; V113 full pytest 593pass does not automatically certify post-patch 4 stress families.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** Post-change four scenarios/adverse actions, source SHA, raw log exit and invariant suite (Gates: G05_CPU_CUDA_WHOLE_WORLD)

### C56 · CPU_ONLY_LEARNED_ACTION_RUNTIME · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Action evidence/relay only wired into CPU env.step; Warp mission.step still takes XY + original comm only.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** CPU/Warp same-seed action event parity, 4096-env 900-step stress (Gates: G08_TRAINING_POLICY_CONTRACT)

### C57 · NO_HYBRID_RL_POLICY · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Canonical MATD3 and MASAC learn continuous XY; two categorical action heads are not part of the learner; no trained experimental policy.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** RL-gradient / action mask / sampled categorical correctness; paired trained seeds (Gates: G08_TRAINING_POLICY_CONTRACT)

### C58 · PRIORITY_SCORE_UNCALIBRATED · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** DS+AoI priority uses engineering score 0.65*mT +0.35*min(age/120,1); these numbers are NOT Zhang/Xue equations.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** Sensitivity 0.25/0.5/0.75 DS, age 30/60/120s; generalization (Gates: G08_TRAINING_POLICY_CONTRACT)

### C59 · BACKLOG_REQUIRED_FOR_PRIORITY · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** If dirty entries <=62, all priorities may transmit identical content, making scheduling choice ineffective.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** 5-seed×900 dynamic UAV scenarios; effect size vs newest-first (Gates: G08_TRAINING_POLICY_CONTRACT)

### C60 · RELAY_HELLO_STALE · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Relay candidates depend on received HELLO routes with 2-s TTL; a valid choice can still be physically disconnected or congested.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** 5seed moving network path selection versus shortest-hop; drop/loop/TTL (Gates: G04_MAC_UPSTREAM)

### C61 · ACTOR_VIEW_CPU_ONLY · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Experimental augmented actor has 569 features (549+4+8+8), cannot be passed to legacy 553-input canonical actor or Warp 549 without an adapter.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** Hybrid actor/critic dimension and masks validated CUDA + PyTorch (Gates: G08_TRAINING_POLICY_CONTRACT)

### C62 · RELAY_NOT_PEER_BELIEF_FUSION · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** Next-hop changes forwarding route of GCS-bound DATA; it does NOT distribute DS belief into a peer map as Zhang paper assumes.
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** Zhang MAX fully-connected ablation vs peer exchange (Gates: không trực tiếp)

### C63 · CANONICAL_NO_REGRESSION_SCOPE · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** New route cache must not change original baseline packet routing when no experimental actions supplied.
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** >=5seed×900 traffic traces and Warp parity (Gates: G05_CPU_CUDA_WHOLE_WORLD)

### C64 · NO_END_TO_END_LEARNED_DELIVERY_GAIN · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** 5-seed synthetic action effects demonstrate controllability, NOT better learned AoI/GCS delivery on natural 15-target mission.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** Five seeds or more×4 worlds×900, paired bootstrap and zero-oracle constraints (Gates: G06_NON_ORACLE_MISSIONS, G08_TRAINING_POLICY_CONTRACT)

### C65 · NETWORK_FRAMES_UPSTREAM_UNPROVEN · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** UavNetSim-inspired MAC uses compact DATA and custom timing, not byte/event-equivalent to UavNetSim SimPy.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** MAC/PHY event parity with same packet trace (Gates: G04_MAC_UPSTREAM)

### C66 · DS_INDEPENDENCE_AND_FP · PARTIAL_CUTOFF_NOT_CALIBRATED
- **Tại sao conflict:** SUM of repeated independent evidence can overconfirm if sensor samples correlated. Switching action modes does not fix it.
- **Bằng chứng / xử lý:** Runtime cooldown opt-in can suppress repeated samples but is not DS dependence proof.
- **Việc còn phải kiểm tra:** Calibrate empirical dependence or bound with cautious combination. (Gates: G01_DS_TEMPORAL_DEPENDENCE)

### C67 · POSTCHANGE_WARP_SAFETY_GATE · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** V112 GPU 4096×900 stress predates V113 safety fix; V117 CPU-only action code has no GPU implementation.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** 4×(4096env×900steps) with no overflow/nonfinite/safety faults (Gates: G05_CPU_CUDA_WHOLE_WORLD)

### C68 · MOTOR_NORMALIZATION_GPU_ABI · PATCHED_HISTORIC_PRESTATE_LIMITED
- **Tại sao conflict:** V118 per-axis scaling of normalized actions before acceleration differs from CPU pre-normalized disk at nonzero previous velocity
- **Bằng chứng / xử lý:** Prior CPU-CUDA projection near-tie corrected; no new full world parity proof.
- **Việc còn phải kiểm tra:** Perform full matched-scenarios CPU/Warp state parity. (Gates: G05_CPU_CUDA_WHOLE_WORLD)

### C69 · CANONICAL_WRAPPER_MISLABEL · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** V118 wrapper did not pin scenario/learner hashes or verify pure-GCS flags; arbitrary alternate DS YAML could be labeled canonical
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** end-to-end learner smoke after freeze (Gates: G08_TRAINING_POLICY_CONTRACT)

### C70 · BROKEN_FOUR_STRESS_RUNBOOK · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** V118 documented command omitted required --output and only one mode rather than 4
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** compare actual output receiver metrics (Gates: G08_TRAINING_POLICY_CONTRACT)

### C71 · NATURAL_ACTION_STUDY_REPRODUCIBILITY · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** V118 3-seed/300-step summary lacks per-seed raw packet traces and script to reproduce its original V117 action experiment
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** reproduce natural 900 step multi-seed raw traces only if trying to claim generality (Gates: G06_NON_ORACLE_MISSIONS)

### C72 · RESET_STRESS_SCOPE · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** 4x 4096x900 tests use reset semantics, monitor world episodes; not each of 4096 worlds untruncated for 900 consecutive steps
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** sticky 900-step episode assertions and statistical mission outcomes (Gates: G05_CPU_CUDA_WHOLE_WORLD)

### C73 · VALIDATION_BUG · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** V120 command used --worlds although legacy gate only supports --envs; and its unconditional auto-reset invalidates uninterrupted episode claim
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Review historical evidence (Gates: không trực tiếp)

### C74 · UNIT_BUG · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** Sector controller returned normalized length 1 but CPU env.step expects XY velocity in m/s (15 m/s max)
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Review historical evidence (Gates: không trực tiếp)

### C75 · CPU_MAC_RACE · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** Simultaneous DIFS/backoff expiries were committed sequentially by radio index; hidden terminals could falsely avoid collisions
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Review historical evidence (Gates: không trực tiếp)

### C76 · AIRTIME_ACCOUNTING · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** HELLO peer-telemetry bytes not charged on CPU; Warp V89 already charged telemetry
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Review historical evidence (Gates: không trực tiếp)

### C77 · DS_RUNTIME_BYPASS · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** Legacy CPU config and direct runtime override could enter Bayesian belief despite DS-only active YAML
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Review historical evidence (Gates: không trực tiếp)

### C78 · CPU_CUDA_TIE · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** FP32 vs FP64 near equal projection costs produced distinct candidate enumeration, up to 2.87m/s
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Review historical evidence (Gates: G05_CPU_CUDA_WHOLE_WORLD)

### C79 · TEST_FIXTURE_SCOPE · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** Test invoked geometry-only projection check_peers=0 yet expected CPU full three-pass peer shield result
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Review historical evidence (Gates: không trực tiếp)

### C80 · STALE_METADATA · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** Active project documentation called obsolete Bayes configuration despite DS-only
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Review historical evidence (Gates: không trực tiếp)

### C81 · SCIENTIFIC_CLAIM · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** V120 claimed only two GPU checks needed to freeze despite correlated DS and full CPU/GPU/SimPy parity gaps
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** Reproduce and close with scoped test (Gates: không trực tiếp)

### C82 · FEASIBILITY_CLAIM · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** Straight-line target-aware path ignoring obstacles/routing falsely described as end-to-end mission feasibility
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Review historical evidence (Gates: không trực tiếp)

### C83 · PHYSICS_FORMULA_BUG · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** zeng_propulsion_power_with_accel_w computed kappa**1.5 * sqrt(1/(hypot(1,x)+x)) instead of exact Zeng 2019 Appendix A Eq. (69) kappa * sqrt(kappa^2/(hypot(kappa,x)+x)), underestimating induced power by 2.35 W at v=10 m/s, a=5 m/s^2
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Run pytest on Vast.ai RTX 3060 (Gates: không trực tiếp)

### C84 · CPU_GPU_PARITY_BUG · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** On terminal safety_violation (safety_failure_terminate=True), Warp _move_v80 zeros intervention[e]=0.0, whereas CPU transition.py kept nonzero shield_intervention and subtracted extra r_blocked_motion penalty
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Run pytest on Vast.ai RTX 3060 (Gates: không trực tiếp)

### C85 · DS_TEMPORAL_CORRELATION · PATCHED_OPT_IN_REMAINS_SCIENTIFIC_OPEN
- **Tại sao conflict:** ds_correlation_mitigation.py only referenced in stand-alone calibration and tests, not in CPU/Warp production sensing
- **Bằng chứng / xử lý:** CPU+Warp sense loop now checks last (UAV,cell) accepted tick with configurable cooldown. CUDA and Warp-CPU microtest 22-22-22-44 when cooldown3. Canonical cooldown0 unchanged.
- **Việc còn phải kiểm tra:** Cần xác định rho/TP/FP để chọn cooldown thực nghiệm, không tự động bật trong YAML. (Gates: G01_DS_TEMPORAL_DEPENDENCE)

### C86 · FREEZE_GATE_UNSOUND · PATCHED_FAIL_CLOSED_FROZEN_FALSE
- **Tại sao conflict:** V122 docs state scenario can be frozen after only pytest on Vast
- **Bằng chứng / xử lý:** V124 release gate outputs scenario_frozen:false; eight scientific blockers and no Phase2.
- **Việc còn phải kiểm tra:** Independent review per G01-G08; never infer freeze from pytest. (Gates: G08_TRAINING_POLICY_CONTRACT)

### C87 · OPTICAL_SENSOR_UNCALIBRATED · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** Optical cone azimuth pilot has geometric direction and slew but inherits constant pd=.85 and pf=.03 for all off-nadir angles
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** Reject action from canonical; require GSD/resolution-aware Pd/Pf measured or literature-calibrated and paired benchmark to promote (Gates: G02_SENSOR_QUALITY)

### C88 · ACTION_TRANSACTIONALITY · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** Pilot initially committed gimbal orientation before validating motion/comm inputs
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Validate action shapes first, commit orientation and network side effects after validation; no Warp pilot implementation (Gates: không trực tiếp)

### C89 · BAYES_LEGACY_ARTIFACTS · DOCUMENTED_INACTIVE_BAYES_ONLY
- **Tại sao conflict:** Legacy bayes_prior_occupancy dataclass field and mission_v91_bayesian.py archived, although DS-only runtime rejects alternate belief_inference_mode
- **Bằng chứng / xử lý:** Canonical DS-only enforces inference mode; deprecated bayes fields remain for compatibility only.
- **Việc còn phải kiểm tra:** Remove legacy only with separate compatibility review; no Bayes runtime in canonical. (Gates: không trực tiếp)

### C90 · CPU_WARP_STATE_PARITY · PARTIAL_BOUNDED_GPU_GATE
- **Tại sao conflict:** Stationary Warp 4096 x900 performance and shield prestate parity are not whole-world transition parity
- **Bằng chứng / xử lý:** V124 GPU port gentle_patrol E4096x900 no-reset PASS, RFstress E4096x900 reset-mode PASS. This is not full CPU-GPU world parity.
- **Việc còn phải kiểm tra:** Cần same-seed matched position+observations+DS+radio+reward rollout. (Gates: G05_CPU_CUDA_WHOLE_WORLD)

### C91 · ACTION_ABLATION_DEGENERACY · REJECTED_ACTION_NO_CHANGE
- **Tại sao conflict:** Proposed learning nonmotion camera azimuth on 6 UAV map risks increasing action dimensions with no demonstrated gain
- **Bằng chứng / xử lý:** Counter-keyed CPU pilot retained nadir canonical higher GCS TP than naive gimbal variants.
- **Việc còn phải kiểm tra:** No camera/altitude/routing/TX/hold action to continuous canonical baseline. (Gates: G08_TRAINING_POLICY_CONTRACT)

### C92 · MAC_PARITY_AND_TRAFFIC · UNRESOLVED_UPSTREAM_EQUIVALENCE
- **Tại sao conflict:** Custom 1s PHY snapshot and 70+8kB compressed DS packet differ from upstream packet-level UavNetSim SimPy 1082B and event scheduling
- **Bằng chứng / xử lý:** Differences of airtime, backoff, HELLO and channel 1s time snapshot persist.
- **Việc còn phải kiểm tra:** Provide profile comparison, possible calibrated surrogate error bounds. (Gates: G04_MAC_UPSTREAM)

### C93 · RF_GATE_LABEL · PATCHED_TRUE_RF_GPU_GATE
- **Tại sao conflict:** V123 repeated CUDA RF-only with noise=4e-11 using Base YAML; not full v105 PEER relay RF stress
- **Bằng chứng / xử lý:** Actual scenario_v105_ds_rf_stress_1km.yaml PEER/MULE stress on RTX3060 PASS_STRESS_RESET_MODE E4096x900.
- **Việc còn phải kiểm tra:** Cần không-reset exact RF-stress non-oracle, peer airtime parity and fidelity. (Gates: G05_CPU_CUDA_WHOLE_WORLD)

### C94 · PAIRED_CAUSAL_DESIGN · PATCHED_CPU_PAIRED_PILOT
- **Tại sao conflict:** Camera comparison keeps seed+trajectory but consumes RNG with different visible-cell sets, making stochastic observations not cell-tick paired
- **Bằng chứng / xử lý:** CPU counter-keyed RNG keyed by seed,UAV,cell,tick; 5seed camera pilot TP nadir 4.0 vs east1.0 sweep1.2. GPU keyed mode explicitly rejects.
- **Việc còn phải kiểm tra:** Thêm CTDE/Warptest chỉ nếu đưa sensor mode vào canonical; không thêm action dựa trên pilot. (Gates: không trực tiếp)

### C95 · DS_DUPLICATE_PROVENANCE_RECORD · PATCHED_OPT_IN_SCIENTIFIC_OPEN
- **Tại sao conflict:** Research DS mitigation module accepted identical per-UAV/cell/timestamp records twice, and failed to reject invalid indices or reversed time.
- **Bằng chứng / xử lý:** Added duplicate/time/index/origin validation and 17 tests; module remains optional research helper.
- **Việc còn phải kiểm tra:** Caution: Warp kernel does not use this helper; runtime cooldown separately tested. (Gates: không trực tiếp)

### C96 · DS_RUNTIME_CPU_WARP_COOLDOWN · PATCHED_OPT_IN_SCIENTIFIC_OPEN
- **Tại sao conflict:** Prior offline correlation mitigation was not connected to real sensing; repeated local observations were counted every tick even under correlated error.
- **Bằng chứng / xử lý:** Config opt-in cooldown0 default; wired CPU and Warp sense kernel, tested real Warp CPU and CUDA counts [22,22,22,44] vs no cooldown [22,44,66,88].
- **Việc còn phải kiểm tra:** Still open science: correlated error model, false positive and true positive trade-off, cooldown choice; not enabled in canonical. (Gates: không trực tiếp)

### C97 · PAIRED_CAMERA_RNG_CONFOUNDER · BOUNDED_TESTED_NOT_FROZEN
- **Tại sao conflict:** Changing visible camera cells consumed a different sequential random draw stream, invalidating naive seed-only paired comparisons.
- **Bằng chứng / xử lý:** Counter-keyed per-(seed,episode,tick,UAV,cell) RNG option for CPU controlled pilot; 5-seed, three-arm JSONL. GPU fails closed.
- **Việc còn phải kiểm tra:** Still needs scene-independent detector/gimbal calibration if revisited. No new canonical action. (Gates: không trực tiếp)

### C98 · RF_PEER_GPU_GATE_MISLABEL · BOUNDED_TESTED_NOT_FROZEN
- **Tại sao conflict:** V123 CUDA RF-only was Base YAML with injected noise; it did not test RF-stress YAML peer/data mule, and no moving no-reset gate.
- **Bằng chứng / xử lý:** V124 GPU port ran v105 true RF PEER/MULE E4096x900 stress reset-mode PASS and gentle_patrol E4096x900 no-reset PASS.
- **Việc còn phải kiểm tra:** Not same test: no-reset gentle patrol is v103 Base; v105 stress does reset, and mission metrics are not learning evidence. (Gates: không trực tiếp)

### C99 · SCIENTIFIC_FREEZE_GOVERNANCE · PATCHED_OPT_IN_SCIENTIFIC_OPEN
- **Tại sao conflict:** Unit tests and stress workload passing cannot independently establish physical/DS/MAC validity or authorise MATD3/MASAC training.
- **Bằng chứng / xử lý:** Added fail-closed V124 freeze readiness gate listing G01–G08; dry run exit 2, scenario_frozen=false.
- **Việc còn phải kiểm tra:** Resolve and independently review eight named gates; leave scenario_frozen=false and phase2=false. (Gates: không trực tiếp)

## Cơ sở học thuật, chỉ viện dẫn đúng phạm vi
- Dempster–Shafer: Zhang et al., IEEE T-IV 2024 https://doi.org/10.1109/TIV.2024.3352581; cảnh báo dependent evidence khác i.i.d. (Denoeux 2008) https://www.sciencedirect.com/science/article/pii/S0004370207001063. Cooldown là heuristic triển khai V124, không phải công thức được paper chứng nhận.
- UavNetSim-v1 upstream: https://github.com/Zihao-Felix-Zhou/UavNetSim và https://arxiv.org/abs/2507.09852 ; Warp DCF-inspired surrogate ≠ SimPy identical.
- Rotary-wing energy: Zeng et al., IEEE TWC 2019 https://doi.org/10.1109/TWC.2019.2902559 ; các hệ số vật lý đang synthetic.
- Sensor imaging: Zwick et al., Sensors 2023 https://doi.org/10.3390/s23020664 ; Pd/Pf không tự suy ra từ hình học không có đo.

Các paper trên hỗ trợ phương pháp/mô hình, KHÔNG chứng minh thông số của project chính xác. Chưa kiểm chứng chính thức quartile theo năm và phân ngành; không gán mọi công trình là Q1/A*.

## Đọc báo cáo vận hành và log
- `docs/V124_RELEASE_AND_SCIENTIFIC_BLOCKERS_VI.md`: thay đổi source, CPU/GPU kết quả, điều kiện nghiệm thu.
- `outputs/v124_conflict_audit/V124_CONFLICT_REGISTRY_C01_C99.json`: đầy đủ trường gốc V123 và overlay V124, CSV đi kèm.
- `outputs/v124_conflict_audit/V123_to_V124_source.diff`: toàn bộ patch so với V123.
- `outputs/v124_conflict_audit/V124_CPU_AND_GPU_LOG_MANIFEST.json`: log nào trong ZIP, log nào trên Vast/Ubuntu riêng.
