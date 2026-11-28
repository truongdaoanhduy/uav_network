# V124: Toàn bộ registry 99 conflict — nguyên nhân, trạng thái, bằng chứng và cách xử lý

**Ngày:** 28/11/2026. **Mốc:** V123 exact source; no new action; DS-only; MATD3/MASAC source untouched.

**LƯU Ý:** 99 mã là inventory lịch sử C01–C99, KHÔNG đồng nghĩa 99 bug đang mở. Các ticket có chữ `HISTORICAL_PATCHED` dựa trên log lịch sử, không phải chạy lại độc lập toàn bộ trong V124. Định nghĩa status trong JSON là cơ sở so sánh.

## 8 nhóm khoa học còn CHẶN đóng băng kịch bản

### G01_DS_TEMPORAL_DEPENDENCE
- **Nguyên nhân xung đột:** Mẫu cảm biến liên tiếp có thể tương quan; Dempster cộng bằng chứng lặp như độc lập, làm gia tăng xác nhận sai; cooldown chỉ là cơ chế cắt mẫu tùy chọn.
- **Cách giải quyết còn thiếu:** Cần hiệu chuẩn rho thời gian/không gian và đánh giá đúng-sai tại GCS theo nhiều seed, cả trường hợp cooldown 0 và 3/9.
- **Mã liên quan:** C04, C05, C22, C23, C35, C41, C49, C66, C85

### G02_SENSOR_QUALITY
- **Nguyên nhân xung đột:** Pd=.85, Pf=.03, mf=me=.8 và FoV/cấu trúc tòa nhà là giả định; chưa có hiệu chuẩn theo GSD, góc nhìn, tốc độ.
- **Cách giải quyết còn thiếu:** Đo/đối chiếu detector empirical hoặc báo rõ synthetic assumptions và sensitivity full mission.
- **Mã liên quan:** C23, C29, C35, C41, C87

### G03_RF_PROPAGATION
- **Nguyên nhân xung đột:** Mô hình RF attenuation 12 dB/vật cản, noise và model năng lượng không đo thực; cấu hình RF-stress đổi nhiều biến cùng lúc.
- **Cách giải quyết còn thiếu:** Chuẩn hóa RF/energy với phép đo hoặc theo profile được thừa nhận; ablation từng biến RF và report sensitivity.
- **Mã liên quan:** C12, C13, C20, C28, C30, C32

### G04_MAC_UPSTREAM
- **Nguyên nhân xung đột:** Warp UavNetSim-inspired DCF surrogate không đồng bộ events/airtime/frozen channel/packet size với upstream SimPy.
- **Cách giải quyết còn thiếu:** Matched packet trace/same traffic, so latency/PDR/queue/drop with upstream; công bố surrogate, không claim event-level parity.
- **Mã liên quan:** C07, C18, C24, C28, C33, C40, C60, C65, C92

### G05_CPU_CUDA_WHOLE_WORLD
- **Nguyên nhân xung đột:** Các gate CUDA stress được đo nhưng chưa có matched rollout đủ CPU-vs-CUDA cùng observations/DS/queue/reward/GCS delivery và random-stream.
- **Cách giải quyết còn thiếu:** Paired moving e.g. E4x300, compare all state/packet invariants, report tolerance separately CPU/FP32.
- **Mã liên quan:** C14, C19, C31, C36, C38, C53, C55, C63, C67, C68, C72, C78, C90, C93

### G06_NON_ORACLE_MISSIONS
- **Nguyên nhân xung đột:** Khả thi theo oracle planner hay trajectory cố định không chứng minh non-oracle mission success/generalization.
- **Cách giải quyết còn thiếu:** 100+ episode holdout/seed, no hidden target observations in actor, GCS target delivery, coverage, FP, latency and AUC.
- **Mã liên quan:** C08, C09, C18, C25, C27, C29, C34, C35, C64, C71

### G07_SUCCESS_SEMANTICS
- **Nguyên nhân xung đột:** GCS TP-ever latch có thể giữ success sau khi DS active posterior đổi bởi bằng chứng mâu thuẫn.
- **Cách giải quyết còn thiếu:** Chọn rõ ever-confirmed hay currently-confirmed mission success; counterexample unit tests và implement end-to-end.
- **Mã liên quan:** C25, C27, C39, C40, C49

### G08_TRAINING_POLICY_CONTRACT
- **Nguyên nhân xung đột:** Backend compact_3 comm API khác learned continuous XY; đầu học MATD3/MASAC chuẩn chỉ 2D; có legacy entrypoints và proposed hybrid action không thuộc baseline.
- **Cách giải quyết còn thiếu:** Pin action dimension=2, deterministic app routing, no learned comm power; freeze only after seven scientific gates.
- **Mã liên quan:** C10, C11, C19, C21, C26, C30, C38, C42, C43, C44, C45, C46, C48, C51, C54, C56, C57, C58, C59, C61, C64, C69, C70, C86, C91

## Từng conflict C01–C99 (không bỏ sót)

### C01 · STALE_TEST · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** V72 expects legacy 65m spawn; V105 S10 changed active 6-UAV spacing to 120m
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** test_v72... on Vast (Gates: không trực tiếp)

### C02 · CONFIG_CONFLICT · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** V95 test enables shared_coverage_reward=true without disabling first_visit; active validation rejects this intentionally
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** test_v95... on Vast (Gates: không trực tiếp)

### C03 · BUG · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** PEER OFF and data_mule ON changes ACK actor feature denominator by factor num_uavs=6 although no data-mule packets can be forwarded
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** test_v109_peer_mule_noop 2 GREEN + full suite (Gates: không trực tiếp)

### C04 · MODEL_LIMITATION · PARTIAL_CUTOFF_NOT_CALIBRATED
- **Tại sao conflict:** Dempster repeated evidence fusion assumes independent samples; correlated sensing can over-confirm
- **Bằng chứng / xử lý:** Runtime cooldown opt-in exists, but canonical 0 and temporal correlation uncalibrated.
- **Việc còn phải kiểm tra:** Empirical rho and false-confirm sensitivity. (Gates: G01_DS_TEMPORAL_DEPENDENCE)

### C05 · RESEARCH_CHOICE · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** confirm_th=.90 retained; .99 improves independent first passage but not sufficient with correlation
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** GCS-delivered false alarms and missed targets under 900-step routes (Gates: G01_DS_TEMPORAL_DEPENDENCE)

### C06 · RESOLVED_BUG · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** DS ignorance reward allowed repeat-hover exploitation, first_visit fixes novelty
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** extended trajectory/reward accounting (Gates: không trực tiếp)

### C07 · UNSUPPORTED_CLAIM · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** GPU MAC is a DCF surrogate, NOT identical SimPy UavNetSim
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** trace parity matched packet ID, retry, ACK, collision, TTL (Gates: G04_MAC_UPSTREAM)

### C08 · MODEL_LIMITATION · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** geometric reachability is not battery-constrained dynamic mission delivery
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** seeded path reachability by battery and RF stress (Gates: G06_NON_ORACLE_MISSIONS)

### C09 · INSUFFICIENT_EVIDENCE · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** S20 2x2 120-step means identical: no measured benefit of PEER ON despite flags
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** 5+ seeds and measurable treatment effect (Gates: G06_NON_ORACLE_MISSIONS)

### C10 · RESEARCH_CHOICE · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** TX power=0.1W fixed and disable learned TX; justified by action utility policy
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** RF sweep and PDR/energy Pareto before optional action (Gates: G08_TRAINING_POLICY_CONTRACT)

### C11 · PHASE2_BLOCKER · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** compact_3 discrete transmission mode exists alongside continuous XY; canonical MATD3/MASAC are not directly hybrid
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** phase 2 adapter interface spec, no algorithm edits yet (Gates: G08_TRAINING_POLICY_CONTRACT)

### C12 · MODEL_LIMITATION · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** steady-level Zeng-like power plus optional acceleration; not identical to aircraft measurement
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** in-flight 900-step GPU energy accounting and battery depletion (Gates: G03_RF_PROPAGATION)

### C13 · MODEL_LIMITATION · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** 12dB/building RF attenuation is synthetic scenario coefficient, not empirical material measurement
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** PHY/collision/PDR sensitivity ablation (Gates: G03_RF_PROPAGATION)

### C14 · OPEN_GAP · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** 4096x900 sticky-reset tests monitor sampled states, not exhaustive policy trajectories
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** action-policy stress; mission energy, radio and safety realism (Gates: G05_CPU_CUDA_WHOLE_WORLD)

### C15 · SCIENTIFIC_VALIDITY · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** actor semantic pooling is non-simplex (target max, empty min, unknown mean), not posterior BBA
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** CPU/Warp numeric feature parity under seeded observations (Gates: không trực tiếp)

### C16 · OPEN_GAP · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** active Warp DS-only; CPU supports legacy Bayesian ablation branch in code but YAML does not invoke it
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** runtime assertion logged per source and hashes (Gates: không trực tiếp)

### C17 · DOCUMENTATION_BUG · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** V108 GPU example names missing tests/test_v102_cuda_reset_sanitizer.py
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** corrected sanitizer result (Gates: không trực tiếp)

### C18 · OPEN_GAP · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** freeze requires pinned upstream packet trace comparison, 5 seeded ablations, cumulative endurance; V108 checklist claims only GPU gates needed
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** formal freeze evaluator with trace artifacts (Gates: G04_MAC_UPSTREAM, G06_NON_ORACLE_MISSIONS)

### C19 · INTERFACE_MISMATCH · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Legacy adapter RAW_DIM=514/DISC_DIM=7 versus active 549/3; canonical learner assumes alive obs[9] and mask obs tail[-3:], but active raw549 has alive obs[5], GCS distance obs[9], HELLO telemetry tail
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** RED/GREEN full actor-mask-decoder critic contract with 549/3 + replay CPU/Warp (Gates: G05_CPU_CUDA_WHOLE_WORLD, G08_TRAINING_POLICY_CONTRACT)

### C20 · EXPERIMENT_CONFOUND · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** The two named benchmark YAMLs change PEER, MULE and noise simultaneously (4.389577e-13 vs 4e-11 W). A difference between the two cannot be attributed causally to only PEER/MULE or only RF noise.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** paired-seed isolated network factorial (Gates: G03_RF_PROPAGATION)

### C21 · INACTIVE_REWARD_KNOBS · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** w_p2p=0 and lambda_b=0 make direct peer-information reward and airtime penalty zero in both active configs. Logging/protocol/flight-energy are still meaningful; do not claim these direct reward incentives are in force.
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** check direct reward sum and indirect GCS/energy effects (Gates: G08_TRAINING_POLICY_CONTRACT)

### C22 · DISABLED_AOI_DS_COUPLING · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** ds_aoi_eta=0 and raw_u16_count preserve evidence with no DS freshness discount. AoI telemetry and first-confirm latency reward remain; cannot claim to implement Xue 2026 full AoI-weighted DS.
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** AoI mediation ablation and comm-delay benefit (Gates: G01_DS_TEMPORAL_DEPENDENCE)

### C23 · PARAMETER_CALIBRATION_GAP · UNRESOLVED_SENSOR_CALIBRATION
- **Tại sao conflict:** Pd=.85 Pf=.03 define observation frequencies; fixed DS masses mf=me=.8 represent evidence strength but are not derived/estimated from these sensor characteristics. Not logically impossible; calibration absent.
- **Bằng chứng / xử lý:** DS mf/me=.8 and Pd/Pf values remain synthetic and not directly calibrated from true detector.
- **Việc còn phải kiểm tra:** Provenance and sensor reliability stress distributions. (Gates: G01_DS_TEMPORAL_DEPENDENCE, G02_SENSOR_QUALITY)

### C24 · CONTROL_PLANE_SEMANTICS · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** Comm action 0 SILENT means no new app DATA packet requested by actor; periodic HELLO/routing/MAC control packets can still transmit. Calling it radio-off/spectrum-off would be misleading.
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** control-plane traffic with all comm actions=0 (Gates: G04_MAC_UPSTREAM)

### C25 · FIRST_VISIT_VS_CONFIRMATION · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** First-visit exploration reward incentivizes new cells while DS threshold .9 with mf=.8 needs >=2 positive observations absent counterevidence; repeated visits may be needed. Potential tradeoff, not established defect.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** coverage-versus-verification reward ablation (Gates: G06_NON_ORACLE_MISSIONS, G07_SUCCESS_SEMANTICS)

### C26 · COMM_AS_ACTION_CLAIM · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Scenario exposes compact_3 comm action, but canonical pure-continuous learner may use heuristic comm rule rather than learning it. Thus cannot claim RL learns communication choice, MAC or routing when fixed local controller is used.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** end-to-end actor-controller semantics with mask and unit tests (Gates: G08_TRAINING_POLICY_CONTRACT)

### C27 · MODEL_SCOPE_GCS_TRUTH · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Ground truth is used by simulator to score TP/FP and determine episode success, appropriate for RL reward. Must prove truth not leaked to observation, policy route choice or critic.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** nonoracle contract audit all paths (Gates: G06_NON_ORACLE_MISSIONS, G07_SUCCESS_SEMANTICS)

### C28 · RF_TIME_DISCRETIZATION · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Active network_substeps defaults to 1: network runs first using current geometry, then motion/sensing. One-second mobility may change links within step; not necessarily bug, but is a fidelity approximation.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** channel-update discretization sensitivity (Gates: G03_RF_PROPAGATION, G04_MAC_UPSTREAM)

### C29 · STATIC_SCENE_GENERALIZATION · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Fixed 50m flight altitude, 15 static targets, 1km square, 5 buildings & two blackout zones, 900s are scenario choices, not proof of general SAR performance.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** geometry and instance sensitivity (Gates: G02_SENSOR_QUALITY, G06_NON_ORACLE_MISSIONS)

### C30 · REWARD_MISSING_NETWORK_COST · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** lambda_b=0 removes direct airtime penalty; TX battery energy still deducted. If the claim is direct latency-airtime-rate tradeoff in RL reward, the corresponding objective is not active.
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** radio energy use vs GCS utility Pareto (Gates: G03_RF_PROPAGATION, G08_TRAINING_POLICY_CONTRACT)

### C31 · RELEASE_GOVERNANCE · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** 587 pytest PASS and CUDA endurance gentle_patrol PASS do not certify scenario truth/physical correctness or RL interface. A reported freeze would be unsupported.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** independent decision of freeze evidence (Gates: G05_CPU_CUDA_WHOLE_WORLD)

### C32 · SOURCE_PROVENANCE · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Paper reports RF noise 4e-11 W; pinned config.py computes -174+10log10(22MHz)+7 dBm=4.389577e-13 W
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** full RF noise sweep (Gates: G03_RF_PROPAGATION)

### C33 · PACKET_AIRTIME_PARITY · UNRESOLVED_UPSTREAM_EQUIVALENCE
- **Tại sao conflict:** Warp evidence compact frame 280+32*entries us (~312us for one entry) vs UavNetSim paper 1024 byte payload (~4328us at 2Mbps); not same offered bytes
- **Bằng chứng / xử lý:** Warp compact packet and snapshot not the UavNetSim-v1 upstream.
- **Việc còn phải kiểm tra:** Build same traffic packet-level SimPy comparison; no false equivalence. (Gates: G04_MAC_UPSTREAM)

### C34 · NEGATIVE_EFFECT · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** Enabling PEER under RF-stress increases peer traffic/retry/drop and reduces mean GCS delivery 56.667% to 24.0%
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** positive partition relay workload with matched routes/bytes; distinct seeds (Gates: G06_NON_ORACLE_MISSIONS)

### C35 · DS_THRESHOLD_TRADEOFF · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** TH0.99 changes required p-n from 2 to 3; GCS false-ever 0.30->0 but delivery 98.0->95.333 in 20 worlds
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** correlated sensor mission test, multi-seed and 0.90/0.99 pre-registration (Gates: G01_DS_TEMPORAL_DEPENDENCE, G02_SENSOR_QUALITY, G06_NON_ORACLE_MISSIONS)

### C36 · ADVERSARIAL_STRESS_SCOPE · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Earlier gentle_patrol is insufficient; V112 adds 4096x900 random motion/boundary/churn with sticky all-step safety and overflow
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** read outcomes of 4 x 4096x900 CUDA tests (Gates: G05_CPU_CUDA_WHOLE_WORLD)

### C37 · PHYSICAL_STATE_COMMIT_BUG · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** CPU step originally computed swept-trajectory safety_violation but committed v_safe position/energy anyway on terminal invalid motion; a first fix broke legacy nonterminal partial sensing mode.
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Post-change full CUDA rollout and safety test action/initial-state sweep; inspect legacy nonterminal unsafe-commit if used outside canonical. (Gates: không trực tiếp)

### C38 · ACTION_ABI_MISMATCH · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** CPU step treats act_vel_cmd as m/s while GPU vectorized interface uses normalized XY controls; raw numbers are non-equivalent without adapter.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** CPU vs Warp paired one-step movement on same initial state, same real speed commands, no safety interventions (Gates: G05_CPU_CUDA_WHOLE_WORLD, G08_TRAINING_POLICY_CONTRACT)

### C39 · SUCCESS_SEMANTICS · UNRESOLVED_SUCCESS_CONTRACT
- **Tại sao conflict:** GCS success/reward use ever-true gcs_tp_latch even if current gcs_tp_active later clears on contradictory evidence.
- **Bằng chứng / xử lý:** GCS ever confirmed latch vs active DS has no finalized scientific success definition.
- **Việc còn phải kiểm tra:** Formal define and assert terminal condition under contradictory evidence. (Gates: G07_SUCCESS_SEMANTICS)

### C40 · BLACKOUT_RECOVERY_SEMANTICS · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Xue paper describes buffer flush at exit; current V113 sends app DATA only if action/route/TTL permit; exit alone is insufficient.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** Boundary exit scenario with forced no-route, then restored route: exactly-once GCS ledger/ack and bounded latency (Gates: G04_MAC_UPSTREAM, G07_SUCCESS_SEMANTICS)

### C41 · ZHANG_REPRODUCTION_SCOPE · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Zhang 2024 assumes fully connected fusion across UAV and MAX counters; active V113 disables application PEER and fuses SUM at GCS.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** Matched base-graph experiment, same sensor seeds, MAX vs SUM and PEER ON/OFF, report overhead and correlations (Gates: G01_DS_TEMPORAL_DEPENDENCE, G02_SENSOR_QUALITY)

### C42 · RUNNER_DEFAULT_RISK · IN_SCOPE_EXCLUDED_RUNNER
- **Tại sao conflict:** GPU training runner defaults to legacy v77 scenario and baseline-mode hybrid, not frozen V103 DS-only with canonical_continuous.
- **Bằng chứng / xử lý:** Legacy runner defaults may still choose v77/baseline-mode hybrid; phase2 not started.
- **Việc còn phải kiểm tra:** Canonical wrapper must pin current DS-only configs when phase2 begins. (Gates: G08_TRAINING_POLICY_CONTRACT)

### C43 · PRIORITY_NONORACLE_RISK · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** Proposed DS-value-aware priority scheduling must not use GCS private receiver ledger or true_targets; only local ACK/timestamp/route info is available.
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** No-oracle taint test for learned priority selector, ack-inference, loss/duplicate recovery (Gates: G08_TRAINING_POLICY_CONTRACT)

### C44 · ACTION_UTILITY_DEGENERACY · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** A learned send/no-send bit can be trivial if SEND whenever legal dominates HOLD because there is no explicit airtime reward penalty; MAC contention and battery still create implicit costs.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** Paired shadow action trials / regret against always-send, include AoI/bytes/delivery/false-confirm CI (Gates: G08_TRAINING_POLICY_CONTRACT)

### C45 · PROPOSED_ACTION_ABSENT · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** Proposed HOLD/NEW/PRIORITY action is NOT in compact_3; action index 2 is PEER_SEND not PRIORITY.
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** Typed action schema, mask+dispatch branch tests, queue priority tests; no claim of implementation before pass (Gates: G08_TRAINING_POLICY_CONTRACT)

### C46 · HYBRID_ALGORITHM_PROVENANCE · OPEN_INHERITED_NEEDS_ACCEPTANCE
- **Tại sao conflict:** Categorical communication scheduling with continuous XY changes the action distribution and policy gradient/objective relative to canonical continuous TD3/SAC.
- **Bằng chứng / xử lý:** No source-supported closure in this V124 audit; original root cause and prior evidence preserved.
- **Việc còn phải kiểm tra:** Independent mathematical audit of hybrid policy, entropy and Q action encodings, separate source and experiment labels (Gates: G08_TRAINING_POLICY_CONTRACT)

### C47 · ZHANG_MOTION_DYNAMICS_DIFFERENCE · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** Zhang continuous action is control force/acceleration integrated into velocity; active V113 action is desired XY velocity projected by kinematic safety shield.
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** Equation/units comparison, physical track parity with common acceleration commands if reproduction required (Gates: không trực tiếp)

### C48 · LAUNCHER_MISSING_V83_ACK · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** New canonical wrapper did not expose/forward downstream mandatory --allow-experimental-physics flag, so launch stops before Warp initialization.
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** GPU 1-env/64-env launch with explicit opt-in; verify no changes to physics (Gates: G08_TRAINING_POLICY_CONTRACT)

### C49 · EVENT_DRIVEN_REPEATED_DATA · SCOPE_CHOICE_OR_DOCUMENTED
- **Tại sao conflict:** Local sensor re-observations renew dirty timestamp every step; event-driven sends once per step when route exists even for static targets, creating repetitive DATA without improvement in short stationary mission.
- **Bằng chứng / xử lý:** Documented scope/research-only choice, not necessarily a runtime defect.
- **Việc còn phải kiểm tra:** Paired 5+ seeds 900 steps; same XY, RF stress, GCS TP latency, PDR, AoI, airtime and energy (Gates: G01_DS_TEMPORAL_DEPENDENCE, G07_SUCCESS_SEMANTICS)

### C50 · PACKET_CAP_MISREAD · HISTORICAL_PATCHED_NOT_RETESTED_IN_V124
- **Tại sao conflict:** The number 62 limits entries per transmitted packet, not dirty-cell cardinality. Saying priority cannot matter because one packet holds everything is a false universal inference.
- **Bằng chứng / xử lý:** Kept prior source/test evidence verbatim; no new isolated V124 reproduction for this specific ticket.
- **Việc còn phải kiểm tra:** Dirty entries histogram under 2 blackouts+RF; compare GCS first confirmation and freshness (Gates: không trực tiếp)


> Các mục C51–C99 nằm trong tài liệu tiếp theo: `V124_CONFLICTS_C51_C99_VI.md`.
