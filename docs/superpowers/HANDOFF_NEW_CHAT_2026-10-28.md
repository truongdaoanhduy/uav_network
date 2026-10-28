Bạn tiếp quản dự án UAV/RL dưới đây từ một chat trước. Hãy dùng bản bàn giao này để đọc đúng source và tiếp tục công việc tôi yêu cầu ở cuối prompt. Tôi muốn bạn làm việc trực tiếp, kiểm chứng bằng bằng chứng, hoàn tất trong phạm vi đã giao; không chỉ đưa ra kế hoạch rồi dừng.

# 1. Cách tiếp quản bắt buộc
1. Đọc AGENTS.md và phần MỚI NHẤT ở đầu docs/superpowers/CURRENT_TASK.md trước khi sửa code hoặc launch job.
2. Kiểm tra repo, branch, HEAD, trạng thái worktree và remote thực tế. Các số liệu/tài khoản/quota/trạng thái cloud trong prompt là snapshot có thời điểm, không phải trạng thái luôn đúng.
3. Nếu HEAD mới hơn mốc dưới đây, đọc diff/commit mới và ưu tiên source + hồ sơ mới nhất. Các phần “Previous completed task” và tài liệu ngày cũ có thể chứa yêu cầu đã được thay thế.
4. Khôi phục ngữ cảnh và đọc các file liên quan theo bản đồ code dưới đây; không cần đọc mọi file không liên quan trước một sửa đổi nhỏ.
5. Nếu cuối prompt chưa có nhiệm vụ mới cụ thể, hãy xác nhận đã tiếp quản và nêu trạng thái đã kiểm tra. Prompt bàn giao tự nó không yêu cầu chạy lại training hay thay đổi scientific config.

# 2. Mục tiêu dự án và yêu cầu của tôi
- Dự án uav_network: mô phỏng UAV 2D tìm mục tiêu, truyền báo cáo qua mạng UAV/relay tới GCS, xét năng lượng, chuyển động an toàn và truyền thông ở mức packet.
- Thuật toán chính MASAC và MATD3; hai cách chạy BASE và PQL-inspired. PQL là kiến trúc collector/critic/policy workers, không phải thuật toán thứ ba độc lập để tự ý đổi tên.
- So sánh thêm cách bố trí replay HYBRID CPU75/GPU25 với FULLGPU. “replay/relay 1 triệu” trong nhiều tin nhắn cũ của tôi thường là replay buffer 1.000.000 transitions; phân biệt với mạng relay UAV.
- Tôi muốn full flow cloud: launch Kaggle GPU -> thấy learner thật sự chạy và metrics online W&B -> có thể tắt laptop/sandbox -> GitHub Actions tiếp tục giám sát/continuation/recovery -> train đủ budget -> Kaggle CPU eval -> visualize/video -> publish W&B/HF.
- Việc W&B hiện “running” riêng lẻ chưa chứng minh full flow độc lập sandbox đã sẵn sàng.
- Tôi muốn logs đủ ngữ cảnh để khoanh vùng crash/failed, train yếu, suy giảm, chậm/đứng, lỗi tài nguyên, checkpoint, eval và publication. Không thể bảo đảm biết mọi nguyên nhân chỉ từ logs.
- Giao diện cần ít cột/plot nhưng có ý nghĩa. Personal Workspace là nơi tôi muốn nhìn; Algorithm và Mode đã ẩn vì tên run có thông tin đó.
- Tôi thích bạn tự làm các bước cần thiết trong phạm vi yêu cầu đã rõ, không hỏi lại các lựa chọn triển khai thông thường. Hỏi khi thiếu thông tin quyết định hoặc sắp làm việc thật sự phá hủy/ngoài phạm vi.

# 3. Repo, môi trường và mốc version
GitHub:
https://github.com/truongdaoanhduy/uav_network

Repo làm việc trong Sandbox trước:
  /home/aduy/Documents/NCKH/uav_network-public-hf-20261022

Đây là worktree/checkout đang dùng; cần bảo toàn file và thay đổi của người dùng. Local scratch của công cụ ChatGPT là filesystem khác, không mặc định chứa repo hay key.txt.

Nhánh làm việc:
  feat/cpu-replay-20261022

Nhánh triển khai:
  main

Mốc đã xác minh trước khi tạo tài liệu bàn giao:
- Runtime/source cuối đã chạy full suite: 56604cc71f294cd94e5251491ab7d1397a95200b
- Báo cáo hoàn tất: 95c4fe5b972e3ecb943168790557337749119fad
- Hồ sơ xác nhận integration: b11a600993f0c9a5531ae685ec95c6d9bac561ab
- main và feat/cpu-replay-20261022 đã cùng ở b11a600 khi kết thúc audit; worktree sạch.
- Tài liệu prompt và đính chính lineage được thêm sau đó bằng thay đổi tài liệu, không sửa runtime. Lấy HEAD mới thực tế, không reset về mốc cũ chỉ vì prompt ghi hash cũ.

Python project yêu cầu >=3.11. Xem pyproject.toml để lấy dependency thật. Có extras cloud/gpu; Warp được pin trong project. Không tự thay phiên bản dependency theo trí nhớ. Phiên Python/Torch trên Sandbox và Kaggle có thể khác nhau: ghi nhận môi trường đo được, không coi chúng là một.

Nếu Sandbox cũ không truy cập được: dùng GitHub đọc source/hồ sơ trước, hoặc checkout riêng khi công cụ cho phép. key.txt, outputs và phiên đăng nhập không có sẵn chỉ vì clone được GitHub; hãy nói rõ khả năng nào đã truy cập được.

# 4. Tài liệu nguồn cần đọc theo nhiệm vụ
Luôn đọc:
- AGENTS.md
- docs/superpowers/CURRENT_TASK.md

Logs/metrics/W&B:
- docs/training-diagnostics.md
- docs/diagnostic-coverage.md
- docs/diagnostics-v1.md
- docs/superpowers/evidence/2026-10-28-diagnostic-coverage.md
- Các JSON cùng prefix 2026-10-28-diagnostic-coverage-:
  tests, personal, browser, online-probe, service, quality-example, review, integration.
- docs/superpowers/specs/2026-10-28-diagnostic-coverage-design.md
- docs/superpowers/plans/2026-10-28-diagnostic-coverage.md
- docs/superpowers/evidence/2026-10-28-diagnostic-coverage-ledger.md

Full flow / GitHub Actions / recovery:
- docs/github-actions-full-flow.md
- docs/superpowers/evidence/2026-10-27-github-external-supervisor.md
- docs/superpowers/evidence/2026-10-27-general-failure-hardening.md
- docs/superpowers/evidence/2026-10-27-crash-flow-recovery.md
- docs/superpowers/specs/2026-10-27-github-external-supervisor-design.md
- docs/superpowers/plans/2026-10-27-github-external-supervisor.md

Acceleration / CUDA / performance:
- docs/superpowers/evidence/2026-10-27-measured-acceleration-policy.md
- docs/superpowers/evidence/2026-10-27-acceleration-policy-kaggle-smoke.md
- Các evidence 2026-10-24 về 1kb matched acceleration và PQL isolation khi cần so sánh nguồn đo.

CPU replay:
- docs/superpowers/CPU_REPLAY_NEXT_TASK.md là brief LỊCH SỬ, có mục pending từ giai đoạn cũ. Đối chiếu CURRENT_TASK, source/tests và evidence mới trước khi kết luận còn dang dở hay chạy lại thử nghiệm.
- Không mở lại một nhiệm vụ đã hoàn tất chỉ vì một checkbox của tài liệu cũ chưa được cập nhật.

# 5. Cấu hình khoa học hiện tại và các khái niệm dễ nhầm
Profile canonical:
  configs/comparison-20k-hybrid75-1kb.yaml

Snapshot source:
- World generation_version=3, 35 targets, 25 obstacles, max_steps=1800, motion_safety=dynamic_window.
- report_bytes=1024, payload_bytes=1024, buffer_bytes=3072. Tên “1kb” của file thực tế chỉ 1024 BYTES ở các trường này; tránh diễn giải thành 1024 bits.
- report_ttl_s=300, max_retries=5; propagation_delay, receiver_admission_mask, power_limited_admission và link_state_observation đều bật.
- peer_range_m=gcs_range_m=500.
- tx_power_min_w=0.001; max_w=0.4; mapping log; PA efficiency=0.35.
- Hai thuật toán đều hidden_sizes=[256,256], replay_capacity=1.000.000, batch_size=8192, learning_starts=131072, updates_per_step=8.
- runtime.environment_backend=warp_packet; num_envs=4096; seed=44.
- episodes=20000; max_train_seconds=36000; log_interval_steps=4096; checkpoint_interval=128.
- replay_device=cpu, replay_cpu_fraction=0.75 là profile hybrid; không đổi thành 100% CPU chỉ vì replay_device ghi cpu.
- eval_episodes=30; eval_seed=2000000044.
- tracking.evaluation_interval=0 và evaluation_interval_steps=0: MINI-EVAL ĐỊNH KỲ ĐANG TẮT.
- periodic_evaluation_episodes=5 / periodic_evaluation_seed=1000000044 tồn tại nhưng không có lịch chạy khi interval bằng 0.
- pql.enabled=false trong YAML; bật theo mode ở launcher/overrides. PQL mode async, worker_timeout_s=600, checkpoint_replay=true.
- tracking.checkpoint_store=huggingface; checkpoint_repo=duy95/uav-search-checkpoints.

Các giá trị trên là snapshot canonical, không thay thế resolved_config/manifest của từng job. Profile FULLGPU hoặc BASE/PQL cần xem overrides thật. Giữ profile 1MB cũ riêng để reproducibility/ablation, không thay nó bằng 1KB âm thầm.

Phân biệt:
- env step / joint transition / collection step / optimizer update / completed episode là các counter khác nhau.
- completed episodes=0 vẫn có thể đang tiến triển trong nhiều env song song.
- time-budget kết thúc một segment khác với đủ 20000 episodes.
- policy warm-start khác exact training resume.
- training outcome khác validation/final CPU evaluation.
- packet PDR khác tỷ lệ target reports được đưa tới GCS.
- return/loss/throughput không phải mỗi cái tự chấm policy tốt/xấu.

# 6. Bản đồ code
Configuration và CLI:
- uav_network/configuration.py: dataclass/config loading/validation.
- uav_network/__main__.py, experiments/cli.py: train/config/evaluate/compare/hardware/dashboard/kaggle.
- cloud/cli.py: launch/status/accounts/resume và resolved manifest.

Thuật toán/replay:
- algorithms/masac.py, algorithms/matd3.py.
- algorithms/models.py, action_utils.py, optimization.py, precision.py.
- algorithms/replay.py: TensorReplayBuffer.
- algorithms/hybrid_replay.py: HybridMeasuredReplay và validation state.
- algorithms/parallel_critics.py, critic_chunks.py.
- algorithms/checkpoints.py.

Training:
- experiments/training.py: BASE loop, counters/checkpoint/failure/lifecycle.
- experiments/pql_training.py: PQL loop.
- experiments/pql_system.py, pql_workers.py, pql_replay.py: collector/critic/policy, queues, replay/counters.
- experiments/environment_checkpoints.py: environment/RNG snapshot và restore.
- experiments/fixed_checkpoints.py: BASE full state, replay/RNG restore validation.
- experiments/pql_checkpoints.py: PQL full state.
- experiments/local_checkpoints.py: local retention/disk metrics.
- experiments/recovery_checkpoints.py: PeriodicRecovery.
- experiments/checkpoint_store.py: HF/archive/manifest transfer.
- experiments/lifecycle.py: observed scheduler termination metadata.

World/network:
- environment/env.py, vector.py, world.py, observations.py, scenario_recovery.py.
- environment/motion.py, avoidance.py, sensing.py, energy.py, geometry.py.
- environment/gpu/*: Warp GPU world, vector, state, motion, observations/geometry.
- communication/reports.py, routing.py, diagnostics.py, link_budget.py, energy.py.
- communication/gpu/*: GPU PHY/MAC/report/event/telemetry implementation.

Metrics/logs/evaluation:
- experiments/tracking.py: online W&B lifecycle, metrics, journal/evidence publication.
- experiments/evidence.py: bounded serialization, redaction, exception categories/chains, resources.
- experiments/run_journal.py: RunJournal, selected evidence, READING_GUIDE.
- experiments/diagnostics.py: warning/cohort diagnostics và failure_report.
- experiments/progress_watchdog.py: native thread-stack timer ownership/progress/cleanup.
- experiments/dashboard.py, dashboard_columns.py: Personal Workspace/plots/17 columns.
- experiments/evaluation.py: deterministic CPU evaluate/run_episode callbacks.
- experiments/async_evaluation.py, evaluation_snapshots.py: optional periodic checkpoint evaluator.
- experiments/comparison_metrics.py, comparison.py: mission metrics/protocol/comparison.
- experiments/rendering.py: image/animation.
- cloud/visualize.py, cloud/media_publish.py: CPU eval/progress/resume/media publication.

Cloud/orchestration:
- .github/workflows/uav_external_supervisor.yml.
- cloud/github_supervisor.py: short-lived external reconciler.
- cloud/continuation.py, controller.py, pipeline_status.py: lineage transitions/receipts.
- cloud/remote.py: Kaggle-side learner subprocesses, environment, finalize/exit signal.
- cloud/client.py, credentials.py, kernel.py, identities.py.
- cloud/archive.py, acceleration.py, execution_check.py.

Đọc module hiện có trước khi tạo lớp/service/monitoring mới. Ưu tiên stdlib và helper đang dùng.

# 7. Full flow GitHub Actions đã làm
- Workflow UAV External Supervisor: cron */15 * * * *, workflow_dispatch có dry_run.
- Đây là các invocation ngắn để reconcile; không giữ Actions runner chạy 10 giờ cùng learner.
- Supervisor dùng dependency orchestration; import cloud.github_supervisor không được kéo torch vào.
- Managed jobs có orchestration=github_actions, external_supervisor_required=true và reserved kernel markers gh-tra-g... / gh-val-g....
- Job cũ thiếu markers/receipt không tự được adopt hồi tố.
- Khi segment train còn thiếu budget, continuation dùng checkpoint phù hợp; thiếu quota hợp lệ thì chờ trên owner/account đúng lineage.
- Khi đủ episodes và completion metadata hợp lệ, chuyển sang Kaggle CPU eval 30 episodes, seed 2000000044, visualize và publish W&B/HF.
- Crash recovery chỉ đi từ full state/provenance đã xác minh; thiếu bằng chứng thì blocked.
- CPU interruption có progress/resume và deterministic retry refs để giảm submit trùng.
- cancelled không tự được hồi sinh.
- Tối đa 3 crash recovery liên tiếp; chỉ segment clean reset counter. Không biến waiting_for_quota thành submitted.
- Lỗi một account không chặn kiểm tra account sau. Các action chưa giải quyết trả nonzero thích hợp, tránh workflow xanh giả.
- supervisor-report.json được upload vào artifact uav-supervisor-report, retention 14 ngày. Probe/setup fail trước reconcile có thể chỉ có console; không đoán là JSON luôn tồn tại.
- W&B/HF --check-services được dùng trong manual dry-run; authentication/read access thành công chưa chứng minh write access hoặc full cloud cycle.

Trước khi nói có thể tắt sandbox:
1. Kaggle root thật sự running và learner W&B có progress thật.
2. Receipt/manifest đúng GitHub-managed markers.
3. Workflow active, account allowlist/identity/credential dataset/quota đã kiểm tra.
GitHub schedule có thể trễ; cron source không bảo đảm mỗi đúng 15 phút sẽ có execution.

Quan trọng: đã có hosted manual dry-run thành công và các phần cơ chế đã kiểm thử. Chưa được coi là đã chạy thực tế toàn bộ production train -> crash -> recover -> full 20k -> visualize trên source mới chỉ vì các test/smoke đều xanh. Lần audit logs gần nhất không launch lại đợt production.

# 8. Credentials và nhiều tài khoản Kaggle
- Credential file của repo: key.txt, trước đây là symlink tới:
    /home/aduy/Documents/NCKH/uav_network/key.txt
- File bị ignore; GitHub không chứa secrets này. Đọc schema/key names cần thiết, không in giá trị, không cat toàn bộ, không commit/copy secrets vào prompt/log/issue.
- Tôi đã cập nhật W&B key mới; đã xác thực thành công ở lần làm trước. Cần kiểm tra lại tính hợp lệ khi thật sự truy cập, không cần hỏi lại chỉ để dùng key cho công việc đã giao.
- 27/10 có 10 local enabled accounts được xác thực/quota read. Đây là snapshot lịch sử, không bảo đảm GPU allocation, phone verification, idle kernels hay dataset access hiện tại.
- GitHub allowlist/secrets lúc quan sát chỉ account_01/account_02. Thêm account vào key.txt LOCAL không tự đồng bộ GitHub.
- Workflow source có slots KAGGLE_API_TOKEN_ACCOUNT_01..24; kiểm tra secret names/allowlist/identity thật trước khi dùng thêm account. Slot có trong YAML không chứng minh đã có token hay account hoạt động.
- Biến liên quan: KAGGLE_ENABLED_ACCOUNTS, WANDB_ENTITY, WANDB_PROJECT, HF_REPO_ID, WANDB_API_KEY, HF_TOKEN; tên biến là cấu hình, giá trị secret phải giữ kín.
- Cùng một account_NN phải được đối chiếu đúng owner/identity giữa local và Actions; đừng chỉ dựa số thứ tự sau khi sửa key.txt.
- Lỗi 401/403 không tự chứng minh account bị ban. Kiểm tra xác thực/quyền hợp lệ; bảo toàn owner/lineage và lịch sử job.
- Kiểm tra quota/accelerator/jobs đang chạy trước một launch GPU mới. Dùng Kaggle khi cần CUDA, không giả lập kết quả CUDA trên CPU.
- Không tự cập nhật/revoke credentials hay mở rộng quyền khi nhiệm vụ chỉ là audit/prompt.

# 9. Recovery/checkpoint đã sửa và giới hạn thật
- G3 MASAC cũ chỉ còn policy checkpoint; không có replay/optimizer/RNG nên không exact-resume được tiến độ đã mất.
- Full-state serializers mới giữ replay, optimizer/scaler/RNG, progress/update clocks và environment state khi hỗ trợ; PQL cần drain workers và replay checkpoint enabled.
- Validate identity/algorithm/config/provenance/counters trước mutation. CPU/CUDA RNG thay device rồi reseed không được gọi là exact continuation.
- Canonical periodic recovery:
    resumable_checkpoint_interval_seconds=3600
    resumable_checkpoint_max_uploads=4
- Effective interval=max(requested interval, max_train_seconds/max_uploads).
  Với 36000/4 là 9000s, khoảng 2,5 giờ, cộng thời gian checkpoint/upload; KHÔNG phải bảo đảm backup thành công mỗi giờ.
- Cap tính số lần THỬ mỗi session, kể cả thất bại; final save là đường khác.
- Chỉ advance recovery/checkpoint_artifact và resumable alias sau upload confirmation. Backup fail giữ pointer tốt trước đó.
- Mỗi attempt dùng một thư mục staging do code sở hữu, dọn sau success/failure; không tích nhiều snapshot lớn trên đĩa Kaggle.
- Full snapshot 1M có thể khoảng 15,5–15,9GB. Local staging bounded không làm remote HF storage thành hằng số; immutable snapshots vẫn cộng dồn và truyền tốn thời gian.
- recovery/seconds_since_confirmed_upload là thời gian từ CONFIRMED UPLOAD, không phải tuổi state lúc chụp, không phải cam kết recoverability tuyệt đối.
- Đã có CPU tests, real two-T4 bounded recovery validation và local large snapshot cleanup evidence ở task cũ; task audit 28/10 không đo lại CUDA hoặc upload trọn 15GB.
- Nếu cần chứng minh khôi phục trên cloud, phải tải/validate checkpoint thật và chạy continuation tương ứng; không suy từ tên artifact “latest”.

# 10. Sự cố MASAC lịch sử và các run cần nhận biết
G3 production source:
  77a1d0f52ed4e78bc2823c3ada03d661568cd923
Các job đó AMP OFF, compile OFF. Policy acceleration sau này không được đưa vào source đã ghim.

MASAC:
- BASE HYBRID 63202ad9: crashed, 8192 episodes.
- BASE FULLGPU 25eecfd1: crashed, 8192.
- PQL HYBRID 6f0bcde3: crashed, 7045.
- PQL FULLGPU 3b39b0c8: crashed, 8196.

Bốn learner và bốn controller mất heartbeat trong 11s, khoảng 02:31:57–02:32:08 ngày 27/10 giờ Việt Nam. Đây là dấu hiệu mạnh của sự cố dùng chung platform/session/account/network/telemetry, chưa chứng minh root cause chính xác. Metrics cuối hữu hạn, thiếu traceback; sampled oomKills=0 không loại trừ đỉnh GPU/host memory cuối. Thiếu credential owner hợp lệ/không có đúng owner trong key file hiện tại đã cản lấy log private Kaggle. Không khẳng định ban hoặc OOM khi chưa có bằng chứng.

MATD3 G3 đã hoàn tất 20000 episodes:
- BASE HYBRID ce8f3be6 -> CPU eval 15db2a32.
- BASE FULLGPU 93fdf127 -> CPU eval 9c62d7c4.
- PQL HYBRID 3fcf25c7 -> CPU eval 239f3eb9.
- PQL FULLGPU 76b73021 -> CPU eval ba98364f.
Cả bốn CPU eval hoàn tất 30 tập và có publication theo báo cáo lịch sử.

ĐỐI CHIẾU LINEAGE MỚI ngày 28/10, 08:20 giờ Việt Nam:
- BASE HYBRID, eval 15db2a32, checkpoint ...uav-network-matd3-ce8f3be6:v11:
  mean delivery=45,8095238%; success=0/30.
- BASE FULLGPU, eval 9c62d7c4, checkpoint ...uav-network-matd3-93fdf127:v11:
  mean delivery=48,0952381%; success=0/30.
- PQL HYBRID, eval 239f3eb9, checkpoint ...uav-network-matd3-3fcf25c7:v11:
  mean delivery=48,9523810%; success=0/30.
- PQL FULLGPU, eval ba98364f, checkpoint ...uav-network-matd3-76b73021:v9:
  mean delivery=46,0952381%; success=0/30.

Đính chính quan trọng: câu trả lời cuối của chat trước đã ghép nhầm 48,95% vào BASE HYBRID. Con số đó thuộc PQL HYBRID. Luôn join bằng checkpoint artifact/provenance, không ghép training và CPU run chỉ vì cùng thuật toán hay cùng xuất hiện gần nhau.

Các số này cho thấy nhiệm vụ chưa thành công toàn bộ trên seed bank đã thử; không đủ tự nó kết luận nguyên nhân hay xếp hạng thuật toán cho mọi seed/scenario.

# 11. Acceleration policy có bằng chứng riêng
Chỉ canonical config, cloud train/validate; explicit user overrides thắng:
- MASAC BASE: FP16 + compile ON + fused OFF.
- MATD3 BASE: FP16 + compile ON + fused OFF.
- MASAC PQL: FP16 + compile ON + fused OFF.
- MATD3 PQL: FP16 + compile OFF + fused OFF.

Helper cloud/acceleration.py materialize overrides và manifest mang theo chúng sang continuation. Không áp policy này toàn cục cho mọi config/benchmark/phần cứng.
Canonical PQL --algorithm both bị từ chối vì hai thuật toán có compile policy khác nhau; launch riêng từng thuật toán. BASE both có thể có policy tương thích, nhưng full-flow production guide dùng một thuật toán/root; đọc gate hiện tại trước khi launch.

MATD3 PQL compile chậm hơn khoảng 11,51% TPS ở phép đo matched T4/G3 đã lưu, không phải định luật mọi workload.
Real two-T4 PQL smoke ở source906ee68 đã kiểm tra policy đúng cho MASAC/MATD3 và execution-check có learner updates thật. Smoke trainer budget ngắn có 0 learner updates, nên không dùng 1-episode smoke làm bằng chứng học tốt hay thay chứng cứ execution-check.

# 12. Observability mới đã hoàn tất ở source56604cc
Các sửa đổi:
- evidence.py dùng stdlib, JSON an toàn/bounded; NaN/Inf thành chuỗi, unknown object thành type marker, không repr/dump tensor/replay tùy tiện.
- Redact configured credential values và env secret/token/password/API-key/access-key/private-key/legacy KAGGLE_KEY families. Không hứa có thể tự nhận diện mọi chuỗi secret không có ngữ cảnh.
- failure.json schema2: exception chain/category, redacted traceback, context, actual run/stage identity, resource snapshot. Category không tự xác nhận root cause.
- Optional console/file/journal/status/upload tách lỗi với nhau, không thay exception chính. Các write/checkpoint/online tracking bắt buộc vẫn giữ hợp đồng hiện tại, không âm thầm train offline.
- BASE/PQL JSON console và failure fallback đã có guard khi stdout đóng.
- Journal chọn warning/replay/PQL/reward/action/safety/reports/eval metrics có sẵn; giữ source-step markers để biết bằng chứng cũ/mới.
- Resource snapshot ở nhịp event tối đa mỗi 60s và fresh lúc failure, có captured time/age: disk free, process RSS/peak RSS, host MemAvailable, CUDA allocated/reserved/peak khi có.
- Đây không phải sampler ngoài tiến trình; nếu cả process/máy dừng, không thể tiếp tục lấy resource.
- ProgressWatchdog native faulthandler, một timer owner/process, giữ FD tới cancel; BASE/PQL kể cả offline và CPU đều tích hợp.
- Default 300s, rearm throttle làm effective inactivity khoảng 270–300s. Stack chỉ ghi vị trí thread/long call, không tự kill hoặc kết luận deadlock. Đây là minor timing được ghi rõ, không phải crash detector tuyệt đối.
- Learner subprocess PYTHONFAULTHANDLER=1; stderr có thể chứa fatal trace. SIGTERM metadata chỉ ghi nếu thực nhận; parent negative return code ghi tên signal. SIGKILL không cho learner tự ghi failure/cleanup.
- CPU on_progress giữ watchdog tiến triển; publication optional thất bại không làm evaluation thật thất bại.
- CPU aggregate có n, seed interval, scenario hash, deterministic/overlap nếu protocol cung cấp, checkpoint step/artifact. Count mô tả kết quả hiện hiển thị, không cộng nhiều rounds/checkpoints.
- PQL failure context giữ last metrics/seeds/counters đã có, tránh lấy thêm worker lock lúc đang lỗi.
- Async evaluation evidence upload/status failures được cô lập và ghi symptom.

Files người xem:
- W&B Logs: stdout/stderr + training_event JSON.
- training-events.jsonl: event history + identity/context.
- training-status.json: latest atomic status.
- metrics-guide.md: guide cùng run.
- diagnostics.jsonl: warning/error evidence.
- cohorts.jsonl: closed outcome windows/sample source.
- failure.json: exception/resource/identity context.
- thread-stacks.log: no-progress/native stacks.
- failed_batch.pt / failure_state.pt khi learner hỗ trợ: phục vụ tái hiện, không tự là full recovery checkpoint.
- resolved_config.yaml, wandb_run.json và CPU metrics.json: config/identity/protocol/records/publication.

Mọi file tùy chọn là best-effort. Mạng/đĩa/console cùng mất thì không còn kênh ghi. Log mới chỉ xuất hiện ở job thực sự dùng source mới; không hồi tố vào run cũ.

# 13. W&B online / Personal Workspace chính xác
Entity/project:
  truongdaoanhduy-uit/uav_search_target

Personal Workspace:
https://forge.coreweave.com/wandb/truongdaoanhduy-uit/uav_search_target/workspace?nw=nwusertruongdaoanhduy

Personal Runs:
https://forge.coreweave.com/wandb/truongdaoanhduy-uit/uav_search_target/table?nw=nwusertruongdaoanhduy

Personal view name:
  nw-nwusertruongdaoanhduy-w
Personal view ID:
  VmlldzoxODA5MjI1Nw==

Giữ đúng view này; không bắt tôi chuyển sang saved workspace khác để thấy kết quả. Saved view khác là dữ liệu độc lập cần bảo toàn.

17 cột đang hiển thị:
1. run:displayName
2. run:state
3. run:createdAt
4. run:duration
5. config:stage
6. summary:evaluation/episodes_completed
7. config:kaggle_account
8. config:runtime.seed
9. summary:global_step
10. summary:train/episodes_completed
11. summary:cohort/delivery_rate_percent_mean
12. summary:cohort/success_mean
13. summary:eval/mean_delivery_rate_percent
14. summary:eval/success_rate
15. summary:episode_budget_complete
16. summary:stop_reason
17. summary:diagnostics/failure_type

Ghim 4 cột đầu. Algorithm và Mode ẩn. Created là lúc W&B tạo run; Runtime của phiên/run, không phải tổng thời gian lineage. All states retained, Created mới trước, không cố ý giấu crashed/failed.

27 core plots + 1 guide:
- Mission quality / completed outcomes: 5 plots, gồm delivery, success, return, search, cohort n.
- Training progress/speed: 3.
- Periodic fixed-policy evaluation: 2, có thể trống vì canonical schedule tắt.
- Final CPU evaluation từng episode: 2.
- Why delivery low: 4.
- Numerical diagnostics: 4.
- Resources: 5.
- Recovery/upload: 2.
Actor loss vẫn ở metric history và profile CLI “diagnostics”; đã chuyển khỏi core để giảm trùng lặp, không xóa dữ liệu.

Mỗi plot tối đa 12 run riêng, aggregate=false; project maxRuns=1000. Đừng tự gộp các thuật toán để tạo một đường đẹp hơn.
Trục:
- system/*: _runtime.
- disk/cohort/training: global_step.
- periodic eval: evaluation/checkpoint_global_step.
- CPU episode metrics: evaluation/episodes_completed.

Native system keys đã đối chiếu:
system/gpu.0.gpu, system/gpu.1.gpu,
system/gpu.0.memoryAllocated, system/gpu.1.memoryAllocated,
system/cpu, system/memory_percent.
CUDA device index là logical index theo CUDA_VISIBLE_DEVICES, đừng tự gán physical GPU.

# 14. Cách đọc chất lượng training
- Ưu tiên delivery tới GCS, mission success, n và protocol/budget/seed tương ứng; return chỉ bổ sung.
- Kiểm tra cohort/episodes, cohort/partial, source_start/end_step; live-world metrics phụ thuộc tuổi episode.
- Success=0 nghĩa chưa có full mission success trong mẫu; không được diễn giải thành chưa gửi được bất kỳ report nào.
- Search cao/delivery thấp: đối chiếu eligibility/silent action, packet PHY/retry/CCA, buffers/TTL/report loss, relay route, motion/shield và energy.
- Loss/TD/Q/grad/entropy/alpha hỗ trợ chẩn đoán, không có ngưỡng loss chung để chấm tốt/xấu.
- TPS/GPU utilization/RAM nói về hiệu năng/tài nguyên, không chứng minh chất lượng policy.
- Chỉ so energy/delivery khi delivery/success đủ tương đương.
- BASE sample-use theo segment có thể khác counter PQL restored; phải đọc nguồn counter.
- Finished cần đối chiếu episode_budget_complete/stop_reason; train xong không đồng nghĩa eval/video publish xong.
- Thiếu dữ liệu giữ blank/null; không điền 0 cho đẹp. Overlap null là chưa biết, không phải false.
- Cảnh báo là heuristic/symptom. Muốn kết luận cause cần reproduction/batch/checkpoint/traceback hoặc log nền tảng.
- Phân biệt kết quả thấp, suy giảm, không ổn định, tiến độ chậm, không có tiến độ, hết một segment, thiếu bằng chứng, lỗi telemetry và lỗi eval/publication.
- Final CPU30 trên một training seed chưa đủ kết luận generalization/algorithm ranking. Canonical mini-eval đang tắt, đừng bật hoặc đổi seed/reward chỉ để lấp plot trống nếu nhiệm vụ chưa yêu cầu.

# 15. Bằng chứng test và trạng thái đã quan sát
Full suite trên đúng runtime56604cc, source frozen:
  924 passed, 95 skipped, 8 warnings,
  1396 subtests passed, 554.26s (~9m14s).
JUnit: 2415 cases gồm subtests/skips, 0 failures, 0 errors.
Task logs gần nhất không chạy GPU validation mới; đừng gọi CPU skips là CUDA pass.
77 focused tests sau fix review đã pass; các focused/full sets chồng lặp, không cộng thành tổng test duy nhất.
compileall/diff check/no-torch supervisor import pass.
Scan 14 credential/environment values: 0 hits trong tracked source/docs lúc hoàn tất.

Đã có independent review; 2 Important đã tái hiện RED và sửa GREEN:
1. CPU optional progress logging làm evaluation abort.
2. Thiếu redaction KAGGLE_KEY và env credential families.
Minor timer precision 30s đã nêu, không tự kill.

Online probe logging-only f0vws7re ở56604cc:
- Nhãn CHECK | diagnostic coverage | no training | 20261028.
- Finished; không có rollout/GPU train hoặc policy quality metrics giả.
- Live status và atomic replacement đã tải lại khi đang running.
- Sau finish tải lại đủ 4 files: journal/status/guide/thread-stacks.
- Xác nhận resource thực đo, source đúng và native Timeout stack.
- Probe dùng timer3s để kiểm chứng; production vẫn default300s.
- File stack xác nhận sau finish không chứng minh upload live stack có latency cam kết.

Snapshot service cuối audit lúc 08:06:53 ngày28/10 giờ Việt Nam:
  385 runs = 273 finished + 82 failed + 30 crashed, 0 running.
384 IDs/states/historyLineCounts cũ giữ nguyên; chỉ thêm canary f0vws7re.
Đây là snapshot, phải query lại trước hành động operational. Không suy rằng có production run đang chạy vì câu chuyện ở chat cũ nói “đã launch”.

# 16. Test/inspection commands để dùng đúng chỗ
Các lệnh sau là kiểm tra, không launch GPU production:

  git status --short
  git branch --show-current
  git log -8 --oneline
  git ls-remote --heads origin main feat/cpu-replay-20261022

  python -m uav_network config --config configs/comparison-20k-hybrid75-1kb.yaml
  python -m uav_network --help
  python -m uav_network kaggle --help

  python -m compileall -q uav_network
  git diff --check

Full offline CPU suite, chỉ khi thay đổi cần kiểm tra toàn bộ:
  HF_HUB_OFFLINE=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m pytest tests -q --tb=short

No-torch supervisor boundary:
  python -c "import sys; import uav_network.cloud.github_supervisor; assert 'torch' not in sys.modules"

Chọn tests theo thay đổi:
- Evidence/logs: test_evidence_coverage.py, test_run_journal.py, test_diagnostics.py, test_progress_watchdog.py.
- Dashboard: test_dashboard.py, test_diagnostics_axes.py.
- Evaluation: test_async_evaluation.py, test_evaluation_resume.py, test_visualization_tracking.py, test_visualization_provenance.py, test_visualization_resume.py, test_visualization_publish.py, test_cpu_handoff*.py.
- Recovery: test_baseline_resume.py, test_exact_environment_resume.py, test_recovery_checkpoints.py, test_retry_recovery.py, test_github_supervisor.py, test_github_supervisor_workflow.py.
- Replay/PQL: test_cpu_replay.py, test_measured_replay.py, test_pql_replay_serialization.py, test_production_hybrid_replay.py và các tests PQL liên quan.
- CUDA: test_cuda_exact_resume.py và tests yêu cầu GPU; chỉ báo pass nếu thật sự chạy trong môi trường phù hợp.
- Acceleration: test_acceleration_policy.py và execution-check evidence liên quan.

Ở lệnh pytest, các test file ở thư mục tests/. Dùng rg --files/rg -n để tìm đúng tên/version hiện có trước khi chạy.

Hosted dry-run khi nhiệm vụ yêu cầu kiểm tra GitHub deployment:
  gh workflow run uav_external_supervisor.yml --ref main -f dry_run=true
  gh run list --workflow uav_external_supervisor.yml --limit 3
Chỉ CLI dry-run/check-services read access không thay cho full production end-to-end.

Lệnh launch production cụ thể: đọc docs/github-actions-full-flow.md và --help mới, chọn account/credential dataset/quota/algorithm/mode/source rõ ràng; không chạy ví dụ chỉ vì nó xuất hiện trong tài liệu bàn giao.

# 17. Công cụ, browser và skills nên dùng
Dùng các plugin có sẵn theo năng lực:
- Sandbox: thao tác repo/environment thật.
- GitHub: source, commits, Actions, logs/artifacts.
- Context7: tài liệu chính thức đúng phiên bản khi cần xác minh API/SDK/Python/Torch/W&B; không suy từ trí nhớ.
- Superpowers: quy trình debug/thiết kế/implementation/verification thích hợp.
- Ponytail: reuse helper/std lib, thay đổi tối thiểu đủ giải quyết vấn đề; tránh thêm service/framework không cần.

Suggested skills:
- superpowers:using-superpowers khi bắt đầu.
- superpowers:systematic-debugging khi gặp lỗi.
- superpowers:brainstorming + writing-plans khi thực sự thêm/đổi hành vi đáng kể; tận dụng phạm vi đã rõ, tránh lặp phỏng vấn vô ích.
- superpowers:test-driven-development cho bug/feature logic.
- superpowers:verification-before-completion trước báo “xong/pass/fixed”.
- requesting-code-review cho thay đổi lớn; nếu dùng executing-plans thì một review độc lập cả nhánh, sửa Critical/Important bằng RED->GREEN.
- engineering-suite-ponytail:entry-ponytail / ponytail.
- handoff / writing-for-agents khi cập nhật hồ sơ chuyển chat.

Browser:
- Tôi đã cho phép đăng nhập W&B qua id.coreweave.com để kiểm tra giao diện. Reuse phiên hợp lệ nếu có; tuân theo browserAuth của công cụ nếu cần xác thực tiếp.
- Quyền đã có không có nghĩa cookie/tab/biến REPL từ chat cũ vẫn tồn tại. Discover tab mới, không dùng cứng tab ID/handle cũ.
- Ưu tiên API/SDK cho kiểm tra dữ liệu và cập nhật workspace đã được giao; browser để xác minh Personal UI/plot/table thật.
- Đừng dùng web search công khai để thay cho dữ liệu private authenticated của project.
- Trước sửa view, backup spec + lấy IDs/state/history fingerprints; sau sửa readback và xác nhận UI. Bảo toàn saved/unrelated views.
- Không sửa trạng thái/summary/history run cũ để làm biểu đồ hoặc báo cáo đẹp hơn.
- Không lưu screenshot cá nhân công khai hoặc vào thư viện nếu tôi chưa yêu cầu.

# 18. Phạm vi còn chưa được chứng minh và điều không nên suy diễn
- Chưa có chứng minh mọi nguyên nhân crash từ logs; hard kill/power loss/platform/account vẫn có khoảng trống.
- Checkpoint policy-only cũ không được “nâng cấp” hồi tố thành full state.
- Không bảo đảm Kaggle không ngắt, quota luôn có, GitHub cron đúng phút, mọi upload thành công.
- Chưa chứng minh một full cloud production cycle mới train -> crash -> recover -> 20k -> CPU visualize trong task audit logs.
- Không có production20k rerun ở cuối task28/10. Nếu nhiệm vụ mới là chạy lại thì phải thực hiện preflight, pin source và launch riêng theo yêu cầu đó.
- CUDA tests/benchmark cũ chỉ chứng minh đúng source/workload của chúng; không tự thành kết quả mới cho source56604cc hoặc phần cứng khác.
- Large full-checkpoint network transfer/storage cost phải đo riêng nếu mục tiêu là chứng minh nó; local cleanup/serialization test không đủ.
- Snapshot counts/quota/credential validity phải refresh trước khi quyết định chạy cloud.
- UI/logging “đủ dùng” không đồng nghĩa policy đã tốt; dùng protocol và nhiều seed khi cần đánh giá khoa học.

# 19. Cách làm việc và tiêu chí hoàn tất cho nhiệm vụ mới
- Tóm tắt ngắn những gì đã xác minh từ repo/live state, rồi thực hiện nhiệm vụ dưới đây.
- Tách observed fact, hypothesis, reproduced bug và unverified limit.
- Khi debug: đọc code/log, xác định giả thuyết, tái hiện, thêm test có ý nghĩa, sửa nguyên nhân, chạy kiểm chứng thích hợp.
- Khi chỉnh metrics/UI: mỗi cột/plot phải có câu hỏi nó trả lời; giữ Personal, meanings/axes/source/n rõ ràng.
- Khi launch: kiểm tra công việc đang chạy/quota/owner/source/receipt; bảo toàn job và dữ liệu hiện có. Chỉ tuyên bố running khi live API và progress chứng minh.
- Khi sửa code: bảo toàn thay đổi của người dùng, dùng worktree/branch thích hợp, không reset/clean/force push tùy tiện.
- Với tác vụ code đã được giao, hoàn tất implementation, review/check cần thiết, cập nhật CURRENT_TASK/evidence và commit/push theo phạm vi đã có. Kiểm tra remote không divergent trước push; đừng coi quyền cũ là quyền phá hủy công việc mới.
- Không cần chạy lại full9phút suite chỉ để đọc/tạo prompt hay sửa tài liệu không đổi hành vi; nhưng thay đổi runtime phải có kiểm thử tương xứng và evidence mới.
- Lệnh dài chạy background, log ra file rồi đọc tail/result; tiếp tục gửi cập nhật ngắn có ý nghĩa, không im lặng quá lâu.
- Báo cáo cuối: đã thay đổi gì, vì sao, source/commit nào, kiểm chứng nào thật sự chạy, kết quả và giới hạn gì còn lại. Không chỉ nói “đã làm” từ tool success.

# 20. Nhiệm vụ tôi muốn bạn làm ở chat mới
[Tôi sẽ ghi yêu cầu tiếp theo ở đây, hoặc gửi ngay sau prompt này. Hãy giữ toàn bộ bối cảnh trên và thực hiện đúng yêu cầu mới; không tự coi tất cả mục lịch sử là việc phải làm lại.]
