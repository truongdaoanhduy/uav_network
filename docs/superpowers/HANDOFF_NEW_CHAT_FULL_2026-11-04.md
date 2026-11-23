# PROMPT BÀN GIAO CHAT MỚI — FULL REFERENCE + REPRO EVIDENCE, 04/11/2026

**CẬP NHẬT MỚI NHẤT:** GitHub reference remote HEAD `000e734cb7fa7141d246b43b363b403c3364862f` ngày 04/11/2026; commit code đổi cờ khoa học cuối `81902df57fad8bc3ce9cadac8f0f10b099b24a89`. Tài liệu ở đây được tạo sau test thực tế PQL seed44. Đừng mặc nhiên coi trạng thái remote này là bất biến; kiểm tra git/W&B/GitHub Actions khi làm việc.

# PROMPT BÀN GIAO CHAT MỚI — UAV NETWORK, 04/11/2026

Bạn là trợ lý kỹ thuật tiếp tục dự án nghiên cứu multi-agent reinforcement learning, Search and Rescue UAV, mạng ad-hoc UAV, Bayesian sensing, hybrid continuous/discrete action, MASAC/MATD3 và PQL. Người dùng yêu cầu thao tác THỰC TẾ trên repo, không chỉ đề xuất. Hãy đọc source, test, W&B và cloud khi được yêu cầu; phân biệt bằng chứng với suy luận; không chế biến đổi baseline để làm thuật toán tốt hơn/xấu hơn. Đọc kỹ toàn bộ prompt trước khi thực hiện task tiếp theo của người dùng.

## 1. Định danh chính xác / trạng thái mới nhất
- Repo local gốc: `/home/aduy/Documents/NCKH/uav_network`; GitHub `truongdaoanhduy/uav_network`.
- **Branch khoa học cần dùng:** `baseline/reference-masac-matd3-20261103`; GitHub remote đã xác nhận commit đổi cờ compile/fused `81902df57fad8bc3ce9cadac8f0f10b099b24a89`. Trong cùng repo có branch `main` cho GitHub Actions và các profile lịch sử; **không đồng nhất main với reference**.
- Isolated working tree đang dùng cho reference: `/home/aduy/Documents/NCKH/uav_network-reference-fused-20261104`; khi thực hiện hãy kiểm tra `git status`, `git rev-parse HEAD`, `git ls-remote`, tránh ghi đè worktree khác. Hiện tại remote reference HEAD sau kiểm thử seed44 là `000e734cb7fa7141d246b43b363b403c3364862f` (docs/evidence); commit thay đổi khoa học cuối vẫn là `81902df57fad8bc3ce9cadac8f0f10b099b24a89`. Kiểm tra HEAD mỗi lần vì có thể tiếp tục thay đổi.
- Cấu hình khoa học duy nhất: `configs/comparison-reference-masac-matd3-xyz-sensing.yaml`; provenance: `docs/methods/REFERENCE_BASELINE_PROVENANCE.md`; task log: `docs/superpowers/CURRENT_TASK.md`; tests: `tests/test_reference_baseline_contract.py`, `tests/test_reference_critic_gradients.py`.
- **Chốt cuối cùng:** `runtime.compile_models=false`, `runtime.fused_optimizers=false`, `runtime.amp_mode="off"` (FP32 eager), `runtime.critic_chunk_size=0` (không chunk khi update), `runtime.vectorize_critics=false`, `matd3.output_normalization=false`. Cả MASAC/MATD3 không share actor/critic. Đây là quyết định cuối, **không tự bật lại**; đã push và test 28 passed + 534 subtests passed.
- Không có training cloud/full-budget mới được khởi chạy; ở task sau đã chạy 8 smoke CPU cực nhỏ, xem mục 10. W&B `truongdaoanhduy-uit/uav_search_target` lần cuối đọc ngày 04/11/2026 có `0 running`, `0 pending`. GitHub Actions `UAV External Supervisor` đang `disabled_manually` sau yêu cầu dừng.

## 2. Mục tiêu khoa học và nguyên tắc
- Nhiệm vụ: 6 UAV tìm kiếm 20 mục tiêu trên bản đồ 1km, camera FOV có detection noise và che khuất; xác nhận mục tiêu ở độ cao thấp; tạo report qua mạng packet, relay/GCS, đảm bảo năng lượng/va chạm; tối ưu chính sách bằng MASAC, MATD3.
- So sánh MASAC BASE, MASAC PQL, MATD3 BASE, MATD3 PQL. BASE và PQL dùng cùng algorithm core, cùng task/hyperparams; PQL khác hệ thống thực thi actor/critic/replay bất đồng bộ.
- **Source-backed baseline, không phải byte-identical upstream SAC/TD3:** action gồm motion XYZ + RF power liên tục và destination categorical; có mask và centralized critic. Bắt buộc đọc `REFERENCE_BASELINE_PROVENANCE.md` để hiểu những UAV-interface adaptations hợp lệ.
- Thứ tự nguồn phương pháp: paper tác giả → mã nguồn tác giả/library uy tín → adapter nhiệm vụ được công bố minh bạch; không tự viết heuristic để "cứu" một thuật toán. Giữ công bằng và reproducibility; không so sánh tốc độ/hội tụ nếu kiến trúc, settings, seeds, episodes khác nhau mà không nêu rõ.
- Từng task: đọc source/log, phân tích nguyên nhân, báo cáo, fix, chạy test, ghi evidence/commit, báo cáo kết quả; không tự động làm thêm task hay tự chạy cloud nếu chưa được yêu cầu.

## 3. Cấu hình reference khoa học (resolved YAML)
- `world`: generation_version=3, num_uavs=6, map_size_m=1000, num_targets=20, num_obstacles=15, max_steps=1500, dt_s=1, max_speed_mps=20, max_accel_mps2=2; 3D, z=0..150m; obstacle z=30..120; `motion_safety=cbf`, safety_distance_m=15 (default class); scene sinh theo seed, có recovery/clearance checks.
- `sensor`: model=bayesian, altitude_init=100m, grid_cell=25m, belief_prior=0.5, FOV=90 độ, altitude anchors [0,50,100,150], Pd=[0.99,0.90,0.80,0.70], Pf=[0.01,0.10,0.20,0.30], fine_altitude=50, confirmation_threshold=0.99, belief_patch_cells=5, belief_coarse_cells=5, verified_empty_belief=0.01, obstacle_occlusion=true. Positive/negative camera readings update posterior; high altitude chỉ tạo belief, low altitude đủ điều kiện mới confirm/report. Không dùng ground truth để âm thầm sửa false-positive belief nếu source hiện không làm vậy.
- **Schema source thực tế đã kiểm tra:** observation_dim=198, centralized state_dim=565, observation schema version=5, continuous_dim=4 (`ax, ay, az, power`), discrete destinations=7 (`silent`, 5 peers, `GCS`). Các tài liệu cũ ghi 173/schema4 là thông tin lịch sử không còn đúng; phải chạy `observation_dim`, `state_dim`, `observation_version` trên resolved YAML.
- `network`: UavNetSim native và Warp packet GPU; 500m peer/GCS, LOS/NLOS urban, 2Mbps, bandwidth22MHz, SINR thr6dB, propagation delay true, receiver admission true, report/payload1024B, buffer3072B, TTL300s, retries5, power learned logarithmic 0.001..0.4 W, PA efficiency0.35, TX/RX circuit 0.1W, power-limited admission. Chính sách chọn next-hop; có CSMA/CA, ACK, retry, custody, packet lifetime và radio energy. Không thay routing bằng oracle/greedy ngầm.
- `energy`: battery=277200J=77Wh/UAV; propulsion power=20+120+0.5||v||²+2||a||² W, dt=1; RF accounting riêng.
- `reward`: discovery+10, delivery+50, information_gain=10, investigation_shaping=10, shaping_gamma=.99, investigation_threshold=.8, communication_efficiency=1, shield_correction=.1, shield_intervention=.5, unsafe_action=1; các loss như blocked/expired/dropped/step/energy còn defaults trong `configuration.py` (blocked0.2, expired10, dropped10, step0.01, RF energy0.01, propulsion energy/kJ0.01). Không tự đổi reward nếu user đang yêu cầu baseline code.
- `runtime`: episodes=20000, eval_episodes=30, seed44, eval_seed=2000000044, num_envs=4096, batch8192, replay_capacity1M, replay CPU75%+GPU25%, updates_per_step8, learning_starts131072, two Kaggle T4 mỗi run trong lịch sử; warp_packet, FP32, no AMP, no compile, no fused, no vectorization, critic chunk0, checkpoint interval128, logging steps4096, max_train_seconds36000, deterministic=true, torch_threads1.
- `tracking`: W&B entity/project đã nêu, group=`uav-reference-baseline-xyz-20t15o-1500`, HF repo=`duy95/uav-search-checkpoints`, checkpoint_store=huggingface, periodic validation interval=3686400 transitions, periodic eval5 episodes, seeds được tách khỏi final test, artifact checkpoint/recovery/CPU visualization.
- `pql`: default disabled, khi PQL dùng `pql.enabled=true`, mode=async, critic_sample_ratio8, warmup_collection_steps32, max_pending_collections2, max_pending_policy_updates2, worker_timeout600, checkpoint_replay=true.

## 4. Thuật toán: những gì code THỰC SỰ làm
- `uav_network/algorithms/models.py`: ReLU MLP 256-256; HybridActor MASAC có `mean`, `log_std`, `destination` categorical; tanh Gaussian + log Jacobian stable, `LOG_STD_MIN=-20`, `LOG_STD_MAX=2`; DeterministicActor MATD3 tanh continuous và masked categorical; TwinCritic Q1/Q2 nhận centralized state + flattened joint actions.
- **Independent actors/critics:** 6 actor riêng; 6 twin-critic modules = 12 Q networks; mỗi agent có actor optimizer riêng, critic parameters được gom dưới một critic Adam (nhưng Q objective phải giữ normalization per agent). CTDE nghĩa là actor input local, critic input global; không đồng nghĩa share weights.
- **MASAC** `algorithms/masac.py`: off-policy SAC with hybrid action; stochastic continuous and masked categorical, twin target minQ, exact sum over valid discrete destinations; alpha continuous và discrete riêng từng agent, automatic temperature, init alpha=1, target continuous=-4 nominal, effective conditional entropy depends on active & silent RF controls; discrete fixed target .2*log(7), no annealing, mask not change target cardinality; no alpha hard clamp, no gradient clip. Gamma=.99, tau=.005, lr=.0003. Code đồng thời có các adapter action gating và entropy masking.
- **MATD3** `algorithms/matd3.py`: deterministic tanh, twin centralized critic per agent, policy_delay=2, target_noise_std=.2, clip=.5, exploration=.1, soft Gumbel-Softmax discrete in actor/target, action env/replay hard one-hot, RF power gated by hard chosen destination, peer action from replay unchanged when updating current actor; per-parameter grad_clip_norm=.5, no output normalization. Không tự bật epsilon schedule/hard straight-through/peer policy substitution.
- Shared actor/critic vẫn có thể tồn tại trong historical configs và code, nhưng reference KHÔNG share. Không thay false thành true chỉ vì tốc độ cao hơn. Checkpoint migration/version and source origin must be inspected.
- `algorithms/precision.py`: AMP only supports off/fp16, FP16 GradScaler clocks per group, overflow diagnostics. Reference off. `algorithms/optimization.py`: gradient finite safety checks, scalar_metrics CPU/GPU telemetry fix, compile_models helper tồn tại nhưng không được gọi trong reference profile. `parallel_critics.py`: eager unvectorized path uses Python loop over critics; optional vmap exists but reference false. `critic_chunks.py`: available but chunk0 means no slicing.
- `algorithms/replay.py`, `hybrid_replay.py`: tensor replay 1M; 75% CPU/25% GPU. `algorithms/checkpoints.py`: model and observation schema compatibility.

## 5. Đường đi dữ liệu environment / communication
1. `configuration.py` + `configuration_xyz.py` validate YAML and CLI overrides.
2. `environment/world.py`, `scenario_recovery.py`, GPU `world.py/state.py`: seed scenario, obstacles, mission state, initialization and reset.
3. `environment/xyz_motion.py`, `cbf.py`, `avoidance.py`, `energy.py`; GPU `xyz_motion.py`, `motion.py`: update physical movement/constraints, collision safety, energy. CBF acts as a projection/minimal intervention; task search stays learned.
4. `environment/bayesian.py`, `sensing.py`, `coverage.py`, `xyz_geometry.py`; GPU `bayesian.py`: FOV, LOS, Pd/Pf, posterior, cells, report confirmation.
5. `communication/reports.py`, `routing.py`, `link_budget.py`, `backend.py`, `upstream.py`, `energy.py`; GPU `communication/gpu/{phy,mac,reports,events,streaming,power}.py`: report creation, custody/buffers/TTL, power/radio, selected relay vs GCS, retransmit and ACK, actual/accelerated packet simulation.
6. `environment/observations.py`, `xyz_observations.py` and GPU equivalents: local masked actor observation and centralized critic state; protect decentralized information boundary.
7. `environment/env.py`, `vector.py`, GPU `vector.py`: Gym/parallel/Warp vector environment.
8. `algorithms/{models,masac,matd3,action_utils,...}.py` → `experiments/{training,pql_system,pql_workers,pql_training,pql_replay}.py`: actions, replay, learner updates.
9. `experiments/{evaluation,comparison,rendering,tracking,telemetry,diagnostics,best_checkpoints,...}.py`: metrics, CPU evaluation, side-by-side comparisons and visualization.
10. `cloud/{cli,client,credentials,kernel,remote,controller,continuation,github_supervisor,visualize,...}.py`: Kaggle submit, pinned commits, quota, checkpoint continuation, GitHub reconciliation, CPU eval, W&B/HF publication.

## 6. Các file cần đọc khi chỉnh task
- Scientific contracts / equations: `docs/methods/REFERENCE_BASELINE_PROVENANCE.md`, `tests/test_reference_baseline_contract.py`, `tests/test_reference_critic_gradients.py`, `tests/test_algorithm_math_regressions.py`, `tests/test_xyz_algorithms.py`.
- MASAC/MATD3: `algorithms/{masac,matd3,models,optimization,precision,parallel_critics,action_utils,replay,hybrid_replay}.py`.
- World/sensing: `environment/{env,world,xyz_motion,cbf,bayesian,observations,xyz_observations}.py` + GPU equivalents.
- Network/power: `communication/{backend,link_budget,reports,routing,energy}.py`, GPU `communication/gpu/`.
- Train/eval: `experiments/{training,pql_system,pql_workers,pql_training,pql_replay,evaluation,comparison,rendering,tracking,best_checkpoints,checkpoint_store,overflow_capture}.py`.
- Kaggle and GitHub: `cloud/{cli,client,credentials,execution_check,remote,continuation,controller,github_supervisor}.py`, `.github/workflows/uav_external_supervisor.yml`, `scripts/sync_github_supervisor_secrets.py`.
- CLI: `python -m uav_network config --config configs/comparison-reference-masac-matd3-xyz-sensing.yaml`; `python -m uav_network train --config ... --algorithm masac|matd3 [--set pql.enabled=true]`; see `uav_network/experiments/cli.py` and `uav_network/cloud/cli.py` for exact syntax. Do not issue cloud commands without explicit approval.

## 7. W&B scientific history and performance findings
- Current reference v2 03/11 run IDs (now stopped/failed; data retained): MASAC BASE `44145eed`, MASAC PQL `6441724a`, MATD3 BASE `a342b713`, MATD3 PQL `6c02fc92`. They used **old commit `4afff25`**, AMP off, compile off, fused off, independent actors/critics, critic chunk0, vectorize off. They did not receive later compile/fused experiments and were not trained with the final 04/11 commit.
- Earlier 02/11 MATD3 BASE `db61bd0c`, PQL `d51b0350`: shared actor+critic, vectorize_critics=true, critic_chunk_size1024, compile=true, AMP off, fused off, MATD3 output norm=true.
- Earlier 31/10 MATD3 BASE `6a64bc40`, PQL `4846934d`: shared actor+critic, vectorized and chunk1024, compile=false, AMP off, fused off, output norm=true.
- Measured W&B timings: new 03/11 MATD3 BASE 55,616 updates / 9,100.22 learner sec = `0.16363 s/update`, total runtime 10,270s; old 02/11 BASE 47,056 updates / 4,111.20 learner sec = `0.08737 s/update`; old 31/10 BASE 59,752 updates / 5,429.24 learner sec = `0.09086 s/update`. New reference learner slower ~1.8–1.9x *per update*; actor/critic independent vs shared plus vectorization and different compile/chunk settings are confounders. **AMP/Fused were off in ALL these compared runs, so they cannot explain the observed difference.** Don't assert causal speedup from compile without matched ablation.
- Historical issues: 3D vs 2D sensing/belief/coverage differences, MATD3 action saturation/tanh gradient, MASAC alpha/inf/AMP overflow checks, safety CBF intervention, battery/relay/false confirms; these are investigative hypotheses, not automatically unresolved bugs in final reference source.
- `optimization.scalar_metrics` fixed mixed device metrics (`torch.as_tensor(v).detach().cpu()` BEFORE stack) at `4afff25` after first 03/11 CUDA test preflight crash. Tests 19 passed at that time. This is telemetry fix, not algorithm modification.

## 8. Kaggle / GitHub Operations: latest state
- Credentials are private in `/home/aduy/Documents/NCKH/uav_network/key.txt` (not git, permissions 0600). **Never print, paste, log, or commit actual API tokens, passwords, GH/HF/W&B secrets.** File is valid JSON with `kaggle.accounts` named `account_01`...`account_10`, plus `wandb`, `huggingface`. Legacy names `account_17`..`account_26` were renumbered in order to 01..10 without changing token bytes. Old v2 receipts still use legacy names; map by confirmed identity, do not edit receipts.
- GitHub `main` has workflow wiring commit `61a8fe4e8171a2ce86230b71e585fd8eacdf97e9`; repository Secrets names 01..10, `KAGGLE_ENABLED_ACCOUNTS` variable lists 01..10, GitHub supervisor `disabled_manually`. Disabled means new scheduled runs will NOT launch automatically; must not claim autonomous continuation. GitHub workflow references a historical pinned branch for supervisor; check current remote before changing it.
- Historical 03/11 first attempt `outputs/reference-cohort-20261103/` had CUDA preflight crash due to CPU/GPU metric stack; second attempt `outputs/reference-cohort-20261103-v2/` has `plan.json`, `batch.py`, `worker.py`, `observe.py`, per-case `receipt.json`; 4 training jobs launched successfully before user requested stopping.
- **Danger:** v2 `plan.json` pinned `source_commit=4afff25...` and `compile_models=false`, v2 `worker.py` explicitly loads config override `runtime.compile_models=false` and validates commit; it is not a generic launcher for 81902df. Use fresh verified spec only if asked. No blind resubmission of a recorded job, no queue waiting, no W&B status fabrication. Stop was requested and completed; leave old jobs stopped. Cloud account/provider rules must be respected; do not launch 10 concurrent accounts to evade usage limits.
- Source `cloud/cli.py` checks clean git tree, quota, exact account, private credential datasets, create Kaggle kernel, pinned commit, W&B run ID, checkpoints; `github_supervisor.py` supports continuation/CPU visualization. Kaggle CLI does not expose a simple `kernels stop`; W&B public API `Run.stop()` sends graceful stop signal. Stop signals are asynchronous; always verify Kaggle/W&B afterward. In final verification 04/11, 4 W&B states = failed, 4 Kaggle states = error, and active W&B count=0. No datasets/notebooks were deleted.
- GitHub previous `main` workflow contains a `replicate` job for MASAC PQL pinned to a historical commit; do not assume it tests the new reference. Tests `tests/test_github_supervisor_workflow.py` allow both `supervise` and `replicate` jobs; tests passed 55 in last workflow update.
- Computer/GPU context from previous tasks: workstation Ubuntu24.04, Python3.11 environment; cloud Kaggle 2×NVIDIA T4 per run. Check actual Python/PyTorch/CUDA at time of operation.

## 9. Guardrails/checks for new chat
1. If request is new scientific edit, inspect latest branch/config, determine affected modules/tests, do NOT assume main is reference.
2. Read source not just docs; make no ungrounded claims; pinned commits for past runs vs current code differ.
3. Never modify historical W&B metadata, Kaggle receipts or completed results to look like current configuration.
4. Make minimal change, run targeted tests and additional full suite where warranted; record pass/fail and device (CPU vs GPU). Avoid claiming CUDA validation from CPU tests.
5. When editing baseline algorithm core use official author papers/code; cite provenance and clearly document any unavoidable UAV adapter; no hidden performance tweaks.
6. Maintain FP32 eager, compile/fused off, chunk0 and output_normalization=false unless user explicitly changes requirement.
7. Before any launch/stop, recheck exact active jobs, submission receipts, authorized account and quotas; avoid duplicate submissions and service-terms violations.
8. Commit/push on correct branch, verify remote HEAD, update `docs/superpowers/CURRENT_TASK.md` with actual evidence. Report which task finished, what changed, tests, commit, unresolved items.
9. When asked for a specific subtask, finish and report it before taking on unrelated work.

## 10. Kiểm thử mới nhất: reproducibility thực tế của PQL cùng seed 44, 04/11/2026
- **Đã chạy thực tế**, không chỉ suy luận; PyTorch 2.7.1+cu126, Warp 1.18.0, máy local **không có GPU CUDA**, chạy CPU Warp. Test có sẵn `PYTHONPATH=.:tests pytest -q tests/test_pql_training.py -k deterministic_async_mode_repeats_scientific_outcomes_and_weights --disable-warnings`: `1 passed, 11 deselected`.
- Thêm 8 phiên smoke trên chính profile XYZ 6 UAV/20 targets/15 obstacles: mỗi thuật toán MASAC PQL, MATD3 PQL × 2 lần seed44 × 2 chế độ. **Chỉ cho smoke:** `runtime.device=cpu`, `num_envs=2`, `episodes=4`, `world.max_steps=4`, batch=4, replay=128, learning_starts=0, PQL warmup_collection_steps=1, critic_sample_ratio=1, max_train_seconds=0; tắt tracking/eval. Các thứ trên KHÔNG phải training 20k, và không thay config chuẩn.
- `runtime.deterministic=true`: MASAC PQL lần 1/2 `model_hash` bằng nhau, `episode_hash` bằng nhau, `settings_hash` bằng nhau; critic updates 7/7, actor updates 7/7, `collector_overlap=false`. MATD3 PQL cũng có các hash bằng nhau; critic7/7, actor3/3, `collector_overlap=false`.
- `runtime.deterministic=false`: MASAC PQL và MATD3 PQL cả hai có **model_hash khác và episode_hash khác** giữa hai lần seed44; `settings_hash` vẫn trùng, critic/actor updates vẫn trùng. `collector_overlap=true` cả hai. Không được khẳng định same seed tự động cho same weights khi concurrent.
- Evidence đã push: `docs/superpowers/evidence/2026-11-04-pql-seed44-cpu-repro.json`. Raw local test harness, logs và summary: `outputs/repro-seed44-20261104/` (ignored, local only, có thể biến mất ở môi trường mới).
- Đúng code `experiments/pql_system.py`: `self.asynchronous = config.pql.mode == "async"; self.concurrent = self.asynchronous and not config.runtime.deterministic`. Như vậy `pql.enabled=true`, `mode=async`, `deterministic=true` là **split critic/policy learners nhưng không chạy 2 background worker threads**; `self.submit` xử lý `_collect_and_learn` trực tiếp. Không được trình bày điều này là full asynchronous concurrent PQL giống kiến trúc gốc. Khi `deterministic=false`, worker threads bắt đầu chạy concurrent nhưng scheduler nondeterminism làm same-seed diverge trong CPU smoke.
- `runtime.deterministic` được dùng thêm ở `experiments/training.py:seed_runtime`: Python/NumPy/Torch seeds, deterministic cudnn, `torch.use_deterministic_algorithms`, disable TF32 và cudnn benchmark. Không đồng nghĩa evaluation policy deterministic, cũng không phải cách thay đổi loss.
- BASE: `experiments/training.py` tập trung learner.update(batch) và `updates_per_step=8`; PQL: `experiments/pql_system.py` tách Actor/Policy worker và Critic/V worker, mailbox snapshots, `critic_sample_ratio=8`; lịch các updates không hoàn toàn đồng nhất với BASE dù các giá trị đều 8. BASE có `collector_overlap=true` khi `num_envs>1`, không đồng nghĩa BASE hoàn toàn tuần tự.
- Reproducibility CPU smoke **chưa chứng minh** giống hệt trên 2×T4/Kaggle, 4096 envs, batch8192, đầy đủ 20k; cần thí nghiệm cố định work budget và chính xác máy tính đó. `max_train_seconds=36000` có thể làm khác số updates/episode nếu stop theo thời gian. Khi debug reproducibility hạn chế/loại bỏ time budget và giữ same software/driver/commit/GPU.
- **Người dùng hiện chỉ muốn seed44 để debug/so sánh ban đầu và không muốn tốn chi phí chạy 5 seeds.** Không tự launch 5 seeds hoặc yêu cầu multi-seed ngay khi user chưa đồng ý. Phân biệt cùng seed lặp lại (reproducibility) với nhiều seed (thống kê độ ổn định cho paper). Cần minh bạch rằng một seed không đủ kết luận thống kê cuối cùng.
- Yêu cầu cần quyết định trong tương lai: muốn **vừa concurrent vừa same-seed weights trùng** phải thiết kế deterministic ordering/commit protocol, per-worker RNG và policy snapshots; đây chưa được làm, có thể làm giảm overlap. Đừng âm thầm thêm scheduler mới hoặc sửa source-backed algorithm core mà không có yêu cầu rõ.

## 11. Best checkpoint, numerical reproducibility và execution toggles
- `experiments/best_checkpoints.py` lưu `best.pt` kèm `best.json` sau **periodic validation hợp lệ**, chọn theo thứ tự lexicographic `success_rate`, `mean_delivery_rate_percent`, `mean_search_rate_percent`, `mean_return`. Nó không phải average vô điều kiện và không phải episode có return cao nhất.
- `final.pt` lưu cuối run; `latest.pt` và `episode_XXXXXX.pt` theo checkpoint interval; `resume_latest.pt` lưu state phục hồi khi thực hiện. Nếu validation chưa hợp lệ thì fallback `final.pt`, không được gọi nó là proven best.
- `tracking.evaluation_interval_steps=3686400`, `periodic_evaluation_episodes=5`, `periodic_evaluation_seed=1000000044`, `runtime.eval_episodes=30`, `runtime.eval_seed=2000000044`; validation và final test tách seed. Lưu ý evaluation có thể bị skip/busy/timeout; phải xác minh logs nếu best absent.
- `runtime.vectorize_critics=false`: mặc định mỗi Q network chạy Python for-loop. Khi true (`algorithms/parallel_critics.py`), code dùng `torch.func.functional_call` + `torch.vmap`, stack chiều `num_uavs` của **weights**; không hợp nhất 6 MLP thành input lớn 6 lần, không share weights. Nếu xét bật true thì phải benchmark VRAM/thời gian và parity của Q/loss/grad FP32 trên cùng setup, không tự bật trong reference.
- `runtime.critic_chunk_size=0` nghĩa là không chia batch trong các call hỗ trợ chunk, nhưng `vectorize_critics` và chunking là hai chiều khác nhau.
- `runtime.coalesce_idle_polls=true` có metadata và đi vào Warp WorldParams; README mô tả gộp polling khi MAC/radio idle. Chưa xác nhận hiệu năng riêng trong final reference; khi được hỏi phải trace CUDA/native path thay vì suy diễn.
- `matd3.output_normalization=false` là giữ actor source-backed. `amp_mode=off`=FP32; compile_models=false; fused_optimizers=false. Tăng tốc GPU bằng setting khác chỉ sau matched benchmark và approval; không đổi baseline ngầm.
- Tham khảo công khai: https://github.com/Improbable-AI/pql ; https://docs.pytorch.org/docs/stable/notes/randomness.html . Paper chính thức PQL tách Actor/P-learner/V-learner và so wall-clock; PyTorch không bảo đảm bitwise across CPU/GPU/software versions. **PQL hiện tại được code gọi `pql-inspired`, không phải exact replica release gốc.**

## 12. FULL resolved reference config (actual `Config.to_dict()`, gồm explicit + defaults)
Đây là snapshot nội dung resolve khi chạy với profile canonical tại commit khoa học 81902df (tài liệu evidence mới không đổi YAML), 10 namespaces, số khóa dưới mỗi namespace: world=24, sensor=17, network=36, energy=6, reward=16, masac=18, matd3=20, runtime=25, tracking=26, pql=11. Không phải file YAML gốc: các giá trị mặc định trong code đã được thêm vào. Đừng nhầm run override của MASAC/MATD3/PQL với các default này.

```yaml
world:
  motion_dimensions: 3
  altitude_min_m: 0
  altitude_max_m: 150
  obstacle_height_min_m: 30
  obstacle_height_max_m: 120
  generation_version: 3
  map_size_m: 1000.0
  num_uavs: 6
  num_targets: 20
  num_obstacles: 15
  dt_s: 1.0
  max_steps: 1500
  max_speed_mps: 20.0
  max_accel_mps2: 2.0
  safety_distance_m: 15.0
  obstacle_clearance_m: 10.0
  motion_safety: cbf
  obstacle_radius_min_m: 30.0
  obstacle_radius_max_m: 70.0
  launch_radius_m: 150.0
  launch_min_spacing_m: 30.0
  target_exclusion_radius_m: 200.0
  gcs_position_m: null
  placement_attempts: 5000
sensor:
  model: bayesian
  grid_cell_m: 25
  belief_prior: 0.5
  altitude_anchors:
  - 0
  - 50
  - 100
  - 150
  pd:
  - 0.99
  - 0.9
  - 0.8
  - 0.7
  pf:
  - 0.01
  - 0.1
  - 0.2
  - 0.3
  confirmation_threshold: 0.99
  fine_altitude_m: 50
  belief_patch_cells: 5
  belief_coarse_cells: 5
  verified_empty_belief: 0.01
  altitude_m: 100
  full_fov_deg: 90
  obstacle_occlusion: true
  visible_target_slots: 4
  visible_obstacle_slots: 3
  coverage_grid_cells: 0
network:
  enabled: true
  receiver_admission_mask: true
  rediscovery_recovery: true
  ack_uses_data_power: true
  upstream_path: third_party/UavNetSim
  peer_range_m: 500.0
  gcs_range_m: 500.0
  tx_power_min_w: 0.001
  tx_power_max_w: 0.4
  tx_power_mapping: log
  power_control: learned
  link_budget_margin_db: 3.0
  pa_efficiency: 0.35
  tx_circuit_power_w: 0.1
  rx_circuit_power_w: 0.1
  power_limited_admission: true
  link_state_observation: true
  report_bytes: 1024
  buffer_bytes: 3072
  report_ttl_s: 300.0
  payload_bytes: 1024
  packet_lifetime_s: 10.0
  max_queue_packets: 64
  queue_poll_us: 1000.0
  max_retries: 5
  obstacle_polygon_sides: 16
  bit_rate_bps: 2000000.0
  carrier_frequency_hz: 2400000000.0
  bandwidth_hz: 22000000.0
  sinr_threshold_db: 6.0
  cca_threshold_dbm: -82.0
  noise_density_dbm_hz: -174.0
  noise_figure_db: 7.0
  nlos_model: urban
  propagation_delay_enabled: true
  obstacle_extra_loss_db: 0.0
energy:
  enabled: true
  battery_capacity_j: 277200.0
  idle_power_w: 20.0
  hover_power_w: 120.0
  speed_squared_coefficient: 0.5
  acceleration_squared_coefficient: 2.0
reward:
  discovery: 10.0
  delivery: 50.0
  blocked_motion: 0.2
  expired_report: 10.0
  dropped_report: 10.0
  step_cost: 0.01
  communication_energy: 0.01
  propulsion_energy_per_kj: 0.01
  communication_efficiency: 1.0
  shield_correction: 0.1
  shield_intervention: 0.5
  unsafe_action: 1.0
  information_gain: 10.0
  investigation_shaping: 10.0
  shaping_gamma: 0.99
  investigation_belief_threshold: 0.8
masac:
  hidden_sizes:
  - 256
  - 256
  share_actor_parameters: false
  share_critic_parameters: false
  replay_capacity: 1000000
  batch_size: 8192
  learning_starts: 131072
  updates_per_step: 8
  gamma: 0.99
  tau: 0.005
  learning_rate: 0.0003
  grad_clip_norm: 10.0
  grad_clip_mode: none
  initial_alpha: 1.0
  target_continuous_entropy: -4.0
  target_discrete_entropy_ratio: 0.2
  target_discrete_entropy_final_ratio: 0.2
  target_discrete_entropy_anneal_updates: 0
  mask_aware_discrete_entropy_target: false
matd3:
  hidden_sizes:
  - 256
  - 256
  share_actor_parameters: false
  share_critic_parameters: false
  replay_capacity: 1000000
  batch_size: 8192
  learning_starts: 131072
  updates_per_step: 8
  gamma: 0.99
  tau: 0.005
  learning_rate: 0.0003
  grad_clip_norm: 0.5
  grad_clip_mode: per_parameter
  policy_delay: 2
  target_noise_std: 0.2
  target_noise_clip: 0.5
  exploration_noise_std: 0.1
  output_normalization: false
  initial_epsilon: 1.0
  final_epsilon: 0.05
  epsilon_decay_steps: 100000
runtime:
  algorithm: masac
  seed: 44
  eval_seed: 2000000044
  episodes: 20000
  eval_episodes: 30
  device: cuda
  torch_threads: 1
  output_dir: outputs/reference-baseline-steps1500-t20-o15-learnedpower-report1kb-xyz-sensing
  checkpoint_interval: 128
  num_envs: 4096
  environment_backend: warp_packet
  shared_memory: true
  collector_overlap: true
  replay_device: cpu
  replay_cpu_fraction: 0.75
  deterministic: true
  compile_models: false
  amp_mode: 'off'
  coalesce_idle_polls: true
  fused_optimizers: false
  vectorize_critics: false
  critic_chunk_size: 0
  log_interval_steps: 4096
  warm_start_checkpoint: none
  max_train_seconds: 36000
tracking:
  enabled: true
  credentials_file: key.txt
  entity: truongdaoanhduy-uit
  project: uav_search_target
  run_name: auto
  group: uav-reference-baseline-xyz-20t15o-1500
  tags:
  - uav-network
  - reference-baseline
  - source-backed-algorithm-core
  - amp-off
  - fp32-eager
  - scenario-v3
  - controlled-20t15o
  - targets20-obstacles15
  - power-learned
  - full20k
  - replay1m
  - cpu75-gpu25
  - batch8192
  - env4096
  - steps1500
  - seed44
  - base-pql-comparison
  - report1kb
  - radio-energy
  - cpu-visualize
  - xyz
  - bayesian-sensing
  - grid25m
  - belief-prior-0.5
  log_checkpoint: true
  checkpoint_upload_interval_seconds: 300.0
  resumable_checkpoint_interval_seconds: 3600.0
  resumable_checkpoint_max_uploads: 4
  local_checkpoints_to_keep: 2
  checkpoint_store: huggingface
  checkpoint_repo: duy95/uav-search-checkpoints
  checkpoint_repo_private: false
  system_metrics: true
  artifact_timeout_seconds: 900
  evaluation_interval: 0
  evaluation_interval_steps: 3686400
  periodic_evaluation_episodes: 5
  periodic_evaluation_seed: 1000000044
  evaluation_timeout_seconds: 900
  diagnostic_window: 20
  delivery_gap_threshold_percent: 40.0
  return_drop_threshold: 20.0
  blocked_rate_threshold: 0.3
pql:
  enabled: false
  mode: async
  critic_sample_ratio: 8
  warmup_collection_steps: 32
  max_pending_collections: 2
  max_pending_policy_updates: 2
  collector_device: auto
  learner_device: auto
  policy_device: auto
  worker_timeout_s: 600
  checkpoint_replay: true
```

## 13. Current file index / developer entrypoints
- Repo trên GitHub: https://github.com/truongdaoanhduy/uav_network/tree/baseline/reference-masac-matd3-20261103 . Run-specific experiment history có thể nằm ở `/home/aduy/Documents/NCKH/uav_network/outputs/...`, không phải mọi file được commit.
- CLI: `uav_network/__main__.py`, `experiments/cli.py`, `cloud/cli.py`. Data classes: `configuration.py`, `configuration_xyz.py`.
- Algorithms: `algorithms/models.py`, `masac.py`, `matd3.py`, `action_utils.py`, `parallel_critics.py`, `critic_chunks.py`, `optimization.py`, `precision.py`, `replay.py`, `hybrid_replay.py`, `checkpoints.py`.
- Physical/safety: `environment/{world,scenario_recovery,env,vector,motion,xyz_motion,cbf,avoidance,energy,geometry,xyz_geometry}.py`, plus `environment/gpu/`.
- Sensor/obs: `environment/{sensing,bayesian,coverage,observations,xyz_observations}.py`, plus GPU counterpart.
- Network: `communication/{backend,upstream,link_budget,reports,routing,energy,diagnostics,telemetry}.py`, plus `communication/gpu/{phy,mac,power,events,streaming,reports}.py`.
- Training and telemetry: `experiments/{training,pql_system,pql_training,pql_workers,pql_replay,async_evaluation,best_checkpoints,evaluation,evaluation_snapshots,comparison,comparison_metrics,study,tracking,diagnostics,reproducibility,run_journal,checkpoint_store}.py`.
- Checkpoint/recovery: `experiments/{fixed_checkpoints,recovery_checkpoints,environment_checkpoints,pql_checkpoints,continuation,local_checkpoints,overflow_capture,progress_watchdog}.py`.
- Kaggle/CI: `cloud/{cli,client,credentials,kernel,remote,controller,continuation,github_supervisor,execution_check,visualize}.py`, `.github/workflows/uav_external_supervisor.yml`, `scripts/sync_github_supervisor_secrets.py`.
- Scientific tests: `tests/test_reference_baseline_contract.py`, `tests/test_reference_critic_gradients.py`, `tests/test_pql_training.py`, `tests/test_pql_system.py`, `tests/test_xyz_training.py`, `tests/test_xyz_algorithms.py`; browse all tests by `find tests -maxdepth 1 -name 'test_*.py'`.
- Legacy source: `/home/aduy/Documents/NCKH/uav_search` (earlier 2D swarm), `/home/aduy/Documents/NCKH/uav_search_target` (3D predecessor). Legacy findings and historical commits are NOT proof of code currently present in reference; read files before reusing.

## 14. User communication / next task handoff
- User speaks Vietnamese, wants detailed but grounded explanations and actual edits only when requested. User is focused on source-backed publication baseline, correct equations, unbiased comparisons, W&B logs, per-task audit, performance of MASAC/MATD3 BASE/PQL, and seed44 repeatability.
- Start by reading this doc + CURRENT_TASK + provenance and fresh `git status` and resolved YAML, then continue with **new explicit request only**. Don't re-ask questions already answered. Don't invent logs, currently running jobs, performance gains, GPUs, or algorithm correctness. Never expose API tokens. Report exact branch/commit + tests + hardware + evidence; specify partial work clearly.
- Preserve 0/false for critic chunking/vectorization/compile/fused/output norm until user explicitly asks differently. Don't start training, change provider accounts, or enable GitHub supervisor without explicit task authorization.

---

# YÊU CẦU MỚI CỦA TÔI
(Dán yêu cầu mới ngay dưới dòng này.)
