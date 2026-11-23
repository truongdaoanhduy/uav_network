# Current status supersedes the proposal snapshot below

The later explicit request on2026-11-06 implemented critic chunk2048 for BOTH
learners and selected shared actor/critic in the canonical reference YAML.
The1024 profiles below remain historical proposals, not the selected recipe.
Statements below that MATD3 ignores chunk describe sourcebceaf23 and earlier.
Current differences and verification:
[Matched shared2048 report](MATCHED_SHARED_2048_CONFIG_DIFF_2026-11-06.md).

---

# Cấu hình công bằng MASAC/MATD3: đối chiếu 02/11, hiện tại và hai đề xuất

Ngày06/11/2026. Source trước khi tạo đề xuất:3d09bc65d26bc7921d5232c934474f16bfe226a7.
Đây là giải thích và các profile ĐỀ XUẤT để review. Không chuyển manifest/default profile, không đổi learner/source, không launch training.

## 1. “Cùng config” phải hiểu ở mức nào

Công bằng cần cùng bài toán, thông tin quan sát, action interface, reward/safety, kiến trúc chia sẻ đã khai báo, dataset/replay capacity, batch/update budget, seed/scenario và evaluation protocol.
Không cần và không thể ép mọi tham số SAC/TD3 thành giống nhau: MASAC có alpha/entropy/learned std; MATD3 có policy delay, target smoothing và exploration noise. Ép chúng giống hệt sẽ thay thuật toán.

Hai cách so sánh hợp lệ:
- Source-specific recipe: giữ chi tiết optimizer/clipping theo nguồn đã chọn cho từng thuật toán; ngân sách và study settings chung. Kết luận áp dụng cho hai implementation này.
- Matched training recipe: chủ động đặt cả hai cùng quy tắc clipping/kiến trúc/runtime. Phải công bố recipe; không gọi là tái lập y hệt từng author source.

Profile A dưới đây giữ recipe lõi mới. Profile B dùng MASAC hyperparameters gần02/11 và global clipping10 cho cả hai để giảm khác biệt recipe. B là biến thể thí nghiệm, không phải tái tạo toàn bộ code02/11.

## 2. Chunk có làm học kém hơn không?

Không có quy luật “không chunk học tốt nhất” hoặc “càng ít phần thì reward càng cao”.
Trong implementation hiện tại, chunk chỉ chia phần tính critic; logical batch và optimizer update không bị chia thành nhiều lần học.
Với B8192:

| critic_chunk_size |Số phần original batch|Ý nghĩa|
|---:|---:|---|
|0|1 full call|Không chunk; không checkpoint trong helper|
|8192 hoặc lớn hơn|1 full call|Helper cũng không chia/checkpoint nếu batch không lớn hơn size|
|4096|2|Mỗi phần lớn hơn, thường cần nhiều peak activation memory hơn1024|
|2048|4|Một điểm đánh đổi cần đo|
|1024|8|Mốc đề xuất ban đầu; từng được dùng ở02/11|
|512|16|Ít bộ nhớ mỗi call hơn, thường nhiều overhead hơn|

Helper còn dùng activation checkpointing: tính lại critic activations trong backward thay vì giữ tất cả graphs.
Không phải checkpoint lưu best.pt/resume.pt. Không chunk không mặc nhiên nhanh hơn trên mọi GPU; thường ít overhead nhưng phải benchmark.
Chunk không bảo đảm hết mọi OOM: replay, env, optimizer, allocator/workspace và host RAM vẫn chiếm bộ nhớ.

Với MASAC shared N6,K7: full B8192 mở rộng thành344064 critic rows; chunk1024 chỉ43008 rows/call.
Một hidden output FP32 từ336MiB thành42MiB/call. Không cộng tất cả tensors để suy exact peak.

Điều kiện giữ ý nghĩa toán học:
- Cùng minibatch8192, đủ7đích, cùng sampled actions và RNG convention.
- Giữ full loss/gradient reduction; một optimizer step/logical update.
- Không optimizer.step/clip riêng từng phần, không giảm số đích/loại samples.
- Critic function recompute phải thuần; code hiện tại lấy sampling bên ngoài, không BatchNorm/Dropout trong critic.
- Floating-point reduction có thể khác trên phần cứng khác; không cam kết mọi trajectory giống từng bit.

Chứng cứ có sẵn: CPU shared oracle6agents,4continuous,O198,S565,B11,hidden16,3updates; chunk0/4 với và không LayerNorm giữ cùng RNG, max parameter difference0 trong mẫu đó. Không phải full8192 GPU validation.

## 3. Một bên chunk, một bên không chunk có thiên vị không?

Không tự động thiên vị về chất lượng nếu chunk giữ nguyên update toán học. Chia phép nhân ma trận để vừa VRAM không phải thêm exploration hay đổi reward.
Khi so tốc độ/wall-clock phải ghi execution settings và dùng cùng hardware/VRAM budget; throughput có thể khác, nên báo cả transitions/sec và chất lượng theo số transitions.

Quan trọng với CODE ĐANG CÓ:
- runtime.critic_chunk_size chỉ được MASAC dùng; MATD3 chưa gọi helper này.
- Đặt1024 trong config chung không khiến MATD3 tự có chunk. Hai file đề xuất ghi rõ điều đó.
- Không thêm chunk vào MATD3 chỉ để có cùng tên cờ rồi gọi là công bằng hơn. Nếu sau này cần bounded MATD3 execution, phải implement và kiểm chứng riêng.

## 4. Vectorize có cần thiết không?

runtime.vectorize_critics đổi cách thực thi các critic qua batched/functional/vmap operations; không tăng số neuron, không đổi256/256 thành mạng lớn hơn, không thêm critic mới.
Nó khác num_envs4096: tắt vectorize_critics không tắt parallel environments.

- Independent critics: có thể giảm Python-loop overhead; cũng có chi phí stack/batch activations và phụ thuộc GPU/backend. Benchmark mới biết nhanh hơn.
- Shared critic: chỉ một TwinCritic module (bên trong vẫn Q1,Q2). MASAC shared destination enumeration và shared actor batching đã là batched paths, không cần cờ này để hoạt động.
- MATD3 shared actor-value batch cũng được chọn vì share_critic_parameters=true. Riêng critic regression/target helper vẫn có thể đổi loop1-module↔vmap1-module khi bật cờ; không nói toàn bộ flag vô hiệu.
- Đề xuất false cho cả hai để giảm biến thực thi trong lần đối chiếu đầu. Khi dùng independent architectures mới benchmark true/false riêng.

torch.vmap.chunk_size là một API parameter khác; không nhầm nó với runtime.critic_chunk_size tự dự án dùng trong critic_chunks.py.

## 5. Các thông số study cũ/mới vốn đã giống nhau

| Nhóm |Giá trị chung đã audit|
|---|---|
|World|6UAV,map1000m,20targets,15obstacles,XYZ,horizon1500|
|Sensing/safety/reward|Cùng world/sensor/network/energy/reward namespaces old/new; giữ nguyênCBF/Bayes|
|Actions|XYZ+RF4continuous,7destinations|
|Observation/critic state|O198,S565 với profile này|
|Sharing của các cloud runs|Actor true,critic true cho cả hai|
|Hidden widths|256/256|
|Learning rate/gamma/tau|3e-4/.99/.005|
|Replay/batch/warmup|1M/8192/131072|
|Training rate|8critic updates/vector collection|
|Environment count/seed|4096/44|
|Replay placement|75%CPU/25%GPU|
|AMP/fused|off/false|
|Budget|20000episodes,1500steps/episode,36000s wall-clock cap|
|Validation/test|Validation5episodes seed1000000044; final test30episodes seed2000000044|

Cùng episodes20000 không đảm bảo cùng transitions nếu độ dài episode khác. Cùng36000s không đảm bảo cùng số updates vì tốc độ khác.
Đánh giá ở các mốc global_step chung để so sample efficiency; đánh giá theo thời gian riêng để so tốc độ. Giữ cùng hyperparameter-tuning budget nếu tuning.
Seed44 tạo điều kiện lặp lại, không khiến hai thuật toán phải có cùng actions/trajectories.

## 6. MASAC cũ so mới: config và tác dụng

|Setting|02/11|05/11 và reference hiện tại|Ý nghĩa/đánh đổi|
|---|---|---|---|
|initial_alpha|0.2|1|Alpha ban đầu cao hơn tăng trọng số entropy lúc đầu, không bảo đảm reward tốt hơn|
|target_continuous_entropy|-4|-4|Target danh nghĩa4controls; code còn điều kiện inactive/silentRF|
|target_discrete_entropy_ratio|0.9|0.2|Cũ target phần lớn entropy khả dụng; mới weight fixed log7|
|mask_aware_discrete_entropy_target|True theo source cũ|False|Cũ theo valid_count; mới theo tổng số actions7|
|final_ratio/anneal|Không có schedule mới được serialize; hành vi cũ0.9 cố định|0.2/0updates|Không anneal trong reference|
|grad_clip_mode/norm|Global10; mode suy từ code cũ|none; norm10 có thể vẫn serialize nhưng không dùng|Clip hạn chế gradient norm; none không cắt|
|critic_chunk_size|1024|0|Giảm peak activations/checkpoint; không phải reward trick|
|vectorize_critics|True|False|Execution; shared dominant enumeration vẫn có|
|compile_models|False|False|Không có thay đổi compile MASAC ở cặp này|
|collector_overlap|True|False ở cloud05/11; direct reference defaultTrue|Đổi timing collection/learning và replay chronology|
|actor_layer_norm|Không có|Run05/11 không có; current referenceTrue|Chuẩn hóa hidden; architecture choice|

Target entropy mới không phải lúc nào cũng nhỏ hơn: chỉ1valid action thì old target0 nhưng fixed target≈0.389. Đây là tradeoff phải theo dõi valid_count/alpha_d, không tự gọi một cách luôn tốt hơn.

Những khác biệt là CODE, không phải chỉ điền YAML:
|Thành phần|02/11|Source hiện tại|
|---|---|---|
|Gaussian log_std|[-5,2]|[-20,2], constants trong models.py|
|Hard log-alpha cap|[-20,5]|Không cap|
|Critic loss shared|MSE1+MSE2|0.5×(MSE1+MSE2)|
|Alpha loss reduction|mean agents|sum agents|
|Guard alpha_d cho1valid|Có|Current bật lại guard khi mask_aware=true; referenceFalse không bật|
Bởi vậy, YAML gần02/11 không phải khôi phục đúng toàn bộ code02/11. Không thêm các key không tồn tại như alpha_max/log_std_min để giả vờ đã khôi phục chúng.

## 7. MATD3 cũ so mới: config và tác dụng

|Setting/cơ chế|02/11|05/11 và core hiện tại|Ý nghĩa|
|---|---|---|---|
|output_normalization|True|False|Old rescale rawXYZ trước tanh; tắt để bỏ bespoke output map|
|grad_clip_mode/norm|Global10|Per-parameter0.5|Hai kiểu clipping khác nhau; không diễn giải chỉ là20×learning-rate difference|
|policy_delay|2|2|Actor/targets cập nhật mỗi2critic updates|
|target_noise_std/clip|.2/.5|.2/.5|Làm mượt target actions|
|exploration_noise_std|.1|.1|Behavior exploration cho deterministic policy|
|epsilon schedule|1→.05/14745600transitions|Fields legacy còn tồn tại nhưng sampler mới không dùng|Đừng tuning một field đã không hoạt động|
|actor/target categorical|Hard/ST và hard targets|Soft Gumbel|Đổi estimator/representation, không chỉ runtime|
|Behavior category|Epsilon-greedy|Categorical sampling|Đổi cơ chế exploration categorical|
|Peer actions actor loss|Current policies detach|Replay peers|Theo MATD3 source/paper convention đã chọn|
|actor_layer_norm|Không|Run05/11 không; current referenceTrue|Hidden norm không bound finalLinear|
|critic_chunk_size|1024 có trong config nhưng learner không dùng|0, vẫn không dùng|Không gán chunk là nguyên nhân MATD3 regression|
|compile_models|True|False|Compilation execution, phải đo tốc độ/backend|
|vectorize_critics|True|False|Cách chạy critic; không đổi widths|
|collector_overlap|True|False ở cloud05/11; direct reference defaultTrue|Data chronology/lag có thể khác|

Core twin-min target,-Q1,delay2 và Gaussian smoothing vẫn tồn tại. Nhiều thay đổi gộp cùng lúc; không chứng minh chỉ output_norm gây hồi quy.

## 8. Cấu hình chung đề xuất: phần cần khóa giống nhau

Hai file đều giữ nguyên scenario/sensing/network/energy/reward của reference hiện tại.
Cả hai thuật toán trong mỗi file cùng:
- shared actor/critic true; hidden256/256; actor LayerNorm true theo yêu cầu đã duyệt.
- LR3e-4,gamma.99,tau.005; batch8192,replay1M,warmup131072,8criticupdates/collection.
- E4096,seed44,deterministicTrue,FP32,compileFalse,fusedFalse,vectorizeFalse.
- chunk1024 chung nhưng current MATD3 không sử dụng.
- collector_overlapFalse được pin để không vô tình đổi chronology so cloud05/11.
-75%CPU/25%GPU replay,1500horizon,20000episodes,36000s cap.
- Validation/test seeds/budget thống nhất; output directories phải riêng từng run.
- Khi chạy thật còn phải khóa cùng PyTorch/CUDA/kernel image. YAML này không tự pin dependency build.

Policy delay vẫn riêngMATD3: cùng8criticupdates không buộc MASAC cũng delay2. SAC cập nhật actor/alpha thường xuyên là khác biệt thuật toán.
SAC cần Gaussian/alpha và categorical entropy; TD3 cần smoothing/exploration noise. Các key này không có “giá trị chung” có nghĩa.

Nếu muốn plain architecture gần original sources hơn, đổi actor_layer_norm=false cho CẢ HAI trong một variant riêng; không một bêntrue/một bênfalse rồi gọi chỉ so thuật toán.

## 9. Profile A: source-specific core, memory-bounded execution

File:configs/comparison-proposed-matched-shared-ln-memory1024.yaml

A giữ MASAC alpha1/ratio.2/fixed target/no-clipping và MATD3 per-parameter clip.5 của source recipe đang dùng.
Các trường này khác có chủ đích vì theo từng nguồn, không phải mọi recipe buộc giống nhau.
A KHÔNG tự nhận là literal upstream implementation; shared/LN/hybrid adapter và memory execution được khai báo. Strict REFERENCE_BASELINE contract của repo vẫn áp dụng riêng canonical profile, không được lặng lẽ sửa để hợp thức nhãn.

Phần lõi rút gọn (các file đầy đủ chứa world/network/reward/sensor/tracking):
```yaml
runtime:
  seed: 44
  num_envs: 4096
  episodes: 20000
  deterministic: true
  environment_backend: warp_packet
  device: cuda
  replay_device: cpu
  replay_cpu_fraction: 0.75
  critic_chunk_size: 1024  # MASAC consumes; MATD3 currently ignores
  vectorize_critics: false
  collector_overlap: false
  compile_models: false
  amp_mode: "off"
  fused_optimizers: false
masac:
  hidden_sizes: [256, 256]
  share_actor_parameters: true
  share_critic_parameters: true
  actor_layer_norm: true
  learning_rate: 0.0003
  gamma: 0.99
  tau: 0.005
  replay_capacity: 1000000
  batch_size: 8192
  learning_starts: 131072
  updates_per_step: 8
  initial_alpha: 1.0
  target_continuous_entropy: -4.0
  target_discrete_entropy_ratio: 0.2
  target_discrete_entropy_final_ratio: 0.2
  target_discrete_entropy_anneal_updates: 0
  mask_aware_discrete_entropy_target: false
  grad_clip_mode: none
matd3:
  hidden_sizes: [256, 256]
  share_actor_parameters: true
  share_critic_parameters: true
  actor_layer_norm: true
  learning_rate: 0.0003
  gamma: 0.99
  tau: 0.005
  replay_capacity: 1000000
  batch_size: 8192
  learning_starts: 131072
  updates_per_step: 8
  policy_delay: 2
  target_noise_std: 0.2
  target_noise_clip: 0.5
  exploration_noise_std: 0.1
  output_normalization: false
  grad_clip_mode: per_parameter
  grad_clip_norm: 0.5
pql:
  enabled: false
  mode: async
  critic_sample_ratio: 8
  warmup_collection_steps: 32
```

## 10. Profile B: MASAC02/11-like hyperparameters, cùng global clip10

File:configs/comparison-proposed-matched-shared-ln-memory1024-masac02novlike.yaml

B giữ phần chung củaA nhưng thay các dòng:
```yaml
masac:
  initial_alpha: 0.2
  target_discrete_entropy_ratio: 0.9
  target_discrete_entropy_final_ratio: 0.9
  target_discrete_entropy_anneal_updates: 0
  mask_aware_discrete_entropy_target: true
  grad_clip_mode: global
  grad_clip_norm: 10.0
matd3:
  grad_clip_mode: global
  grad_clip_norm: 10.0
```

B đáp ứng thử nghiệm cùng clipping recipe, đồng thời gần MASAC02/11 hơn về alpha/entropy/clip.
Đây là matched-hyperparameter study variant. Nó thay recipe clip.5 của MATD3 hiện tại; không gọi là tái lập nguyên bản pinned Ackermann source.
B vẫn giữ MSE factor.5,alpha loss sum,log_std[-20,2],không alpha hard-cap,Gumbel/replay-peer core mới và hiddenLN. Output normalization vẫnFalse. Do đó KHÔNG phải rollback code cũ.

Nếu ưu tiên source-backed recipe hiện tại, dùngA làm mốc. Nếu muốn kiểm tra giả thuyết hparams MASAC02/11 và chung clipping, dùngB thành cặp matched riêng. Không trộn MASAC củaB với MATD3 củaA rồi gọi cùng recipe.

## 11. Bốn case từ mỗi profile

|Case|runtime.algorithm|pql.enabled|
|---|---|---|
|MASAC BASE|masac|false|
|MASAC PQL|masac|true|
|MATD3 BASE|matd3|false|
|MATD3 PQL|matd3|true|

Đổi hai trường lựa chọn algorithm/PQL theo bảng, cùng filename profile và common settings; output/run/group identity phải phân biệt.
PQL deterministicTrue làm các stage serialize trong implementation hiện tại, dù mode=async; không cam kết hiệu năng concurrency đúng async paper.
Khi đánh giá, so MASACBASE với MATD3BASE và MASACPQL với MATD3PQL trước; BASE/PQL scheduler là một factor khác.

Lệnh xem cấu hình, KHÔNG train:
```bash
python -m uav_network config --config configs/comparison-proposed-matched-shared-ln-memory1024.yaml --set runtime.algorithm=matd3 --set pql.enabled=true
```

## 12. Kiểm chứng và giới hạn

- Đã tạo2profile proposal, không thay canonical/reference hoặc các manifests đang dùng.
- Đã load/validate8 combinations:2profiles×2algorithms×BASE/PQL.
- Đã assert11common architecture/hyperparameter fields giống giữa hai algorithm blocks.
- Đã construct CPU learners và deterministic actions hữu hạn trong[-1,1],1actor/1TwinCritic module,2hiddenLayerNorm cho cả8case.
- CờPQL được validate và dùng cùng learner constructor; đây không phải chạy integrationPQL/fulltraining mới.
- Không đo full8192GPU VRAM/throughput trong task này; không bảo đảm chunk1024 hết mọi OOM hoặc một profile hội tụ tốt hơn.
- Kiểm chứng chunk mathematical/RNG equivalence nhỏ và source audit được lưu trong docs/superpowers/evidence/2026-11-06-four-run-code-audit.json.
- Source contract/structure guards:13passed,557subtests in2.15s; canonical source/profile unchanged.
- Evidence riêng của các proposal:docs/superpowers/evidence/2026-11-06-proposed-matched-configs.json.

## Nguồn

- Source hiện tại:algorithms/critic_chunks.py,parallel_critics.py,masac.py,matd3.py; config cũW&B đã audit theo source05beef6c.
- Báo cáo đầy đủ:docs/methods/MASAC_MATD3_CODE_COMPARISON_2026-11-06.md.
- PyTorch checkpoint:https://docs.pytorch.org/docs/2.11/checkpoint.html.
- PyTorch vmap:https://docs.pytorch.org/docs/2.11/generated/torch.vmap.html.
- Nguồn baseline và declared adapters:docs/methods/REFERENCE_BASELINE_PROVENANCE.md.
