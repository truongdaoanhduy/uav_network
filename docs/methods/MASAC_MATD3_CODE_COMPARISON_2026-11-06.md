# So sánh source MASAC/MATD3: 02/11, các run 05/11 và code hiện tại

Quan sát online ngày 06/11/2026, giờ Việt Nam. Repo: https://github.com/truongdaoanhduy/uav_network.
Source được audit: `db5345a63a5e5fab5843109dc46a952e49f1814b`, branch `experiment/hfbest-embedded-source-steps1500-20261105`.
Phạm vi: đọc W&B config/summary/system history/status/thread logs; so sánh Git theo source commit run khai báo; đọc source tác giả/thư viện; kiểm tra CPU nhỏ. Không thay đổi learner/config hoặc khởi chạy training.

## 1. Kết luận chính

1. **Cả bốn run cũ đều dùng source 05beef6c.** Các run mới dùng574f36c6(BASE) hoặc ab2e953d(PQL). Hai commit mới khác ở cloud preflight, không khác thuật toán.
2. **Tắt chunk làm MASAC mất cả chia batch và activation checkpointing.** Source và mẫu VRAM phù hợp với áp lực bộ nhớ tăng; chưa có traceback chứng minh OOM là nguyên nhân crash.
3. **Chunk không được MATD3 gọi.** Không dùng việc đổi1024→0 để giải thích trực tiếp chất lượng MATD3.
4. **MATD3 mới có bằng chứng head XYZ bão hòa tanh và actor gradient0.** Nhiều yếu tố đổi cùng lúc nên không quy toàn bộ hồi quy cho một cờ.
5. **MASAC cũ chạy ổn hơn về tiến trình nhưng chưa hội tụ tốt:** delivery rolling cuối4.5% BASE/15% PQL; dừng vì time budget, chưa đủ20k episodes.
6. **Code hiện tại đã thêm hidden LayerNorm sau các run lỗi.** Không có run sản xuất được audit nào chứa thay đổi này. LayerNorm chưa được kiểm chứng hội tụ.
7. Có thể giữ lõi thuật toán mới và dùng thực thi chunk/checkpoint tiết kiệm bộ nhớ, không cần khôi phục entropy heuristics/clipping/output normalization của bản cũ. Cần công bố setting và giữ cùng logical batch/gradient/optimizer steps.

## 2. Danh tính run và source

| Nhóm | W&B run | Source |
|---|---|---|
| MASAC BASE02/11, suffix d522cf | [c8d5ce6b](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/c8d5ce6b) | 05beef6cd0d97d39d51b220f9f7789e915b5bc10 |
| MASAC PQL02/11, suffix e9729e | [2ce9fb9e](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/2ce9fb9e) | 05beef6cd0d97d39d51b220f9f7789e915b5bc10 |
| MATD3 BASE02/11, suffix e3778d | [db61bd0c](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/db61bd0c) | 05beef6cd0d97d39d51b220f9f7789e915b5bc10 |
| MATD3 PQL02/11, suffix c338a0 | [d51b0350](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/d51b0350) | 05beef6cd0d97d39d51b220f9f7789e915b5bc10 |
| MASAC BASE05/11 | [0006c2a7](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/0006c2a7) | 574f36c6d36d6182d3204a53f9e497498d3470f8 |
| MASAC PQL05/11 | [550c3c6b](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/550c3c6b) | ab2e953d37520eb5896700f1264d1e4ba80fdfad |
| MASAC PQL replica05/11 | [78d36ef3](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/78d36ef3) | ab2e953d37520eb5896700f1264d1e4ba80fdfad |
| MATD3 BASE05/11 | [140e6169](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/140e6169) | 574f36c6d36d6182d3204a53f9e497498d3470f8 |
| MATD3 PQL05/11 | [ead641c4](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/ead641c4) | ab2e953d37520eb5896700f1264d1e4ba80fdfad |

Old metadata Git commit khớp source_git_commit. Các run embedded-source mới không có Git metadata tại runtime; source_git_commit là provenance được run ghi lại. Không suy source từ suffix tên run.

Full tên bốn run do người dùng chỉ định:
- TRAIN-MASAC-PQL-SHARED-ACTOR-CRITIC-XYZ-BAYES-HYBRID-C75-G25-E4096-B8192-R1M-G3-S44-20261102-033955-e9729e
- TRAIN-MASAC-BASE-SHARED-ACTOR-CRITIC-XYZ-BAYES-HYBRID-C75-G25-E4096-B8192-R1M-G3-S44-20261102-034013-d522cf
- TRAIN-MATD3-BASE-SHARED-ACTOR-CRITIC-XYZ-BAYES-HYBRID-C75-G25-E4096-B8192-R1M-G3-S44-20261102-031752-e3778d
- TRAIN-MATD3-PQL-SHARED-ACTOR-CRITIC-XYZ-BAYES-HYBRID-C75-G25-E4096-B8192-R1M-G3-S44-20261102-031752-c338a0

Đọc diff đầy đủ:
- [02/11 → HEAD được audit](https://github.com/truongdaoanhduy/uav_network/compare/05beef6cd0d97d39d51b220f9f7789e915b5bc10...db5345a63a5e5fab5843109dc46a952e49f1814b).
- [BASE05/11 → PQL05/11 source](https://github.com/truongdaoanhduy/uav_network/compare/574f36c6d36d6182d3204a53f9e497498d3470f8...ab2e953d37520eb5896700f1264d1e4ba80fdfad).

## 3. Kết quả thực tế, không đánh đồng finished với hội tụ

| Run | State | Episodes đã ghi | Joint transitions | Learner updates | Search rolling cuối | Delivery rolling cuối | Return rolling cuối | Dừng vì |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MASAC BASE cũ | finished |16384|25034752|48648|19.75%|4.5%|-2549.36|time_budget|
| MASAC PQL cũ | finished |16362|22880256|44440|20%|15%|-2233.63|time_budget|
| MATD3 BASE cũ | finished |20000|24219648|47056|60.75%|59%|-160.06|episode_budget|
| MATD3 PQL cũ | finished |20000|24813568|48216|69.75%|67.75%|-34.49|episode_budget|
| MASAC BASE mới | crashed |0|2506752|4648|chưa có|chưa có|chưa có|không có stop_reason|
| MASAC PQL mới | crashed |0|143360|32|chưa có|chưa có|chưa có|không có stop_reason|
| MASAC replica mới | crashed |0|126976|0 được ghi nhận|chưa có|chưa có|chưa có|không có stop_reason|
| MATD3 BASE mới | finished |20000|30720000|59752|0%|0%|-4221.48|episode_budget|
| MATD3 PQL mới | finished |20000|30720000|59752|0%|0%|-4221.48|episode_budget|

Rolling là20 completed episodes, không phải best-ever hay final held-out evaluation. Các run cũ đều có rolling mission success0. So sánh trên chưa ghép cùng transitions và budget.
Riêng MASAC BASE mới: summary cuối lag ở2498560 transitions/4632 updates; training-status mới hơn ghi2506752/4648. Bảng ưu tiên status mới hơn. Không có completed episode không có nghĩa chưa cập nhật learner:4096 env cần tới1500 vector steps cho một cohort timeout.

## 4. Những phần không đổi

Git diff từ05beef6c tới HEAD không có thay đổi trong environment/, communication/, pql_system.py hoặc pql_workers.py.
Với từng cặp BASE/PQL cùng thuật toán, các namespace world/sensor/network/energy/reward old-vs-new khớp chính xác.
Do đó không có bằng chứng source reward/CBF/network/sensing đổi đã gây hồi quy trong bốn cặp này.

Chung cho các run:
-6UAV, map1000m,20targets,15obstacles, XYZ, horizon1500, seed44, shared actor+critic.
-Batch8192, replay1M, CPU75%/GPU25%, num_envs4096, warmup131072,8updates/vector collection.
-2×T4, AMP off, fused Adam off, widths256/256, LR3e-4.
-Critic global state565, actor observation198, continuous XYZ+RF4, destination7.
-Masks vẫn tồn tại; silent RF/inactive motion gating và liên kết local actor/global critic vẫn là adapter UAV.

Runtime thay đổi:
| Thành phần |02/11 MASAC|02/11 MATD3|05/11 cả hai|Reference YAML hiện tại, không launcher override|
|---|---|---|---|---|
| critic_chunk_size |1024|1024 nhưng MATD3 không dùng|0|0|
| compile_models |false|true|false|false|
| vectorize_critics |true|true|false|false|
| collector_overlap |true|true|false|true qua dataclass default|
| AMP/fused |off/false|off/false|off/false|off/false|
| PyTorch runtime |2.11.0+cu128|2.11.0+cu128|2.11.0+cu130|phụ thuộc môi trường chạy|
| Python runtime |3.13.15|3.13.15|3.13.16|phụ thuộc môi trường chạy|

Không đổ lỗi MASAC cho việc tắt compile: MASAC cũ được hỏi cũng không compile.
Với shared critic, vectorization flag không bypass phép enumeration shared đang chiếm bộ nhớ.
BASE overlap đổi thời điểm học: path overlap học từ replay đã có trong lúc env step đang chạy; path không overlap thêm collection mới rồi mới học. Đây có thể đổi data chronology/policy lag, không chỉ tốc độ.

## 5. MASAC: thay đổi phương trình và ổn định hóa

| Thành phần |02/11|05/11|Code hiện tại|
|---|---|---|---|
| Critic regression shared |MSE1+MSE2|0.5×(MSE1+MSE2)|giữ05/11|
| Initial alpha |0.2|1|1 ở reference|
| Target discrete entropy |0.9×mean(log valid_count)|0.2×log7≈0.38918|giữ05/11|
| Alpha loss reduction |mean trên6 agents|sum trên6 agents|giữ05/11|
| Khi cả batch của agent chỉ có1 đích |skip/restore alpha_d|vẫn cập nhật theo fixed target|giữ05/11|
| Clamp log-alpha |[-20,5]|bỏ hard clamp|giữ05/11|
| Gradient clipping |global norm10|none|none|
| Gaussian log_std |[-5,2]|[-20,2]|[-20,2]|
| Critic bounded execution |chunk1024+checkpoint khi có grad|full batch, không checkpoint|reference vẫn0|
| Actor hidden normalization |không|không|có tùy chọn, reference bật|

Các thay đổi có thể đổi exploration/entropy/gradient và chất lượng học; không tự chứng minh crash.
Alpha gradient do mean→sum tăng6× khi cùng input, nhưng không được gọi là6×learning rate Adam vì moment normalization.
Hệ số0.5 critic loss cũng không đơn giản là learning rate giảm một nửa dưới Adam.

Fixed target không bỏ mask chọn đích. Khi chỉ1đích hợp lệ, entropy khả dụng=0 nhưng target≈0.389; alpha_d có thể bị đẩy tăng. Đây là điểm cần theo dõi theo valid_count; ở PQL32updates cuối alpha_d≈1.00965, chưa có bằng chứng bùng nổ.
Log_std lower bound mới cho phép distribution hẹp hơn; đó là source-backed SAC choice, không phải bằng chứng hội tụ tốt hơn.

Nguồn code:
- [MASAC shared enumeration](https://github.com/truongdaoanhduy/uav_network/blob/db5345a63a5e5fab5843109dc46a952e49f1814b/uav_network/algorithms/masac.py#L184): unchanged old→new.
- [Critic loss](https://github.com/truongdaoanhduy/uav_network/blob/db5345a63a5e5fab5843109dc46a952e49f1814b/uav_network/algorithms/masac.py#L237).
- [Alpha/entropy update](https://github.com/truongdaoanhduy/uav_network/blob/db5345a63a5e5fab5843109dc46a952e49f1814b/uav_network/algorithms/masac.py#L296).
- [Actor distributions](https://github.com/truongdaoanhduy/uav_network/blob/db5345a63a5e5fab5843109dc46a952e49f1814b/uav_network/algorithms/models.py#L30).
- Old equivalents:05beef6c:masac.py172–189,225,278–307; models.py30.

## 6. MASAC memory: tại sao chunk liên quan thực sự

Shared own-destination expectation vẫn giữ nguyên thuật toán:
N×B×K=6×8192×7=344064 critic rows.
Joint action width=6×(4+7)=66, critic input width=565+66=631.
Không phải chỉ8192 hàng đi qua critic.

| Tensor FP32 |Full call, chunk0|Mỗi call nếu original-batch chunk1024|
|---|---:|---:|
| Alternatives [N,B,K,N,C+K]|86.625MiB|10.828MiB|
| Repeated state [NBK,565]|741.563MiB|92.695MiB|
| Concatenated critic input [NBK,631]|828.188MiB|103.523MiB|
| Một hidden output [NBK,256]|336MiB|42MiB|
| Hai hidden ReLU×hai Q branches|1344MiB|168MiB|

Đây là số byte suy từ shape, không phải đo peak VRAM. Không cộng tất cả như các allocations cùng tồn tại; lifetime/autograd/workspace/allocator khác nhau.
Actor update freeze critic weights nhưng vẫn cần gradient qua critic để học action, nên không thể coi phần critic này là no_grad.
Backward có thể tạo thêm gradient input [344064,631].

[critic_chunks.py6–18](https://github.com/truongdaoanhduy/uav_network/blob/db5345a63a5e5fab5843109dc46a952e49f1814b/uav_network/algorithms/critic_chunks.py#L6) không đổi từ bản cũ:
- chunk0 gọi full expansion trực tiếp.
- chunk1024 chia original batch thành8calls.
- Khi grad bật, dùng torch.utils.checkpoint(use_reentrant=False,preserve_rng_state=False), không giữ cả8 activation graphs.
- Sampling nằm ngoài function critic được checkpoint; critic không có Dropout hay BatchNorm cập nhật statistics.
- Logical batch, đủ7đích và số optimizer step vẫn giữ nguyên.

Replay không đổi nhưng lớn:250k GPU rows≈3.3006GiB,750k CPU rows+metadata≈9.9102GiB. Một sample8192≈110.75MiB. replay_device=cpu không có nghĩa100%CPU khi replay_cpu_fraction=.75.
PQL có thêm model objects; một shared full learner chỉ≈3.94MiB tham số, nhỏ hơn replay/activations. Gradient/moments/snapshots còn thêm bộ nhớ, nhưng không nên quy mọi pressure cho replica models.

## 7. GPU telemetry và giới hạn chẩn đoán crash

system.gpu.*.memoryAllocatedBytes trong các mẫu W&B:
| Run |Mẫu hệ thống|GPU0 max quan sát|GPU1 max quan sát|
|---|---:|---:|---:|
| MASAC BASE cũ|300|3.364GiB|4.429GiB|
| MASAC PQL cũ|300|3.423GiB|4.436GiB|
| MASAC BASE mới|184|3.364GiB|7.999GiB|
| MASAC PQL mới|3|3.423GiB|8.003GiB|
| MASAC replica mới|2|3.307GiB|8.003GiB|

Collector dùng GPU0; learner và GPU portion replay dùng GPU1. Hai T4 không tự gộp thành một VRAM pool. Cả hai GPU có hoạt động trong các run có đủ mẫu, phân vai khác nhau.
Các số là maxima của mẫu thu được, không phải torch.cuda.max_memory_allocated hoặc peak tuyệt đối. Capacity mỗiT4 được ghi≈14.56GiB.
Pressure tăng khoảng1.8× ở GPU learner so với các mẫu cũ; phù hợp source chunk/checkpoint thay đổi. CUDA build và execution cũng khác nên không gọi đây là A/B một biến.
Mẫu8GiB dưới capacity không loại trừ một transient allocation gây OOM sau mẫu cuối; cũng không chứng minh OOM.

Trạng thái cuối ba run MASAC đều chỉ ghi event=progress:
- BASE:2026-11-05T09:35:19.600181Z,2506752 transitions/4648updates.
- PQL:2026-11-05T09:35:15.450544Z,143360/32.
- Replica:2026-11-05T09:35:19.637394Z,126976/0.
Chúng dừng ghi tiến độ trong khoảng4.2giây dù đã chạy các thời lượng khác nhau. Đây là dấu hiệu cần kiểm tra termination ngoài learner; không đủ chứng minh manual kill/platform termination.
Không có output.log/exception traceback OOM trong file W&B lấy được cho các run này. Thread-stacks chủ yếu ghi arm/reset progress watchdog; không phải báo cáo OOM hay bằng chứng deadlock.

Đã thử đọc Kaggle console đúng kernel refs bằng cơ chế credential hiện có. CLI2.2.5 trả “Authentication required to call the Kaggle API”; chưa lấy được console cuối. Không sửa/xóa credentials hay launch job. Vì vậy verdict đúng là **OOM plausible, chưa xác nhận**; không tự thay bằng một nguyên nhân chắc chắn khác.

## 8. MATD3: thay đổi ảnh hưởng học

| Thành phần |02/11|05/11|Current|
|---|---|---|---|
| XYZ output normalization|true|false|reference false|
| Categorical actor gradient|hard argmax + softmax straight-through|soft Gumbel tau1/hardFalse|giữ05/11|
| Target category|hard argmax one-hot|soft Gumbel sample|giữ05/11|
| Behavior category|epsilon-greedy valid destinations|Gumbel-max categorical|giữ05/11|
| Peer actions trong actor update|detached current policies|detached replay actions|giữ05/11|
| Gradient clipping|global10|per-parameter0.5|giữ05/11|
| Independent critic mean correction|chưa có|bù số independent critics|giữ05/11; multiplier1 ở shared runs|
| Hidden LayerNorm|không|không|reference bật, historical YAML có thể không bật|

Các thay đổi được gộp trong giai đoạn c0a935e; không có training attribution tách riêng từng yếu tố.
Serialized epsilon_decay_steps14745600→100000 không phải causal update schedule hiện tại vì epsilon sampler đã không còn được dùng.
Critic chunk1024→0 không hoạt động ở MATD3: source không gọi helper chunk này.
Twin-min Bellman target, policy delay2, Gaussian target smoothing0.2/clip0.5, exploration std0.1, actor -Q1 và tau.005 vẫn giữ core.

Mốc code: matd3.py74(categorical),119(target),143(behavior),208(critic reduction),230–252(replay peers),262(actor loss),285–296(target/delay); optimization.py18(clipping).
[MATD3 current source](https://github.com/truongdaoanhduy/uav_network/blob/db5345a63a5e5fab5843109dc46a952e49f1814b/uav_network/algorithms/matd3.py).

## 9. Cơ chế MATD3 xấu: bằng chứng và attribution

Audit checkpoint thật ngày05/11 được lưu ở docs/methods/MATD3_REGRESSION_AUDIT_2026-11-05.md và evidence/2026-11-05-matd3-saturation-audit.json.
Lần này đối chiếu lại source/config và W&B gradients; không tuyên bố đã tải/probe lại checkpoint trong task mới.

Bằng chứng đã đo:
- New validation-best snapshots được chọn tại3686400 transitions/3476actor updates.
- Raw XYZ≈[+20,-15,-19] trên reset seeds44/1000000044/2000000044.
- Tanh ra[+1,-1,-1]; CPU FP32 scalar XYZ derivatives=0.
- Rollout mới dồn cả6UAV xuống y=0,z=0 từstep30, vẫn kẹt ởstep300.
- W&B actor gradients cuối0; shield intervention99.50%, commanded stall97.65%.
- Old BASE/PQL cuối có actor gradients≈0.074/0.070, delivery59/67.75%.
- Camera footprint ởz0 bằng0, ảnh hưởng trực tiếp khả năng tìm/xác nhận mục tiêu.

Old output normalization chia raw XYZ cho max(1,mean(abs(XYZ))). Với3chiều, mỗi normalized logit có độ lớn≤3, không phải≤1; tanh scalar derivative có bound dương trong exact arithmetic. Điều này không bảo đảm whole-network Jacobian khác0.
Same-weight counterfactual bật phép tính đó đã phục hồi scalar derivative khác0, nhưng chưa phải retraining A/B.
Vì output norm, clipping, sampler, peers, execution và software đồng thời khác, không được viết “100% regression do tắt output norm”.

## 10. Current HEAD khác bad05/11 ở đâu

Learning source của MASAC/MATD3 giữ core05/11, thêm optional hidden LayerNorm:
- MATD3 commit4a5aa22; MASAC/shared-builder a5841ea.
- nn.LayerNorm eps1e-5, affine=true, sau mỗi encoder Linear và trước ReLU; widths256/256.
- Output heads, critic networks, SAC distribution/loss/entropy và MATD3 output_normalization=false không đổi.
- Default actor_layer_norm=false bảo toàn legacy; reference YAML bật true cho cả hai.
- Historical shared YAMLs vẫn có thể omit flag và resolve false. Phải xem resolved config launcher thực tế.
- Current reference YAML mặc định share_actor=false/share_critic=false; old/new cloud runs đều true. Không so current default với old shared rồi gán toàn bộ delta cho LayerNorm.
- Current direct-reference collector_overlap resolve true, nhưng các bad05/11 runs dùngfalse. Cần khóa nó khi làm A/B.
- Checkpoint cũ giữ architecture cũ; LayerNorm cần fresh training, không tự chèn vào old weights.

LayerNorm ở hidden không bound output Linear. ReLU không làm action luôn dương vì sau ReLU còn signed Linear head trước tanh. Không có bằng chứng full20k/GPU rằng variant hiện tại hội tụ.

## 11. BASE/PQL: phần nào thực sự khác

Source pql_system.py/pql_workers.py không đổi giữa old/new/current. Policy/critic workers gọi cùng update_actor/update_critic core.
Với2GPU: collector GPU0, critic GPU1, policy GPU1. BASE cũng tách collector/learner như vậy.
Tất cả cohort đang so có deterministic=true. Code concurrent=async AND NOT deterministic, nên các stage PQL này chạy tuần tự; scheduling_deterministic=1.
Không có căn cứ quy crash cho critic/actor graphs đồng thời hoặc queue tích tụ chỉ vì config ghi mode=async.
BASE overlap là factor khác: timing thêm replay/học có thể đổi sample chronology; PQL vẫn có bộ điều phối/snapshot split riêng. Không khẳng định BASE=PQL từng bit hoặc speedup đúng paper.

## 12. Baseline có nguồn: giữ gì, khai báo gì

| Thành phần |Nguồn đối chiếu|Kết luận|
|---|---|---|
| TD3 deterministic tanh, twin-min, delay/noise|Fujimoto source|giữ được core; original actor không có output norm/LayerNorm|
| MATD3 replay peers|paperEq4 và author trainer|new path có nguồn; old current-peer substitution là variant lịch sử|
| Soft Gumbel và clip từng tensor0.5|pinned author implementation|new path có nguồn; không phải tự nghĩ ra để làm kém|
| SAC0.5MSE, logstd[-20,2]|SB3 SAC source|new choices có nguồn|
| Alpha1, entropy weight0.2, no hard bounds|BenchMARL MASAC config|có nguồn; không phải universal default của mọi Hybrid SAC implementation|
| Simultaneous categorical+continuous SAC|Hybrid SAC Delalleau; Hybrid-MASAC|phải mô tả critic multi-pass/exact expectation và CTDE adapter|
| Shared parameters, agent-ID inputs|study architecture choice|cần khai báo; không gọi literal Ackermann reproduction|
| Masks, RF hard gating, hard replay/soft training|UAV interface|cần khai báo adapter và estimator convention|
| Hidden LayerNorm|standard PyTorch module|architecture variant, không phải mặc định original TD3/SAC|
| Chunk/checkpoint|memory execution|có thể giữ phương trình; không phải RL heuristic mặc định của paper|

Các khác biệt còn ngăn claim copy y hệt toàn bộ author repo:
- Ackermann code thêm1e-3×mean(square(act_pd.flatparam())). Đây là regularizer của **policy distribution parameters/logits**, không phải neural-weight decay. Cả bản cũ và mới của project không thêm term này theo lựa chọn paper-equation objective.
- Ackermann target tau.01; project.005 theo TD3/study. Source Box branch của author dùng Gaussian; continuous deterministic tanh của project theo Fujimoto.
- Author MPE replay giữ soft action vectors; UAV replay re-encode integer destinations thành hard one-hot, còn actor/target dùng relaxed samples.
- BenchMARL std mapping ở config là biased_softplus; project dùng exp(log_std) theo SAC/SB3 style. Không gọi project là nguyên bản toàn bộ BenchMARL.
- Widths/LR/batch/replay/env count/budget là hyperparameters thí nghiệm, không phải chứng nhận “chuẩn” chỉ vì trùng một thư viện.

Reference contract của repo hiện tự cấm chunk, PQL và sharing cho nhãn REFERENCE_BASELINE. Đây là contract của dự án, không phải định lý paper rằng cứ chunk là mất baseline.
Nếu dùng memory-safe shared/PQL variants thì cần khai báo và, khi được yêu cầu implement, điều chỉnh contract/profile có chủ đích. Không vô hiệu guard hoặc âm thầm đổi definition.

## 13. Hướng giữ baseline mới mà không quay lại code cũ

1. Giữ objective/twin-Q/entropy/sampler/peer semantics đã chọn theo nguồn; freeze scenario/reward/CBF.
2. Tách plain actor baseline(actor_layer_norm=false) và declared LayerNorm architecture(true). Giữ sharing giống nhau trong cặp so sánh.
3. Cho MASAC, dùng memory execution profile chunk/checkpoint với logical batch8192, đủ7choices, cùng sampled actions/RNG, cùng full gradient reduction, một optimizer step/update. Không clip/step riêng từng chunk.
4. Giữ Adam/entropy/clipping core mới; không cần khôi phục hard alpha cap, mask-aware target hay output norm cũ chỉ để tiết kiệm VRAM.
5. Pin PyTorch/CUDA/driver/dependency build và source commit. Đo GPU1 peak tensor/reserved memory theo từng phase; system samples không thay được peak tracing.
6. Kiểm chứng GPU bounded forward/backward trước full20k. Hiện sandbox local không cóCUDA, nên task này không đo GPU fullbatch8192 peak hay tái hiện OOM.
7. Sau đó fresh seed44 A/B cùng transitions, validation seeds/budget; log logits/Jacobian/gradients/altitude/stall/CBF/search/delivery. Chỉ đổi một factor trong mỗi test quy nguyên nhân.

Không có yêu cầu rollback hay production launch được thực hiện trong audit này.

## 14. Kiểm chứng được thực hiện trong task này

Lệnh:
`python -m pytest -q tests/test_critic_chunks.py tests/test_reference_baseline_contract.py tests/test_reference_critic_gradients.py tests/test_actor_layer_norm.py --tb=short`

Kết quả: **64 passed in4.77s**. Các test hiện có kiểm tra bounded execution, values/action-gradients, RNG và optimizer-step consistency.
Source-size/SHA guard sau khi tạo tài liệu:2 passed,554 subtests in0.16s; git diff --check sạch.

Bổ sung diagnostic shared đúng6agents/7destinations/4continuous/observation198/state565:
- CPU, batch11, hidden16/16; so chunk0 với chunk4.
- Chạy3logical updates, có và không hidden LayerNorm.
- RNG giống hệt; parameters/metrics trong tolerance3e-5.
- Max final parameter difference đo được0.0 ở cả hai cases; không suy thành guarantee bitwise trênGPU/fullbatch.
- Không sửa source/test/config hay dùng GPU job cho diagnostic này.

Nguồn chứng cứ máy đọc: docs/superpowers/evidence/2026-11-06-four-run-code-audit.json và 2026-11-06-four-run-config-diffs.json.
Raw downloads và scripts trong /tmp/uav-code-audit-20261106 là intermediate, không chứa trong report.
Kết quả full suite1287passed của source trước đó vẫn là historical validation; không được ghi là đã rerun toàn bộ suite trong task này.

## 15. Nguồn chính và code anchors

- Fujimoto TD3: https://github.com/sfujim/TD3/blob/master/TD3.py
- MATD3 paper: https://arxiv.org/pdf/1910.01465
- Pinned author MATD3: https://github.com/JohannesAck/MATD3implementation/blob/fd6c7d0df4fc4effd4e0fa11abdbc7d12d8f01a1/matd3/matd3/trainer/matd3.py
- Author distribution/clipping: cùng revision, matd3/common/distributions.py và tf_util.py.
- SB3 SAC: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/sac.py
- BenchMARL config: https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/algorithm/masac.yaml
- Hybrid SAC: https://arxiv.org/abs/1912.11077
- Hybrid-MASAC: https://proceedings.mlr.press/v270/fu25c.html
- PQL paper: https://proceedings.mlr.press/v202/li23f.html
- PyTorch checkpoint: https://docs.pytorch.org/docs/2.11/checkpoint.html
- CUDA memory: https://docs.pytorch.org/docs/2.11/notes/cuda.html
