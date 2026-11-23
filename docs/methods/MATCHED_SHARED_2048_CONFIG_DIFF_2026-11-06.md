# MASAC/MATD3 shared2048: code đã sửa và toàn bộ config còn khác 02/11

Ngày đối chiếu:06/11/2026 ICT. Bản cũ:05beef6cd0d97d39d51b220f9f7789e915b5bc10.
Bản trước lần sửa này:52e500a65108e04df6f1ed0fd01dc64a51e70e4a.
Source implementation đã push:c792aa07b07d2860e5556e75eac5b64929999a06.
Delivery record ở docs/superpowers/CURRENT_TASK.md.

## 1. Đã chỉnh gì theo yêu cầu mới

Cấu hình hiện tại được chọn rõ ràng là:
`configs/comparison-reference-masac-matd3-xyz-sensing.yaml`.

|Cấu hình chung sau sửa|MASAC|MATD3|
|---|---|---|
|Shared actor|True|True|
|Shared critic|True|True|
|Actor hidden LayerNorm|True|True|
|Critic chunk, original replay rows|2048, thực sự được dùng|2048, thực sự được dùng|
|Hidden widths|256,256|256,256|
|Logical batch|8192|8192|
|Replay / CPU share|1M /75%|1M /75%|
|Environments / warmup transitions|4096 /131072|4096 /131072|
|LR /gamma /tau|3e-4 /.99 /.005|3e-4 /.99 /.005|
|Critic updates/collection|8|8|
|Seed /deterministic|44 /True|44 /True|
|Vectorize /compile /AMP /fused|False /False /off /False|False /False /off /False|

Chỉ đổi ba loại giá trị của canonical YAML: shared actor, shared critic và chunk.
Các hyperparameter còn lại giữ nguyên để người dùng quyết định sau bảng so sánh.
Một shared critic là1 module TwinCritic, bên trong vẫn có2 mạng Q1,Q2.
Shared actor nhận observation198 chiều và one-hot UAV ID6 chiều.

Trước sửa, MATD3 bỏ qua critic_chunk_size; MASAC chỉ chunk Q kỳ vọng.
Sau sửa, cả hai chunk các đường target Q, actor-through-critic và critic regression.
Không chia logical batch thành nhiều lần optimizer.step; loss reduction, clipping,
sampling/noise và lịch target/actor update vẫn giữ nguyên.

2048 là giới hạn original replay rows trước expansion, không phải số rows cuối
vào mọi MLP. MASAC shared mở rộng tối đa2048*6*7=86016 rows/call; MATD3 actor
Q mở rộng2048*6=12288. Vì công thức khác nhau, VRAM hai bên không nhất thiết bằng.
Checkpoint ở đây là recompute activations khi backward, không phải lưu model.
Actor graphs, replay và environment vẫn chiếm bộ nhớ;2048 không bảo đảm hết mọi OOM.

Dataclass defaults và YAML lịch sử không bị sửa. Hai file đề xuất tên memory1024
vẫn là proposal cũ, chưa được chọn. Không dùng chúng làm “current” của báo cáo này.
Các script tạo/chọn matched study đã đồng bộ guard2048; chúng vẫn giữ các trục
shared/independent của ma trận10case cũ, không tự đổi toàn bộ10case thành shared.
Chọn4case shared hoặc dùng canonical trên để so sánh4trường hợp người dùng yêu cầu.
Không submit/chạy lại job cloud trong công việc này.

## 2. Nguồn cấu hình cũ và cách so sánh

|Trường hợp|Run W&B cũ|
|---|---|
|MASAC BASE|[c8d5ce6b](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/c8d5ce6b) —20261102-034013-d522cf|
|MASAC PQL|[2ce9fb9e](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/2ce9fb9e) —20261102-033955-e9729e|
|MATD3 BASE|[db61bd0c](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/db61bd0c) —20261102-031752-e3778d|
|MATD3 PQL|[d51b0350](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/d51b0350) —20261102-031752-c338a0|

Dùng snapshot W&B đã đọc bằng API có xác thực, không suy từ tên run.
Mỗi config mới được resolve từ canonical YAML và chỉ chọn runtime.algorithm cùng
pql.enabled cho trường hợp tương ứng. So toàn bộ key của schema Config:
194 leaf fields cũ,201 leaf fields mới. Không điền default mới vào chỗ cũ thiếu.
Những key không có ở config cũ được ghi “không có field”, kèm hành vi source cũ.

Theo yêu cầu, phần các khác biệt còn lại dưới đây loại runtime.critic_chunk_size.
JSON chứng cứ lưu toàn bộ4cặp config, từng key khác và đánh dấu field của learner
không được chọn. Chênh lệch raw JSON không đồng nghĩa chênh lệch hành vi có hiệu lực.

## 3. MASAC cũ so với MASAC hiện tại: tất cả key còn khác

Các hàng dưới đây áp dụng giống nhau cho BASE và PQL.

|Key|Cũ 02/11|Hiện tại|Tác dụng /điểm cần quyết định|
|---|---|---|---|
|masac.actor_layer_norm|Không có field; actor không LN|True|Chuẩn hóa hidden features; không chặn logits cuối và không bảo đảm tanh hết saturation|
|masac.grad_clip_mode|Không có field; source global clip10|none|Bỏ clipping gradient. grad_clip_norm vẫn10 trong config nhưng không được dùng để clip ở mode none|
|masac.initial_alpha|0.2|1.0|Entropy penalty ban đầu lớn hơn với cùng log-prob; alpha vẫn tự học, không cố định bằng giá trị này|
|masac.target_discrete_entropy_ratio|0.9|0.2|Giảm mục tiêu entropy categorical; policy có thể tập trung lựa chọn sớm hơn|
|masac.mask_aware_discrete_entropy_target|Không có field; source dùng số lựa chọn hợp lệ|False|Chuyển từ target phụ thuộc mask sang0.2*log(7) cố định; trạng thái chỉ có1 lựa chọn có thể không đạt target dương|
|masac.target_discrete_entropy_final_ratio|Không có lịch/final field|0.2|Field mới; hiện không tác động riêng vì anneal_updates=0|
|masac.target_discrete_entropy_anneal_updates|Không có lịch; ratio0.9 cố định|0|Không anneal ở cả hai recipe; đây là field mới, không phải mới bật annealing|
|runtime.vectorize_critics|True|False|Đổi cách thực thi, không tăng hidden width/số mạng; shared Q enumeration vốn đã batched|

Với mask có m lựa chọn hợp lệ, source cũ dùng0.9*log(max(m,1)), có guard giữ alpha_discrete của một UAV khi toàn bộ batch của UAV đó chỉ
có1lựa chọn hợp lệ. Target hiện tại là0.2*log(7).
Đây là khác biệt cấu hình có liên quan trực tiếp exploration, không phải chỉ VRAM.
Các guard/source khác được liệt kê riêng ở mục6.

## 4. MATD3 cũ so với MATD3 hiện tại: tất cả key còn khác

Các hàng dưới đây áp dụng giống nhau cho BASE và PQL.

|Key|Cũ 02/11|Hiện tại|Tác dụng /điểm cần quyết định|
|---|---|---|---|
|matd3.actor_layer_norm|Không có field; actor không LN|True|Chuẩn hóa hidden features, không giới hạn head logits|
|matd3.grad_clip_mode|Không có field; global|per_parameter|Clip từng tensor parameter riêng, thay vì một norm chung cho cả nhóm|
|matd3.grad_clip_norm|10|0.5|Thay cả ngưỡng lẫn quy tắc clipping; không thể gọi đơn giản là nhỏ hơn20lần|
|matd3.output_normalization|True|False|Bỏ phép chia raw XYZ theo mean absolute magnitude trước tanh; logits lớn có thể saturation|
|matd3.epsilon_decay_steps|14745600|100000|Giá trị serialized khác; code hiện tại không dùng epsilon schedule nên chỉnh field này không đổi exploration|
|runtime.vectorize_critics|True|False|Thay loop/vmap execution; shared actor-Q path vẫn batched|
|runtime.compile_models|True|False|Bỏ JIT Inductor ở critic helper và actor encoders; giảm compile overhead, có thể đổi throughput/VRAM, chưa benchmark GPU|

initial_epsilon=1 và final_epsilon=.05 không đổi giá trị, nhưng cũng không còn được
sampler hiện tại sử dụng. Phải sửa source nếu muốn khôi phục epsilon-greedy cũ.
Output normalization cũ tác động XYZ, không RF power. Nó không phải nn.LayerNorm.
Với3 chiều XYZ, scaling cũ bảo đảm mỗi raw motion component có độ lớn<=3,
không bảo đảm logits nằm[-1,1]. Không bật lại cơ chế này trong lần sửa.

## 5. Các khác biệt serialized còn lại và những phần thật sự không đổi

Cả4case còn khác đúng3key quản lý file/ghi nhận; không đổi phương trình học:

|Key|Cũ|Hiện tại|
|---|---|---|
|runtime.output_dir|/kaggle/working/train/masac hoặc /matd3|outputs/reference-baseline-steps1500-t20-o15-learnedpower-report1kb-xyz-sensing|
|tracking.group|uav-05beef6-fair-cohort-20261102|uav-reference-baseline-xyz-20t15o-1500|
|tracking.tags|Tags cohort cũ, có auto-continue/shared và compile status|Tags reference/source-backed/fp32-eager; danh sách đầy đủ ở JSON|

Config luôn serialize cả2nhóm masac và matd3. Vì vậy trong run MASAC còn xuất hiện
5key MATD3 khác:actor_layer_norm,epsilon_decay_steps,grad_clip_mode,
grad_clip_norm,output_normalization. Chúng không được MASAC learner sử dụng.
Trong run MATD3 còn7key MASAC khác:actor_layer_norm,grad_clip_mode,initial_alpha,
mask_aware_discrete_entropy_target,target_discrete_entropy_anneal_updates,
target_discrete_entropy_final_ratio,target_discrete_entropy_ratio.
Chúng không được MATD3 learner sử dụng. Giá trị chính là ở mục3–4, không bỏ sót.

Tổng raw serialized differences sau khi loại chunk:
MASAC BASE/PQL mỗi case16key =8ở mục3 +5nhóm không dùng +3metadata.
MATD3 BASE/PQL mỗi case17key =7ở mục4 +7nhóm không dùng +3metadata.
Ngoài những key này, mọi leaf field trong schema Config khớp snapshot cũ.

Các nhóm world,sensor,network,energy,reward khớp hoàn toàn ở cả4cặp:
6UAV,20target,15obstacle,map1000m,max_steps1500,XYZ,CBF,Bayesian,
cùng reward/network/radio/energy. PQL config cũng khớp từng trường khi so đúng
BASE với BASE và PQL với PQL:async,ratio8,warmup32,pending2,timeout600.

Các phần không đổi:shared/shared,hidden256/256,LR3e-4,gamma.99,tau.005,
B8192,E4096,replay1M,CPU75%,learning_starts131072,updates8,seed44,
20000episodes,deterministicTrue,FP32,AMPoff,fusedFalse,
eval30seed2000000044 và các setting evaluation/checkpoint khác.
MATD3 delay2,target noise.2,noise clip.5,explore noise.1 không đổi.
MASAC continuous entropy target-4 không đổi.

runtime.collector_overlap:old=True và canonical hiện tại=True, nên không phải
khác biệt cũ/current của bảng này. Cloud05/11 và script matched study lại pinFalse.
Đó là nguồn dễ nhầm khi trộn “canonical YAML”, “cloud run đã chạy” và “proposal”.
Deterministic PQL giữ split learners nhưng không chạy hai worker đồng thời;
runtime flag được serialize không có nghĩa pipeline PQL thực sự overlap.
AMP/compile/vectorize không tự động là biện pháp làm reward tốt hơn.

Các key W&B ngoài schema như hardware,pipeline,source_git_commit,timestamp,
kaggle_url và account label là metadata của job, không được nạp vào Config.
Chứng cứ JSON ghi danh sách key ngoài schema; không chép credential.
Đã thấy môi trường cloud cũ PyTorch2.11/cu128/Python3.13.15 và cohort mới
PyTorch2.11/cu130/Python3.13.16. Chưa có job mới để gán phiên bản cloud cho
source vừa sửa; CPU kiểm thử không phải phép so sánh CUDA giữa hai cohort.

## 6. Cùng config chưa đủ tái lập code02/11

Các thay đổi source sau đây vẫn tồn tại; không được “khôi phục” bằng YAML:

|Phần|Code02/11|Code hiện tại|
|---|---|---|
|MASAC log_std bounds|[-5,2]|[-20,2]|
|MASAC shared critic loss|MSE1+MSE2|0.5*(MSE1+MSE2)|
|MASAC alpha loss reduction|Mean qua agents|Sum các objective mỗi agent; không suy là Adam LR tăng6lần|
|MASAC hard alpha clamp|log_alpha clamp[-20,5]|Không có hard clamp|
|MASAC alpha khi toàn batch của UAV chỉ còn1lựa chọn|Skip/restore discrete-alpha update cho UAV đó|Recipe mask_aware=False không dùng guard đó|
|MATD3 behavior destination exploration|Epsilon-greedy trên valid destinations|Categorical sampling qua Gumbel-max|
|MATD3 actor categorical gradient|Hard argmax + softmax straight-through|Soft Gumbel-Softmax tau1|
|MATD3 target categorical action|Hard argmax|Soft Gumbel sample|
|MATD3 peer context trong actor Q|Current-policy peer actions detached|Replay peer actions detached|

LayerNorm đã được duyệt trước lần sửa này và được serialize đúng với checkpoint.
Không lấy các run05/11 chưa có LayerNorm làm kết quả của source mới này.
Shared/LN/hybrid-action adapter được khai báo rõ; không tự nhận byte-for-byte
vanilla upstream. Nhóm entropy riêng MASAC và delay/noise/target actor riêng
MATD3 phản ánh thuật toán, không cần sao chép lẫn nhau để so sánh.

Gradient clipping là một lựa chọn chung vẫn khác giữa2recipe hiện tại:
MASAC none, MATD3 per_parameter.5, theo các nguồn đã khai báo của mỗi recipe.
Người dùng chưa yêu cầu chọn global10 chung trong lần sửa này; không tự áp.
Nếu muốn kiểm soát cả optimizer recipe, cần chạy thêm một variant được khai báo,
ví dụ cả hai global10; không gọi variant đó là nguyên xi mặc định của mọi thư viện.

## 7. Crash và convergence: kết luận có thể rút ra

MASAC memory tăng là giả thuyết có cơ sở: bỏ bounded destination enumeration
đã làm full expansion tới344064 critic rows trong một lần tính. Triển khai này
giảm phạm vi activation critic, nhưng chưa có traceback chứng minh các crash
trước là CUDA OOM. Không thể suy cứ job crash là GPU “bị nổ” hoặc code mới sai.

MASAC BASE mới đã tới4648updates theo training-status mới hơn summary4632;
PQL tới32; replica không ghi update sau126976transitions. Không đúng khi nói
cả3run đều chết ngay trước update đầu tiên. Log gần cuối3run dừng trong4.2giây
nên việc runtime/platform bị dừng cũng cần được kiểm tra khi có console.

MATD3 mới trước LN có bằng chứng raw XYZ rất lớn, tanh gần±1 và actor_grad0;
nó đã chạy xong chứ không cùng dạng crash của MASAC. LN/chunk không phải chứng
minh đã sửa convergence. Bản MATD3 cũ rolling delivery59%/67.75%, còn MASAC
cũ4.5%/15%; không nên gọi cả4bản cũ đều hội tụ thành công.

Để tìm nhân quả training, cố định current shared recipe và chỉ đổi một yếu tố
mỗi lần. Đối với MATD3 ưu tiên raw logits/tanh derivatives cùng exploration/
peer context; MASAC ưu tiên memory trace,alpha/entropy/mask/gradient norms.
Theo cùng transitions và held-out evaluation; cùng episode cap không bảo đảm
cùng data budget. Không tự áp các ablation hoặc tự bật output norm.

## 8. Kiểm chứng của lần sửa

- RED mới:11failed/8passed; phát hiện MATD3 không chunk, MASAC regression chưa
  chunk, saved activations không giảm và canonical còn6actor/critic.
- GREEN focused:41passed cho chunk/reference/critic-gradient tests.
- Bộ mới gồm23tests passed:shared/independent,vectorized/nonvectorized,
  B11/chunk4 có phần cuối3, RNG,gradients,clipping,Adam state,TD3 delay.
- Kiểm tra saved storage của MATD3 actor-Q nhỏ hơn80% full-batch trong test;
  critic weights frozen vẫn truyền input gradients đúng.
-4CPU smoke dùng Warp collector thật:MASAC/MATD3 BASE/PQL,6UAV,
  O198/S565. Giảm widths16,B4,E4,horizon4 và chunk2 để chunk có tác dụng.
  Mỗi run có>=2updates, checkpoint load lại thành1actor/1TwinCritic,weights finite.
- Canonical chunk2048/shared được kiểm tra construction/config riêng.
  Không chạy productionB8192/GPU; không tuyên bố peak VRAM hay convergence.
- Script chuẩn bị/chọn study đã qua5checks sau cập nhật guard2048.
- Full suite cuối:1310passed,103skipped,8warnings,1632subtests passed,
  265.92giây,exit0. Review độc lập không có Critical/Important/Minor findings.
- CURRENT_TASK và completion record ghi actual source commit cùng các giới hạn.

## 9. Tệp kiểm tra và nguồn

- [Toàn bộ4cặp config và key khác](../superpowers/evidence/2026-11-06-shared2048-remaining-config-differences.json).
- [Audit W&B/code trước lần sửa](MASAC_MATD3_CODE_COMPARISON_2026-11-06.md).
- [Nguồn và các adaptation được khai báo](REFERENCE_BASELINE_PROVENANCE.md).
- PyTorch checkpoint: https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/checkpoint.md.
- Current source:algorithms/critic_chunks.py,masac.py,matd3.py,
  tests/test_matched_critic_chunks.py và canonical XYZ YAML.
