# Đọc kết quả và điều tra training trên W&B

Personal Workspace: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/workspace?nw=nwusertruongdaoanhduy

Mục tiêu là thấy kết quả nhiệm vụ, tiến độ và bằng chứng giải thích vấn đề.
Không có một ngưỡng loss chung để kết luận MASAC/MATD3 học tốt. Log và cảnh báo
chỉ ra triệu chứng; cần đối chiếu checkpoint, seed, batch hoặc traceback để xác
minh nguyên nhân. Chi tiết mẫu số và seed bank ở [diagnostics-v1.md](diagnostics-v1.md).

## Bảng Runs trong Personal Workspace: 17 cột, chỉ ghim 4 cột đầu

| Cột | Ý nghĩa |
|---|---|
| Name, State | Tên nhận biết run và trạng thái W&B thực tế. Giữ cả running, finished, failed, crashed. |
| Created | Thời điểm W&B tạo run, không phải thời điểm submit job Kaggle. |
| Runtime | Thời gian của run/phiên này, không cộng tất cả phiên trong lineage. |
| Stage, eval episodes completed | Phân biệt train/validate/visualize; số tập của kết quả eval đang hiển thị (không cộng dồn các checkpoint). BASE/PQL và thuật toán đã có trong Name; ẩn Mode/Algorithm để tránh lặp. |
| Account, Seed | Xác định chủ job và seed để đối chiếu. |
| Global step, completed training episodes | Tiến độ thật; một episode đang chạy chưa được tính là hoàn tất. |
| Cohort delivery, cohort success | Kết quả episode đã kết thúc trong cửa sổ transition gần nhất. |
| Eval delivery, eval success | Điểm tổng hợp đánh giá CPU (`eval/*`), không lấy episode CPU cuối cùng thay cho trung bình. |
| Episode budget complete | Đã đủ tổng episode hay chỉ kết thúc một phiên. |
| Stop reason, failure type | Vì sao phiên dừng; loại exception nếu code kịp ghi nhận. |

Bảng không lọc theo trạng thái, ID hoặc nhóm; sắp xếp Created mới nhất trước.
Ô trống là chưa có bằng chứng, không phải 0. Mọi run vẫn giữ nguyên history,
config và summary. Các cột phụ xem trong trang từng run khi cần.
`finished` chỉ xác nhận run kết thúc bình thường: cần xem `episode_budget_complete`
và `stop_reason` trước khi kết luận đã train đủ 20.000 tập.

## 27 đồ thị và một hướng dẫn đọc

| Nhóm | Số plot | Cách dùng |
|---|---:|---|
| Mission quality — completed outcomes | 4 | Delivery, success, return, search trên các episode đã kết thúc. |
| Training progress and speed | 3 | Episode hoàn tất, joint transitions/s, số lần dùng mẫu trên transition mới. |
| Periodic fixed-policy evaluation | 2 | Delivery/success của checkpoint trên seed đánh giá cố định; mặc định thu gọn. |
| Final CPU evaluation | 2 | Delivery/success của từng episode đánh giá cuối. |
| Why delivery is low | 4 | Packet PDR, im lặng dù có receiver hợp lệ, chuyển động bị chặn, J/delivery. |
| Numerical diagnostics | 5 | Actor/critic loss, TD error, Q, critic gradient; mặc định thu gọn. |
| Resources | 4 | GPU0/1 utilization, GPU0/1 memory allocated %, CPU %, RAM %. |
| Recovery and upload cost | 2 | Thời gian từ upload full-state được xác nhận và thời gian snapshot + upload; thu gọn. |

Mỗi plot hiển thị tối đa 12 run riêng, không gộp trung bình các thuật toán. Chọn
các run cùng scenario, replay, precision, seed và budget để so sánh. Bảng vẫn
chứa mọi run. Giới hạn này giảm đường chồng lên nhau; không xóa dữ liệu run khác.

Trục X khác nhau theo bằng chứng:

- Training và cohort: `global_step` (joint transition), không nhầm với số episode.
- Mini-eval: `evaluation/checkpoint_global_step`, tức bước của policy được chụp;
  evaluator hoàn tất trễ không làm đổi checkpoint nguồn.
- Từng episode CPU cuối: `evaluation/episodes_completed`.
- CPU/GPU/RAM: `_runtime` theo giây, vì SDK thu system metrics riêng với learner.

Canonical `configs/comparison-20k-hybrid75-1kb.yaml` đang **tắt mini-eval định kỳ**.
Không tự bật lịch đánh giá hoặc đổi scientific config trong thay đổi dashboard.
Nhóm periodic trống là hợp lệ. Không dùng training return để khẳng định khả năng
tổng quát hóa khi chưa có đánh giá độc lập. Điểm CPU cuối và protocol có trong
`metrics.json`; `eval/mean_delivery_rate_percent` và `eval/success_rate` là điểm
tổng hợp hiển thị trên bảng.

## Đánh giá policy và tìm hướng cải thiện

Ưu tiên delivery tới GCS và success trên cùng protocol, sau đó xem search và
return. So nhiều seed đánh giá tương đương; một training seed hoặc một episode
đẹp chưa đủ kết luận thuật toán hơn nhau. Loss giảm hoặc GPU sử dụng cao không
đồng nghĩa policy tốt hơn.

| Quan sát | Bằng chứng cần xem tiếp | Chỉ kết luận sau khi đối chiếu |
|---|---|---|
| Search cao, delivery thấp | Packet PDR, eligible-silent, reports/expiry/drop, relay, energy và action/reward chi tiết | Có thể liên quan định tuyến, không truyền, tắc hàng đợi hoặc hết pin; search đơn lẻ không xác định nguyên nhân. |
| Delivery/success giảm theo checkpoint | Cùng seed bank, cùng budget, cohort n/partial, return và numerical diagnostics | Phân biệt policy suy giảm với khác seed, cohort ít mẫu hoặc episode còn dở. |
| Reward tăng mà delivery không tăng | Thành phần reward, blocked-motion, discovery/delivery, energy/penalties | Kiểm tra hành vi đang được thưởng và mục tiêu nhiệm vụ; chưa tự ý chỉnh reward. |
| TD/Q/gradient tăng mạnh hoặc nonfinite | `failure.json`, batch/state lỗi, clipping, entropy/alpha và learner counters | Xác minh lỗi số học, scale hoặc update trước khi đổi learning rate/precision. |
| TPS giảm, outcome chưa đổi | CPU/GPU/RAM theo runtime, replay/sample-use, mốc snapshot/upload, thời gian eval | Phân biệt nghẽn compute, lấy mẫu, I/O và đánh giá; throughput chỉ đo tốc độ. |
| Episode hoàn tất còn 0 nhưng step/update tăng | Phase, episode step, horizon và completed counter | Với nhiều env chạy song song có thể chưa env nào kết thúc; không gọi là đứng train chỉ vì episode=0. |
| W&B mất heartbeat | Console cuối, status cuối, checkpoint đã xác nhận, trạng thái/log Kaggle | Mất heartbeat không chứng minh OOM, ban hoặc compute đã chết. |

Completed-outcome windows có `cohort/episodes`, `cohort/partial`,
`cohort/source_start_step`, `cohort/source_end_step`. Xem cỡ mẫu và cửa sổ trước
khi so sánh; live-world search/battery phụ thuộc tuổi episode đang chạy.
Packet PDR đo gói tin, khác tỷ lệ mục tiêu báo về GCS. J/delivery chỉ so có ý nghĩa
khi mức delivery/success tương đương. BASE sample-use đo phiên hiện tại; PQL dùng
replay counters có thể đã restore, nên cần xem nguồn counter khi resume.
Cảnh báo no-search, report-loss, blocked-motion, return-drop là heuristic theo
scenario; không phải mọi cảnh báo đều là bug.

## Logs và Files

| File hoặc nơi xem | Nội dung |
|---|---|
| W&B Logs | stdout/stderr và các dòng JSON `training_event`: thời gian UTC, phase, run/source/seed, tiến độ và evidence chọn lọc. |
| `training-events.jsonl` | Lịch sử start/progress/checkpoint/recovery/evaluation/failure/close, giữ context liên tiếp. |
| `training-status.json` | Bản trạng thái nhỏ mới nhất, ghi bằng atomic replace; đăng ký upload live. |
| `metrics-guide.md` | Hướng dẫn ý nghĩa metric, nguồn mẫu và trục X đi cùng run. |
| `diagnostics.jsonl`, `cohorts.jsonl` | Cảnh báo/đánh giá và các cửa sổ outcome với nguồn, cỡ mẫu, partial. |
| `failure.json` | Exception chain, traceback đã redaction, loại lỗi bảo thủ, tài nguyên RAM/CUDA/đĩa, signal thực nhận và context/seed/metrics cuối có sẵn; không tự xác nhận nguyên nhân gốc. |
| `thread-stacks.log` | Stack mọi thread sau 300s không có marker tiến độ mới; lặp lại, không tự kill. Có cho BASE/PQL và CPU, kể cả không bật W&B. Native fatal traceback đi vào stderr khi subprocess được chạy với `PYTHONFAULTHANDLER=1`. |
| `failed_batch.pt`, `failure_state.pt` | Với lỗi learner có hỗ trợ: batch thật và state sau lỗi để tái hiện. Đây là evidence chẩn đoán, không phải full checkpoint phục hồi. |
| `resolved_config.yaml`, CPU `metrics.json` | Cấu hình/protocol, checkpoint nguồn, aggregate CPU và kết quả publication tương ứng. |

Journal chỉ lấy scalar chọn lọc và identity cho phép; không dump tensor, replay,
credential hoặc cả config vào từng dòng. NaN/Inf ghi rõ bằng chuỗi, dữ liệu chưa
biết giữ null. Ghi ở nhịp progress sẵn có và khi chuyển giai đoạn, không ghi mỗi
optimizer update. Lỗi ghi console, append file, atomic status hoặc upload tùy
chọn được cô lập để không biến thành lỗi learner.

`checkpoint_snapshot` đánh dấu đang chụp state và `checkpoint_upload` đánh dấu
đang truyền. Đồng hồ `recovery/seconds_since_confirmed_upload` chỉ bắt đầu sau
khi full-state upload được xác nhận; policy-only không reset đồng hồ. Đây không
phải tuổi của state lúc được chụp, cũng không phải bảo đảm full checkpoint của
phiên trước đã được kiểm tra trong phiên mới.

Status nhỏ đăng ký `live` và yêu cầu refresh `now` sau các mốc chuyển giai đoạn;
progress yêu cầu refresh tối đa một lần/phút ở nhịp progress kế tiếp. Không dựa
riêng vào watcher: phép thử online SDK 0.30.0 bỏ lỡ cập nhật atomic replace trong
110 giây, dù bản cuối được upload khi finish. Đây là yêu cầu upload best-effort,
không phải cam kết độ trễ mạng một phút. Guide dùng `now`; journal/diagnostics tăng dần đăng ký
`end`, rồi yêu cầu flush `now` lúc caught failure/close. Cohorts giữ live policy.
W&B Logs có các event trong khi chạy. Hard kill/mất mạng có thể cắt phần chưa
đồng bộ: không hứa có traceback cuối hoặc mọi event trước crash. Khi W&B chưa
khởi tạo được, tìm file trong output Kaggle/local. Full-state recovery và
supervisor ngoài sandbox được mô tả ở [github-actions-full-flow.md](github-actions-full-flow.md).

Jobs đã submit dùng SHA cố định. Dashboard mới đọc được metric lịch sử hiện có;
log/metric mới chỉ xuất hiện ở job dùng source mới. Không tạo bù evidence cho
run cũ và không sửa lịch sử failed/crashed thành finished.

## Cập nhật lại view và xem chi tiết khi cần

```bash
python -m uav_network dashboard --config configs/comparison-20k-hybrid75-1kb.yaml --profile core
```

Profile `diagnostics` có các plot action/reward/entropy/alpha/warning chuyên sâu.
Chỉ dùng nó ở một saved view riêng khi điều tra; chạy vào view chính sẽ thay bố
cục của view đó. Config/history/file chi tiết vẫn có ở từng run.

Tài liệu nguồn: [W&B logging](https://docs.wandb.ai/models/track/log),
[console logs](https://docs.wandb.ai/models/app/console-logs),
[SAC diagnostics](https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/sac/sac.html).

## Kiểm tra độ bao phủ và các tình trạng khác

Xem [ma trận bằng chứng và giới hạn](diagnostic-coverage.md) để chọn file cần đọc,
bằng chứng phải lấy thêm và bước kiểm tra tiếp. Không có log nào tự giải thích
mọi nguyên nhân. Nếu không có traceback, không được gán OOM/ban chỉ từ crashed.

Hai plot bổ sung là **cohort/episodes** (độ lớn mẫu của outcome) và
**storage/disk_free_bytes** (đĩa tại nhịp log training). Plot đĩa dùng global_step;
CPU/GPU/RAM của SDK dùng runtime. Actor loss vẫn có trong metrics và profile
`diagnostic`, nhường chỗ trong giao diện chính cho bằng chứng dễ dùng hơn.

Journal giữ warning/replay/PQL/reward/action/safety/report có sẵn, source step
của từng nhóm và resource snapshot best-effort mỗi 60s ở nhịp event; xem tuổi mẫu.
Failure lấy snapshot mới. CPU có tiến độ từng episode/throttled 30s, checkpoint,
seed đầu/cuối, scenario hash, deterministic và overlap khi protocol có cung cấp.
Không sửa lịch mini-eval đang tắt và không tự đổi policy/scenario để tăng điểm.

Chỉ jobs dùng commit mới có các log mới. Phiên đã chạy với commit cũ không thể
nhận thêm code; dữ liệu thiếu của run lịch sử giữ nguyên, không bịa lại.


## Handled AMP overflow evidence

An AMP skip is not automatically a model failure. Learner scalar telemetry uses null for nonfinite numbers plus `<metric>_nonfinite` and `<metric>_nonfinite_kind`; finite gradient norms reset their flags. Episode/history/summary and W&B scalar channels share this conversion. Genuine nonfinite learner losses and FP32 gradient errors still fail through the existing guards.

BASE and PQL learners attempt one snapshot per critic/actor group at the first AMP skip. `overflow-<role>-<group>.pt` is diagnostic evidence, not a training-resume checkpoint. It stores the original batch, model, owned unscaled pre-clipping gradient tensors and their statistics, optimizer/scaler state, update counters and pre-forward Torch RNG. The adjacent JSON names the source/device, checks input finiteness and distinguishes `nonfinite_gradients` from `norm_reduction_overflow`. Capture/upload failures are optional and bounded; they cannot prevent scaler backoff or stop healthy training.

Replay a saved snapshot separately:

```bash
python -m scripts.replay_overflow snapshot.pt --device cuda:1 --output comparison.json
```

Replay constructs independent FP32/FP16 learners from the same saved values/RNG. It records the first observed nonfinite parameter gradient, per-parameter gradient statistics, finite losses/model state and AMP skip counts. FP32 bypasses AMP counters, so its `applied` field is null. Replay is eager; it does not claim to reproduce compiled kernels or asynchronous RNG interleaving. The first nonfinite parameter observed by a hook is not proof of the first arithmetic operation that overflowed. Diagnostic snapshots contain no environment/replay history sufficient for exact training resume.

For future canonical sessions, the first full recovery backup is due after min(effective interval, 1800 seconds); later attempts retain the bounded session spacing. With 10h/four attempts and interval=1h, first due is 30 minutes and later attempts are spaced 2.5h after the prior attempt finishes. Successful upload confirmation can occur later than the due time. Existing source-pinned jobs retain their original timing and code.
