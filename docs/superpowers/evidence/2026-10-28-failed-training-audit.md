# Chẩn đoán ba training run fail ngày 28/10/2026

Kiểm tra trực tiếp W&B và Kaggle lúc **12:48:44–12:48:46 ngày 28/10/2026, giờ Việt Nam** (05:48 UTC). Repo kiểm tra ở `3d3a829286eda353da449589c0cf4cbb64c70415`; ba run lỗi pin runtime `1aa095f29b8505bbf43ca42033c51ea96d144dd5`.

## Kết luận

Cả ba run bị dừng bởi **lỗi serialize telemetry chứa `inf`**, tại `json.dumps(record, allow_nan=False)`. Đây là nguyên nhân trực tiếp đã có traceback và đã tái hiện bằng metric lưu trong failure report. Chưa có bằng chứng quy lỗi crash cho replay HYBRID CPU75/GPU25 hoặc nhánh action continuous/categorical.

Một vấn đề riêng khiến supervisor không tiếp tục được: **chưa có full-state checkpoint đã xác minh**. Lịch hiện tại đẩy backup đầu đến 150 phút, trong khi ba run fail sau khoảng 65–87 phút tính từ khi W&B khởi tạo. Policy checkpoints vẫn có, nhưng không chứa replay/optimizer/RNG/environment để exact resume.

## Bằng chứng từng run

| Run | W&B / Kaggle | Thời điểm lỗi (VN) | Episode đã ghi | Metric làm JSON lỗi |
| --- | --- | --- | ---: | --- |
| [MASAC PQL 03df03cc](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/03df03cc) | failed / ERROR | 11:57:38 | 2.720 | critic_grad_norm = inf |
| [MATD3 BASE 3befcfb8](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/3befcfb8) | failed / ERROR | 11:38:41 | 7.620 | actor_grad_norm = inf |
| [MATD3 PQL 4f8f7d7d](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/4f8f7d7d) | failed / ERROR | 11:39:32 | 7.861 | critic_grad_norm = inf |

Traceback BASE trỏ đến `uav_network/experiments/training.py:254`; hai run PQL trỏ đến `uav_network/experiments/pql_training.py:152`. Thông báo giống nhau:

```text
ValueError: Out of range float values are not JSON compliant: inf
```

Counter `episodes` trong exception context cao hơn số episode đã ghi một đơn vị, vì code tăng counter trước khi serialize episode kế tiếp. Bảng trên dùng summary/status đã ghi, không gọi episode lỗi là episode hoàn tất.

Cả ba job-failures receipt trên Kaggle đều ghi learner exit code 1. Log [GitHub supervisor 40638183428](https://github.com/truongdaoanhduy/uav_network/actions/runs/40638183428) ghi cả ba account 02–04 là `recovery_blocked / RecoveryUnavailable / checkpoint_missing`.

## Tại sao gradient inf chưa chứng minh model hoặc hybrid sai

Cả ba chạy AMP FP16. `MixedPrecision._step` kiểm tra loss hữu hạn, unscale trước clipping, bỏ qua update khi norm không hữu hạn, cập nhật GradScaler và trả chính norm đó làm metric. Theo [PyTorch 2.11 AMP](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html), bỏ qua optimizer step có gradient inf/NaN là cơ chế bảo vệ bình thường.

Snapshot trước lỗi vẫn có critic loss hữu hạn: MASAC PQL 108,63; MATD3 BASE 225,98; MATD3 PQL 236,02. Các counter AMP ghi nhận lần lượt critic skipped 9, actor skipped 8, critic skipped 15. Riêng MATD3 BASE kiểm tra model lúc exception và lưu `nonfinite_parameter_names=[]`. Hai run PQL không có trường kiểm tra tham số tương đương trong failure report, nên không suy rộng bằng chứng model hữu hạn cho hai run đó.

Chuỗi đã xác minh ở mức code và dữ liệu: norm không hữu hạn được cơ chế AMP xử lý -> metric vẫn chứa float inf -> episode record nhúng `learner=latest` -> strict JSON từ chối inf -> exception thoát vòng train. [Python 3.13 JSON](https://docs.python.org/3.13/library/json.html) quy định `allow_nan=False` phải raise ValueError cho NaN/Infinity.

Chưa tái hiện toàn bộ chuỗi phép toán CUDA sinh ra norm inf của từng run, nên chưa xác định phép toán hoặc tensor đầu tiên gây overflow. Không cần giả định hybrid sai để giải thích exception này. Việc BASE lẫn PQL đều lỗi cũng không ủng hộ kết luận lỗi chỉ nằm ở async PQL. MATD3 BASE compile ON, MATD3 PQL compile OFF nhưng cùng lỗi, nên tắt compile không phải bản sửa cho serializer.

Các resource snapshot lúc lỗi còn khoảng 19,65–20,01 GB RAM khả dụng, 20,21–20,66 GB đĩa trống; GPU learner reserved khoảng 4,11–5,05 GB trên T4 14,56 GiB. Traceback là ValueError, không phải CUDA OOM hay platform kill. Đây là bằng chứng về lần crash này, không phải cam kết tài nguyên luôn đủ về sau.

## Vì sao có checkpoint nhưng vẫn checkpoint_missing

`PeriodicRecovery.__init__` dùng:

```python
interval = max(resumable_checkpoint_interval_seconds,
               max_train_seconds / resumable_checkpoint_max_uploads)
next_due = started + interval
```

Với config thực tế: `max(3600, 36000 / 4) = 9000 giây = 150 phút`.

Mốc bắt đầu scheduler sớm hơn W&B init một chút, nhưng cả ba exception đều xảy ra trước lần backup đầu. Thực tế cả ba summary ghi `recovery/full_checkpoint_available=0`; các model artifact đã liệt kê đều có `resumable=false`. Chỉ giảm `resumable_checkpoint_interval_seconds` sẽ vẫn bị công thức max đẩy về 9000 giây.

Supervisor đang tuân thủ điều kiện provenance/full state: policy weights hoặc `failure_state.pt` không đủ để gọi là exact resume. Giữ lại chúng để phân tích hoặc đánh giá; warm start từ policy phải được khai báo thành run mới, không nối giả tiến độ cũ.

## Phương án sửa đề xuất, theo thứ tự

1. **Sửa telemetry ở đường ghi episode chung cho BASE và PQL.** Tận dụng `json_safe` sẵn trong `experiments/evidence.py` cho riêng snapshot learner metrics, hoặc mapping metric không hữu hạn sang null kèm nhãn inf/NaN. Giữ các metric hữu hạn, reward và outcome nguyên vẹn. Dùng cùng quy ước khi replay history, xuất summary và log W&B; chuỗi inf/NaN chỉ dành cho evidence, scalar chart dùng giá trị thiếu kèm cờ nonfinite. Chưa áp dụng thay đổi này vào production.
2. **Giữ guard số học.** Vẫn bỏ qua AMP update không hợp lệ và back off loss scale; tiếp tục fail rõ khi forward loss/model parameters thực sự không hữu hạn. Không sửa bằng cách bật allow_nan, nuốt mọi ValueError, thay inf thành số 0 hoặc sửa tensor học. Nếu overflow lặp dày sau khi sửa telemetry, đo skip rate và phép toán liên quan, rồi làm canary FP32 có kiểm soát; chưa có cơ sở đổi learning rate/reward/hybrid.
3. **Đưa backup đầy đủ đầu tiên lên sớm.** Tách thời điểm first backup (đề xuất 15–30 phút, đo thời gian upload trước khi chốt) khỏi cách giãn các lần sau; giữ giới hạn số upload và staging một bản để phù hợp đĩa Kaggle. Xác minh hash, replay, optimizer, scaler, RNG, environment và con trỏ artifact trước khi quảng bá resumable. Không chỉ thay con số interval vì công thức hiện tại sẽ vô hiệu hóa thay đổi đó.
4. **Thêm regression ở ranh giới đang thiếu.** Ép actor/critic norm inf trong MASAC/MATD3 BASE/PQL; xác minh skip giữ nguyên tham số và Adam state, episode JSON/summary/history/W&B vẫn hợp lệ, finite scientific metrics không đổi. Thử resume với record đã serialize. Test riêng lịch backup đầu đến trước 1 giờ và upload lỗi không mất pointer trước. Sau RED -> minimal fix -> GREEN, chạy regression và full suite; thêm canary CUDA có kiểm soát trước rollout.
5. **Xử lý lineage sau verification.** Ba run cũ chưa có checkpoint đủ exact resume theo bằng chứng hiện tại. Nếu chọn chạy lại, dùng lineage mới và ghi rõ bắt đầu mới hoặc warm start, giữ đúng account và config khoa học; không tự tạo duplicate hoặc hủy các run đang khỏe.

## Đã kiểm tra trong lượt này

- Tái hiện JSON ValueError cho cả ba bộ metric đã lưu; `json_safe` trên riêng learner metrics làm strict JSON serialize thành công, giữ nguyên mọi metric hữu hạn. Đây là kiểm tra giải pháp ở cấp serializer, chưa phải bản vá end-to-end.
- Chạy `python -m pytest tests/test_precision.py tests/test_recovery_checkpoints.py -k 'overflow or nonfinite or wall_clock_spreads or recovery_settings' -q`: **7 passed, 40 deselected in 7.29s**. Các test này xác minh guard AMP trên CPU và lịch backup đang có; không chứng minh production đã sửa.
- Diff runtime/config liên quan giữa source các run và HEAD đang kiểm tra rỗng.
- Fresh W&B + Kaggle lúc 12:48 VN: MASAC BASE E4096 `4f98f2c9`, MASAC BASE E2048 `a04dbdef`, MASAC PQL E2048 `8db188ea` vẫn running, heartbeat mới và tiến độ tăng. Cả ba chưa có full checkpoint đã xác nhận tại snapshot này. Hai run E2048 vẫn dùng serializer tương tự; giảm batch/env không loại bỏ bug này.
- Không sửa production code/config; không thay W&B state/metadata; không submit, cancel hoặc relaunch training. Lượt này hoàn tất chẩn đoán và đề xuất theo yêu cầu. Evidence có cấu trúc: [2026-10-28-failed-training-audit.json](2026-10-28-failed-training-audit.json).

## Follow-up: nguồn norm inf — kiểm tra 2026-10-28T06:02Z

Đọc thêm history cuối của cả ba run và download failure_state.pt của MATD3 BASE. Snapshot có 240 tensor model, tất cả hữu hạn, max absolute parameter 2.122197389602661. Không có batch đúng lúc overflow trong các W&B files đã liệt kê; code chỉ dump batch khi learner raise, trong khi AMP skip không raise và exception xảy ra sau đó tại JSON.

History + failure context: MASAC PQL critic scale 8192 -> 4096, skipped 8 -> 9; MATD3 BASE actor scale 65536 -> 32768 -> 16384, skipped 6 -> 7 -> 8; MATD3 PQL critic vừa tăng scale 8192 -> 16384 rồi tại failure giảm 8192, skipped 14 -> 15. Đây là bằng chứng mạnh phù hợp AMP overflow/backoff, nhưng code cũng backoff khi FP32 norm reduction overflow: chưa phân biệt được hai nhánh vì thiếu raw gradients và branch marker. Không khẳng định đã tìm phép toán đầu tiên.

models.py dùng Linear/ReLU, không LayerNorm/BatchNorm; GPU observations/state chuẩn hóa vị trí/map_size, vận tốc/max_speed, pin/capacity. Đây là kiểm tra source, không chứng minh batch lỗi hữu hạn hay phân phối ổn. Không có cơ sở quy lỗi thiếu LayerNorm.

Bước chẩn đoán cần thêm: tại nhánh AMP skip đầu tiên, lưu batch và model trước update, scaler state trước/sau, RNG, dtype, finite/min/max từng input; tách raw gradient nonfinite trước clipping khỏi norm reduction overflow và ghi layer đầu tiên. Replay cùng batch/model/RNG FP32 đối chiếu FP16. Chỉ nếu FP32 cũng lỗi mới truy tiếp dữ liệu/target/model, không mặc định thêm normalization. Minh họa CPU độc lập: float16(3)*32768=inf, float16(3)*16384=49152; không phải replay batch lịch sử.
