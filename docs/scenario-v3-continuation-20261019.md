# Scenario V3: kiểm tra tiếp ngày 19/10/2026

Thời điểm kiểm tra: **08:41 ngày 19/10/2026, UTC+7** (2026-10-19T01:41:21.299503+00:00).
Báo cáo này ghi nhận một mốc kiểm tra; hai run học 128 env chưa hoàn tất.

## Quyết định đã chốt

Giữ scientific baseline tại `c8dba4f80ff3c9f26556f5ea13d14b0a95d89f1c`. Không promote `8b0282af3c6a9bf67774f2eb5b818d6b26290d5c`: bài A/B không chứng minh được cải thiện hiệu năng thực tế.

| Collector 4096 env | Commit | W&B | Transitions/s | Transitions | Updates |
|---|---|---|---:|---:|---:|
| Baseline | c8dba4f | [16045411](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/16045411) | 91.17761 | 57344 | 0 |
| Candidate | 8b0282a | [33000c42](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/33000c42) | 91.48193 | 57344 | 0 |

Candidate nhanh hơn **0.3338%** trong một cặp đo. Chưa có bằng chứng lợi ích lặp lại đáng kể. Đã đối chiếu cấu hình world, network, sensor, energy, reward, MASAC, MATD3, PQL và runtime; chúng khớp nhau.

Cả 14 mốc tiến độ có search, delivery, battery, replay, updates và completed episodes trùng chính xác. Hai file `evaluation/metrics.json` tải trực tiếp từ Kaggle giống nhau từng byte, SHA-256:

`e892d2be5691f531fc4b93e6393ba69af8cd854bb00e10bc648a802f45a93864`

Evaluation collector dùng Warp trên CPU, một episode 1800 bước: tìm 14/35, giao 5/35, kết thúc time limit. Đây là checkpoint **0 updates**, không phải kết quả học và không dùng để kết luận baseline tốt hay kém. Sự trùng khớp này chỉ là evidence bổ sung cho workload đã đo, không thay thế full native/Warp/CUDA parity.

## Hai run đang chạy

| Run | W&B | Kaggle | Trạng thái lúc kiểm tra |
|---|---|---|---|
| MASAC BASE, 128 env | [bedd17bb](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/bedd17bb) | [fwijevwe/uavnet-validate-20261018-233451-5f9212](https://www.kaggle.com/code/fwijevwe/uavnet-validate-20261018-233451-5f9212) | running ở cả hai nơi |
| MATD3 BASE, 128 env | [1e5569af](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/1e5569af) | [iofiewfjw/uavnet-validate-20261018-233552-fd6464](https://www.kaggle.com/code/iofiewfjw/uavnet-validate-20261018-233552-fd6464) | running ở cả hai nơi |

Cả hai chạy commit c8dba4f, seed 44, PQL off, fresh start, batch 256, updates/collection 8, learning starts 131072, time budget 33000 giây. Cường độ sau warmup vẫn là 8 × 256 / 128 = 16 sample uses/transition. Giữ 35 target, 25 obstacle, 1800 bước, report 1 MB, buffer 3 MB và toàn bộ mô hình mạng V3.

**Mốc tiến độ công khai cuối vẫn là step 128, updates 0**, ghi lúc khoảng 06:38–06:39 UTC+7. Đây là số liệu cũ; không được coi là trạng thái learner hiện thời. System telemetry mới hơn cho thấy GPU utilization 100%, VRAM khoảng 6,83 GB trên mỗi job. GPU activity không tự chứng minh simulator vẫn tăng bước.

Đã kiểm tra `OnlineTracker.progress()`: chỉ ghi khi đủ `log_interval_steps=32768`; sau mốc 128, lần định kỳ tiếp theo là 32896. Khoảng log thưa là một nguyên nhân có thể giải thích dashboard chưa đổi. Chưa xác nhận được warmup; gate yêu cầu cả global step ≥ 131072 và updates > 0.

Kaggle CLI chưa trả live console logs cho hai job đang chạy. Không khởi chạy bản trùng, không hủy hoặc sửa cấu hình job hiện tại.

## Đo nút thắt hiện tại

Đã hoàn tất probe chỉ đọc trên **CPU**, tại worktree perf 8b0282a, cấu hình V3 nguyên vẹn ngoại trừ device và số env phục vụ profiling:

| Phép đo | Giá trị |
|---|---:|
| Env / warmup / bước đo | 4 / 4 / 64 |
| Tổng thời gian đo | 75.98713 s |
| Thời gian packet_step | 75.72886 s |
| Tỷ lệ packet_step | 99.6601% |
| CCA checks tích lũy tại bước 68 | 41091656 |
| Frame count lớn nhất tại các mốc lấy mẫu | 7 |
| Frame capacity / retention | 1632 / 8384 µs |

Đây không phải CUDA benchmark. Evidence xác nhận packet kernel chiếm gần toàn bộ thời gian của probe; chưa phân bổ được chi phí riêng từng hàm CCA/PHY/event heap trên GPU.

Đọc code cho thấy LISTEN tiếp tục kiểm tra sau 1 µs khi kênh rỗi; IDLE kiểm tra lại theo slot khi bận. Lượng polling lớn là giả thuyết ưu tiên để profiling tiếp. Không kết luận full retained-frame scan là thủ phạm duy nhất: probe lấy mẫu chỉ thấy tối đa 7 retained frames. Không thay polling bằng khoảng thời gian lớn hơn nếu chưa chứng minh giữ thứ tự event, thời điểm carrier sense, retry, collision, propagation delay và telemetry.

**Lưu ý môi trường local:** khi chạy helper nằm trong `/tmp`, editable install của `.venv` có thể import checkout main cũ. Lần probe đầu dừng ở bước đọc config và đã bị loại. Lần thành công dùng:

`PYTHONPATH=/home/aduy/Documents/NCKH/uav_network-cpu-flow-20261014`

Đã xác nhận `uav_network.configuration.__file__` thuộc đúng worktree. Các script kiểm tra tiếp cần chọn nguồn import rõ ràng.

## Phần còn chờ kết quả

1. Poll bedd17bb và 1e5569af, xác nhận đã qua warmup bằng metrics mới.
2. Khi Kaggle hoàn tất, tải evaluation/metrics.json, summary.json, diagnostics.jsonl, cohorts.jsonl cùng evaluation log; kiểm tra stop reason và số learner updates thực.
3. Đối chiếu MASAC128/MATD3128 với checkpoint cũ và oracle 35/35: delivery, custody, receiver-eligible/report-bearing, progress-peer/report-bearing, relay toward/away GCS, below-required-power, propulsion/radio energy và shield correction.
4. Phân biệt undertraining, shared implementation issue và khác biệt thuật toán trước khi sửa reward/scenario.
5. Chỉ sau short gate đủ evidence mới cân nhắc full25k/PQL. Mọi sửa network phải có regression và native/Warp/CUDA functional parity.

Chỉ bổ sung evidence và tài liệu ở nhánh audit. Simulator, baseline thuật toán, reward và các job đang chạy chưa thay đổi.

Dữ liệu máy đọc được: [scenario-v3-continuation-20261019.json](benchmarks/scenario-v3-continuation-20261019.json).
