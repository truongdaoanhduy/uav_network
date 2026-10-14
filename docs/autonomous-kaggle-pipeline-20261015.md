# Kaggle GPU training → CPU visualization khi tắt máy

Run được gửi ngày 15/10/2026 (giờ Việt Nam), dùng source cố định
`0ceb7f1779aee618fe8c59a23e7b045b02e9fe42`.

- GPU parent: https://www.kaggle.com/code/haibro1234/uavnet-train-20261014-181508-908e2d
- W&B MASAC: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/ea73429e
- W&B MATD3: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/501d6d1a
- CPU child dự kiến: https://www.kaggle.com/code/haibro1234/uavnet-train-20261014-181508-908e2d-cpu

CPU child chưa tồn tại khi parent mới bắt đầu. Parent tự tạo nó sau khi các
learner đã kết thúc và upload checkpoint thành công. Trạng thái hiện tại
phải xem trên Kaggle/W&B; các URL ở đây không có nghĩa run đã hoàn tất.

## Cấu hình đã gửi

`configs/motion-pql.yaml`: 4.096 env, seed 44, batch 8.192,
replay 500.000, tối đa 4.096 episode cho mỗi thuật toán. PQL dùng
GPU0 để thu thập, GPU1 để learner/replay/policy; MASAC rồi MATD3
dùng hai GPU lần lượt. Kaggle parent có timeout 39.600 giây (11 giờ),
runner dành thời gian cho checkpoint/upload/handoff; chạm giới hạn thời gian
có thể dừng trước episode budget và ghi rõ `stop_reason`.

CPU evaluation: 30 scenario từ seed 2.000.000.044, cùng checkpoint/backend
đã lưu; PNG, MP4, metrics và paired comparison. CPU child có timeout
3.600 giây, `enable_gpu=false`, `enable_tpu=false`, không truyền accelerator.

## Vì sao tắt máy vẫn chạy

Lệnh `kaggle kernels push` đã gửi saved script lên Kaggle. Máy người dùng
chỉ dùng để gửi job và đọc trạng thái. Trên server, GPU runner upload model
artifact với version cố định, gọi Kaggle API gửi CPU child rồi thoát.
CPU child tải `checkpoint.pt` từ W&B và tự chạy evaluate/render.
Không có local poller, cron trên laptop hay agent ChatGPT giữ vai trò handoff.

W&B training hiện Running và `global_step` tăng là dấu hiệu learner đã bắt đầu.
MATD3 xuất hiện sau wave MASAC. Khi thấy MASAC đang training, laptop và
cuộc trò chuyện có thể đóng; job tiếp theo đã được lập trình trong parent.

CPU chỉ được tạo khi checkpoint upload thành công. Lỗi upload, launch hoặc
render được ghi lại; tự động hóa này không bảo đảm phục hồi mọi lỗi dịch vụ.
Parent gửi CPU request trước khi thoát nên có thể có vài giây overlap giữa
hai session, nhưng evaluate/render không giữ GPU session chạy.

## Cách gửi lại flow

```bash
.venv/bin/python -m uav_network kaggle launch --stage train \
  --config configs/motion-pql.yaml --algorithm both --account account_02 \
  --required-gpu-hours 22 --timeout-seconds 39600 \
  --credential-dataset haibro1234/uav-wandb-credential-masac50k-d59860c \
  --set runtime.eval_episodes=30 --set runtime.eval_seed=2000000044 \
  --cpu-visualize
```

`--required-gpu-hours` kiểm tra quota còn lại, không giới hạn chi phí 22 giờ.
Không gửi lại lệnh chỉ vì đóng máy hoặc W&B chưa xuất hiện trong lúc setup.

Bằng chứng flow đã chạy hết trước lượt dài này:
[GPU→CPU smoke và checkpoint khớp](benchmarks/pql-cpu-handoff-20261014.json).
Lượt đó cả GPU và CPU đều Complete, CPU không có CUDA, video giải mã được,
weights tải xuống khớp training. Đây là kiểm tra orchestration, không phải
bằng chứng hội tụ của lượt dài đang chạy.

Manifest local: `outputs/kaggle/uavnet-train-20261014-181508-908e2d/launch.json`.
Kaggle GPU output có `cpu-visualization.json` chứa URL CPU thực tế khi gửi thành công.
Kaggle CPU output có `visualization-result.json`, `hardware.json` và
`visualize/{masac,matd3}/final_episode.mp4`, `final_episode.png`, `metrics.json`;
`visualize/comparison.json` và `comparison.csv` khi đủ hai checkpoint.

Snapshot W&B đã xác nhận MASAC Running trên CUDA, 4096 joint transitions.
Hai benchmark preflight của session này đều Passed trước training.
[Bản ghi readiness](benchmarks/pql-autonomous-launch-20261015.json).
