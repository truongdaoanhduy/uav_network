# Chạy lại V84 theo từng bước

Mã hoạt động nằm tại `source_repository/experiments/v80_vast_20261126` trong ZIP, hoặc `experiments/v80_vast_20261126` khi clone nhánh GitHub. Tên thư mục được giữ để bảo toàn lịch sử; runner báo `version: V84`.

## 1. Kiểm tra source trước khi dùng GPU

```bash
cd source_repository/experiments/v80_vast_20261126
python3 scripts/maintenance/verify_source_manifests.py --check
python3 scripts/maintenance/check_project_policy.py --scope full
PYTHONPATH=.:research/zhang_xue/src pytest -q tests research/zhang_xue/tests
```

Môi trường đã kiểm: Python 3.12.16, Torch 2.14.1+cu130, Warp 1.18.0, RTX 3060 Laptop 12 GiB. Đây là bản ghi môi trường đã chạy, không phải yêu cầu mọi máy phải cài đúng toàn bộ phiên bản. Không ghi đè driver hoặc môi trường đang dùng để ép khớp.

## 2. Đối chiếu cùng ngân sách dữ liệu thật

Trên máy hiện tại, physical GPU1 được chọn bằng `CUDA_VISIBLE_DEVICES=1`, sau đó runner dùng `cuda:0`. Trên máy khác, chọn GPU còn trống trước khi thay số này. Hai tiến trình trong vòng lặp chạy tuần tự.

```bash
for algorithm in matd3 masac; do
  for mode in raw dual raw_penalized; do
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:research/zhang_xue/src \
    python3 scripts/gpu/run_v81_real_rollout.py \
      --algorithm "$algorithm" --replay-action-mode "$mode" \
      --envs 64 --steps 300 --horizon 900 --hidden 64 --batch-size 128 \
      --learning-starts 4096 --replay-capacity 65536 --seed 44 \
      --shield-penalty-lambda 1.0 --log-interval 10 \
      --world-device cuda:0 --learner-device cuda:0 \
      --allow-experimental-physics \
      --output "outputs/v84_${algorithm}_${mode}_e64_s300.json"
  done
done
```

Runner mặc định nạp `configs/learners/v81_matd3_masac_stable.yaml`; mỗi JSON ghi effective config và SHA đầu vào. `--hidden 64`, `--batch-size 128`, `--learning-starts 4096` và `--replay-capacity 65536` ghi đè các trường tương ứng của thuật toán đang chạy. Các tham số này được giữ như nhau ở các cohort; không nhầm config gốc H256/B8192 với giá trị thực thi H64/B128.

`raw` là baseline dùng shield như một phần môi trường. `raw_penalized` thêm chi phí correction vào mục tiêu huấn luyện. `executed`/`dual` là relabelling thực nghiệm; chưa được chứng nhận idempotence toàn cục. Không chọn `dual` chỉ vì tên nghe an toàn hơn.

Đối chiếu số `updates`, `true_transition_count`, `dual_penalty_transitions_added`, `replay_rows_inserted`, safety subtypes và learning diagnostics. Native reward và training reward có nghĩa khác nhau. Log thưa đếm events theo cửa sổ; trường mean reward vẫn chỉ là snapshot bước ghi.

## 3. Kéo dài chẩn đoán và kiểm tra song song

Sau khi kiểm tra log 300 bước, đổi `--steps 900` để có đủ một horizon. Các lượt 900 bước trong gói là online diagnostics seed44, không phải đánh giá held-out.

E4096 trong gói dùng `--envs 4096 --steps 3 --batch-size 8192 --learning-starts 8192`, các tham số khác giữ như trên. Hai update thật kiểm được collection/replay/optimizer/overflow ở quy mô này. Không diễn giải thành 900 bước hoặc benchmark throughput.

Chưa mở E4096 × 900 vì lượt E64 dài vẫn còn pair-safety failures. Cần xác định tập trạng thái joint khả thi, bổ sung observation qua HELLO có timestamp, rồi so sánh nhiều seed và evaluation độc lập trước khi mở ngân sách lớn.

## 4. Tìm bằng chứng

- `01_V83_AUDIT_V84_FIXES_VI.md`: lý do từng bản sửa và giới hạn.
- `02_RUNS.csv`: tất cả cohort V83 nguyên bản và V84 đã sửa, kèm SHA log và commit.
- `03_V83_VERDICT_MATRIX.csv`: đủ 8 nhóm nhận định V83.
- `04_V80_24_ISSUES_UPDATED.csv`: đủ 24 mục cũ, không gộp OPEN thành FIXED.
- `current_raw_evidence`: RED/GREEN, CPU/CUDA, JSON/JSONL/stdout, hardware và lệnh chạy.
- `kaggle_history`, `history_archives`, `history_v82_raw`, `vast_v80_historical_raw`: bằng chứng lịch sử giữ nguyên; không đổi nhãn thành lượt V84.
- `patches`: các diff giữa phiên bản; source thực tế lấy theo commit trong SOURCE_PROVENANCE.json.

Không submit Kaggle mới ở lượt này vì GPU Vast đã dùng được. Không cần đọc key.txt để tái hiện các phép kiểm đã chạy trên Vast.
