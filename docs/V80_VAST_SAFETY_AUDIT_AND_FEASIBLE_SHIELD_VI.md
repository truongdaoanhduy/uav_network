# V80 — KIỂM TOÁN BẰNG CHỨNG VAST.AI V79 & TRIỂN KHAI KHIÊN AN TOÀN KHẢ THI (66-CANDIDATE + 3-SWEEP CD)

**Ngày:** 2026-11-26 (UTC+07). **Nguồn kiểm toán:** `UAV_V79_VAST_GITHUB_KAGGLE_ALL_EVIDENCE_20261126.zip` (commit GitHub `6c536e8f1d690a253c82be55ce9a471db47680fd`, 25 lượt chạy Vast.ai trên `2× RTX 3060 Laptop 12 GiB`, PyTorch `2.14.1+cu130`, `warp-lang==1.18.0`).

## 1. Giải mã toán học: Tại sao MATD3 chết sớm 63 lần và MASAC chết sớm 117 lần (0/64 episode chạm 900 bước)?

Kiểm tra trực tiếp `matd3_e64_h900_300steps_long.progress.jsonl` (63 chết: **55 UAV–UAV, 8 geometry**) và `masac_e64_h900_300steps_long.progress.jsonl` (117 chết: **100 UAV–UAV, 17 geometry**, chết ngay từ bước 3) cho thấy **4 nguyên nhân gốc trong `_move_v79`**:

1. **Thiếu phép chiếu né ngang 66 ứng viên**: Khác với CPU `kinematic_safety_shield` (`flight_safety_energy.py:200–253`), `_move_v79` chỉ biết **phanh thẳng 1D** (`ratio = max(0, ||v|| - a_max) / ||v||`), không tận dụng thành phần gia tốc vuông góc để rẽ ngang quanh tòa nhà/đồng đội → tạo ra **bẫy phanh đuôi (Rear-End Braking Trap)** và trượt thẳng vào trụ tòa nhà.
2. **Hiệu ứng domino thứ tự `peer > u` (Gauss-Seidel 1 lượt)**: Trong Phase 1, UAV `u` kiểm tra `peer > u` bằng `cand[peer]` thay vì `v_safe[peer]`. Khi `peer` bị ép phanh sau đó, `v_safe[u]` không bao giờ được chiếu lại → Phase 2 giết episode.
3. **Dùng bộ lọc biên phanh bảo thủ làm điều kiện giết episode (Chết oan!)**: Phase 2 của `_move_v79` gọi thẳng `_pair_safe_v79` (chứa biên phanh đơn phương $v_{\text{close}}^2 / (2 a_{\max})$ đòi tới `140 m` khi bay ngược chiều $30\text{ m/s}$!). Do đó hai UAV cách nhau `59.1 m > 50.0 m` (chưa hề va chạm!) vẫn bị bật `failed[e] = 1`, phạt `-100` và cắt episode — trái ngược với CPU `trajectory_safety_violations` vốn chỉ kết liễu khi có **va chạm vật lý thực tế** (`min_sep < 50.0 m`).
4. **Đội hình khởi tạo quá sát biên ($71.59\text{ m}$) + Actor mù đồng đội/NFZ + Bệnh lý tối ưu hóa MATD3**: Biên dư ban đầu chỉ còn $21.59\text{ m}$ (nhỏ hơn quãng đường phanh $22.5\text{ m}$). Log bước 300 của MATD3 cho thấy `saturated_action_y_fraction_at_backward = 0.9258` (92,6% bão hòa `tanh`) và `q1_relu_0/forward_sample_zero_fraction = 0.9049` (90,5% Dead ReLU).

## 2. Đánh giá 6 bản vá & 8 mục mở (A–H) của Vast.ai V79

- **6 bản vá Vast.ai**: Đúng kỹ thuật và trung thực TDD, nhưng có 3 lỗ hổng ẩn: (i) `run_v79_real_rollout.py:138` gọi `auto_reset_gpu()` **trước** `world.guard()`, xóa sạch `failed`/`overflow` trước khi kiểm toán; (ii) `fix_actor_map_position.patch` sửa `actor_maps.py` nhưng V79 gọi `_observe_map_v79`; (iii) `mission_v79.py:316` vẫn gọi `_candidate_v78` hardcode `15.0`/`5.0`.
- **8 mục mở (A–H)**: Đều đúng hướng; ngoài ra còn thiếu 2 phát hiện quan trọng: (1) Actor 514-D hiện mù hoàn toàn trước vị trí UAV đồng đội & tòa nhà NFZ; (2) `actor_maps.py` dùng chiết khấu mass $m_t e^{-\eta \Delta t}$ khác đại số với `_fuse_uq88_v79` dùng chiết khấu count $p e^{-\eta \Delta t}$.

## 3. Bản vá V80 đã code & kiểm chứng (GIỮ NGUYÊN `safe_d=50m`, `a_max=5`, fail-closed)

- `research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py`:
  - `_candidate_v80`: nhận `dt_s`, `v_max`, `a_max` động (bỏ hardcode `15.0`/`5.0`).
  - `_project_uav_v80`: chiếu **66 ứng viên** trên đĩa gia tốc bằng online `argmin` không cấp phát bộ nhớ (tương đương toán học 100% CPU `kinematic_safety_shield`).
  - `_move_v80`: Pass 1 (hình học) + Pass 2 (**3 vòng coordinate descent** chỉ dùng `v_safe`), Phase 2 tách rõ va chạm vật lý thật (`boundary=1`, `building=2`, `uav_pair=4`) với cảnh báo biên phanh (`braking_margin_warning=8`), kèm cờ `--strict-braking-terminate` cho A/B ablation với V79.
- `research/zhang_xue/src/zhang_xue/gpu/mission_v80.py` & `scripts/gpu/run_v80_real_rollout.py`: Kiểm tra `overflow` **trước** `auto_reset_gpu()`, hỗ trợ `cuda` và `cuda:N`, ghi đủ 4-bit safety subtypes.
- `tests/test_v80_feasible_shield_parity.py`: Chứng minh **0 va chạm** trong bẫy phanh đuôi và **0 va chạm xuyên suốt 120 bước random policy với 6 UAV từ đội hình `71.59 m`**.

## 4. Lệnh chạy xác minh V80 trên Vast.ai / Kaggle CUDA

```bash
export PYTHONPATH="$PWD:$PWD/research/zhang_xue/src"
python3 scripts/gpu/run_v80_real_rollout.py --algorithm matd3 --envs 64 --steps 300 --horizon 900 --batch-size 128 --hidden 256 --world-device cuda:0 --learner-device cuda:0 --allow-experimental-physics --output outputs/v80/matd3_e64_h900_s300.json
python3 scripts/gpu/run_v80_real_rollout.py --algorithm masac --envs 64 --steps 300 --horizon 900 --batch-size 128 --hidden 256 --world-device cuda:0 --learner-device cuda:0 --allow-experimental-physics --output outputs/v80/masac_e64_h900_s300.json
```
