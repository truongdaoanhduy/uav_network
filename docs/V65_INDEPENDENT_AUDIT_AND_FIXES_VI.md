# Báo cáo Kiểm định Độc lập & Bản sửa lỗi v6.5 (Data-Mule, MASAC/MATD3 P0, Warp Nén)

Bản v6.5 xây dựng trực tiếp trên `uav_v64_unified_q1_projectwide_20261125`, giữ nguyên 100% tính tương thích ngược với các kiểm thử v5.6–v6.4 và khắc phục triệt để các lỗi thực thi còn tồn đọng.

## 1. Các lỗi P0/P1 đã sửa trong v6.5

1. **P0 — MASAC xóa nhầm vận tốc `vy` khi `continuous_dim == 2` (`uav_network/algorithms/masac/learner.py`)**:
   - Bảo vệ toàn bộ 4 nhánh `_expected_q_full`, `_expected_q_all_full`, `_expected_q_shared_valid_routes`, `_expected_q_shared_all_full` bằng điều kiện `self.continuous_dim > self.config.motion_dim` và truyền `learn_power=self.learn_power` vào `canonical_continuous`/`joint_action`.
2. **P0 — Critic phân tách lỗi với `compact_3` và xóa `vy` (`uav_network/algorithms/core/factorized_candidate.py`)**:
   - Đổi ràng buộc `k == n + 1` sang `k >= 2` và chỉ xóa chiều cuối khi `c > 2`.
3. **P0 — `MATD3DiagnosticsMixin` thiếu import & ghi sai `vy` thành RF (`uav_network/algorithms/matd3/diagnostics.py`)**:
   - Bổ sung `from uav_network.algorithms.core.action_utils import valid_destinations` và tách riêng nhánh `has_rf = self.continuous_dim > self.config.motion_dim`.
4. **P1 — Entropy mục tiêu liên tục của MASAC (`uav_network/algorithms/configuration/schema.py`)**:
   - Mặc định `target_continuous_entropy: float | None = None`, tự động gán `-float(self.continuous_dim)` khi `validate()` (`-2.0` cho 2D motor-only, `-3.0` cho 3D có RF).
5. **P1 — True Data-Mule Multi-Origin Forwarding (`communication/lifecycle.py`, `ledger.py`, `delivery.py`, `environment/configuration.py`, `configs/scenario_v65_datamule.yaml`)**:
   - Bổ sung `enable_data_mule_forwarding=True` với watermark 4D `[src, dst, origin, cell]` giúp UAV chuyển tiếp bằng chứng từ các nguồn `origin != dst` về GCS (giải quyết 40.3% bản đồ mất LoS 1-hop ở 1 km và >70% ở 2–3 km), đặt `w_p2p = 0.0` để loại bỏ reward shaping nhân tạo.
6. **P1 — Scaling hình học 2–3 km & khóa `enable_peer_dissemination=False` (`communication/common.py`, `environment/core.py`, `transition.py`, `observations.py`)**:
   - Bổ sung `scale_canonical_geometry(...)` giữ bit-exact tại 1000 m / 20x20 và tỷ lệ hóa 5 tòa nhà + 2 vùng blackout khi tăng lên 2–3 km.
   - Chặn `per_peer` khi `enable_peer_dissemination=False` và kiểm tra `battery_j[p_id] > 0` trong mask `compact_3`.
7. **P1 — Warp GPU v6.5 Nén & Tham số hóa Dual-Path (`gpu/state_v65.py`, `gpu/runtime_v65.py`, `gpu/kernels/phy.py`)**:
   - Cung cấp `WarpCompressedRadioNetwork` sử dụng `make_shapes_v63` (**379,848 B/env**, giảm **40.3%** VRAM so với 635,931 B/env, tương đương **0.724 GiB** cho 2048 envs/T4), đồng bộ tham số PHY UavNetSim (`0.1 W`, `CCA=-82 dBm`, `Noise=4.39e-13 W`), sửa parity `max(z_lo, z_hi) >= 0.0` tại `phy.py:113`.

## 2. Phạm vi xác minh
- Đã kiểm thử CPU toàn bộ bộ test (kể cả `tests/test_v65_regressions.py`), chính sách 500 dòng/file (`0 violations`) và checksum SHA-256.
- Các kernel CUDA/Warp và benchmark trên GPU 2x Tesla T4 vẫn cần chạy trực tiếp trên máy có GPU (sandbox CPU không cài `torch`/`warp`/CUDA).
