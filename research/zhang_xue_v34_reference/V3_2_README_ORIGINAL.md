# Zhang + Xue + UavNetSim-v1 + Hybrid MASAC/MATD3 — Verified v3.2 Package

Gói **v3.2** (ngày 2026-11-25) khắc phục triệt để 3 lỗ hổng được phát hiện ở vòng kiểm định v3.1 (lỗi payload tĩnh `570 B`, lỗi hiệu chuẩn $\lambda_B=100$ khi PDR 2-hop $<30\%$, và lỗ hổng thưởng dương `+5.0` trên ô False Positive), đồng thời bổ sung:
- Bộ mô phỏng sự kiện rời rạc tối giản (`MiniDiscreteEventNet`) kiểm chứng thực tế **2-hop DATA + E2E ACK ngược chiều + blackout + queue expiry + hồi phục**.
- Script kiểm thử **Gate 4 PyTorch** (`uav_adapter/gate4_hybrid_marl_test_template.py`) để chạy ngay trên máy có PyTorch/Kaggle GPU.

> Minh bạch phạm vi kiểm chứng: Các kiểm thử Gate 1–3 (kể cả 2-hop DATA/E2E ACK trên `MiniDiscreteEventNet`) đã chạy pass trong môi trường Python/NumPy cục bộ; Gate 4 (PyTorch gradient) và Gate 5 (tích hợp trực tiếp SimPy UavNetSim + Warp parity) cần chạy trên môi trường có `simpy` và `torch`.

## Danh mục tệp

| Tệp | Nội dung |
| :--- | :--- |
| `01_V3_2_FIXES_EVIDENCE_AND_Q1_HYBRID.md` | Bằng chứng & lý do chi tiết cho 4 bản vá v3.2 + khảo sát 5 paper Q1 (2024–2026) về Hybrid MARL cho UAV |
| `config_v3.2.yaml` | Cấu hình khóa cứng **0 trường `null`**, payload động `12 + 8*k` (MTU 512 B, tối đa 62 ô), $\lambda_B=30$ (curriculum `10→30`), tách TP/FP Latch |
| `uav_adapter/packets_and_queues.py` | `SearchDataPacket` đóng gói binary thật (`struct`, hard-raise `ValueError` khi vượt MTU), `E2EAckPacket`, `PacketQueueManager`, `MiniDiscreteEventNet` (2-hop + blackout) |
| `uav_adapter/reward_engine.py` | `RewardEngine` tách tuyệt đối `team_tp_latch` ($c \in \mathcal{T}^*$) và `team_fp_latch` ($c \notin \mathcal{T}^*$) + $\lambda_B=30$ |
| `uav_adapter/ds_belief.py` | `ds_log_space_safe` (`int-diff` FP32 + FP64→FP32 khi $m_f \ne m_e$) & `BeliefTable` (`uint16`, Max-plus-Sum / MAX) |
| `uav_adapter/phy_patch.py` | `sinr_calculator_patched` áp dụng $P_t G_t G_r$ đối xứng cho cả tín hiệu lẫn nguồn nhiễu |
| `uav_adapter/gate4_hybrid_marl_test_template.py` | Bộ kiểm thử PyTorch sẵn sàng chạy cho Gate 4 (dấu 2 entropy, `tanh` Jacobian, $\bar{\mathcal{H}}_d = 0.6\ln K_{\text{valid}}$, `grad_norm > 0`) |
| `tests/test_v3_2_full_gates.py` | Bộ test tự động kiểm chứng toàn bộ các bản vá v3.2 (đã chạy pass 100%) |
