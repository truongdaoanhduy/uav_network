# 01 — Bằng Chứng, Lý Do Các Bản Vá v3.2 & Khảo Sát Q1 Hybrid MARL (2024–2026)

## 1. Bốn bản vá cốt lõi trong v3.2 (và lý do tại sao)

1. **Đóng gói Payload nhị phân động & khóa cứng MTU (`uav_adapter/packets_and_queues.py`):**
   - *Lỗi v3.1:* `SearchDataPacket` gán cứng `payload_bytes = 512` nên 1 ô hay 100.000 ô đều báo `570 B`, làm sai airtime (gói 1 ô bị phóng đại `2.19×`), throughput và contention.
   - *Bản vá v3.2:* Dùng binary layout `struct` chuẩn big-endian: **12 B app header** (`>HBBIHH`) + **8 B/entry** (`>BHHHB`: `origin u8, cell u16, n_pos u16, n_neg u16, pad u8`).
     $$\text{payload\_bytes} = 12 + 8\cdot |\text{entries}|, \qquad \text{frame\_bytes} = 58 + \text{payload\_bytes}$$
     Tối đa **62 entries** cho MTU `512 B` (`508 B` payload / `566 B` frame); từ entry thứ 63 tự động **`raise ValueError`**.

2. **Hiệu chuẩn lại $\lambda_B$ theo xác suất giao gói thực $\text{PDR} < 100\%$:**
   - *Lỗi v3.1:* Với $\lambda_B = 100$ và gói đầy MTU trên 2-hop (`7.24 ms`), 1 ô mới (`+2.4`) đòi hỏi $\text{PDR}_{\min} = 30.17\%$ (và 40–60% khi có MAC retries) mới không bị lỗ kỳ vọng $\implies$ nguy cơ cao làm policy collapse về `NO_SEND`.
   - *Bản vá v3.2:* Chốt **$\lambda_B = 30.0$** (dải an toàn `[20, 40]`, khuyến nghị curriculum `10 → 30` trong 20–30% đầu training). Với payload động thực tế, gói `1 cell` 2-hop chỉ cần $\text{PDR}_{\min} = 4.13\%$, gói `32 cells` 2-hop chỉ cần `6.61%` (chịu được cả 3× MAC retry ở PDR 20–30%), trong khi mọi gói trùng vẫn bị phạt âm (`-0.05` đến `-0.16`/bước) nên `NO_SEND` luôn thắng spam.

3. **Vá lỗ hổng Reward Hacking dương (`+5.0`) trên ô False Positive (`uav_adapter/reward_engine.py`):**
   - *Lỗi v3.1:* `team_target_latch` không lọc theo `true_target_mask` ($\mathcal{T}^*$), nên 1 ô trống dính 2 báo giả ($m_F = 0.96 \ge 0.95$) nhận `+10.0` từ `r_target` trừ `5.0` từ `r_FP` = **`+5.0` lãi ròng**.
   - *Bản vá v3.2:* Phân hoạch tuyệt đối: `team_tp_latch` chỉ kích hoạt khi $c \in \mathcal{T}^* \land m_F \ge \tau$ (`+10.0`), còn `team_fp_latch` chỉ kích hoạt khi $c \notin \mathcal{T}^* \land m_F \ge \tau$ (`-5.0` kích hoạt + tùy chọn phạt duy trì nhẹ `-0.1`/bước để khuyến khích xác minh hạ $m_F < \tau$).

4. **Bổ sung bài kiểm thử thực tế 2-hop DATA + E2E ACK ngược chiều + Blackout (`MiniDiscreteEventNet`):**
   - Chứng minh trực tiếp bằng mô phỏng sự kiện rời rạc: gói `U0 -> U1 -> U2` đi qua 2 hop, `U2` hợp nhất Max-plus-Sum và phát `E2EAckPacket` (90 B) đi ngược `U2 -> U1 -> U0`, `U0` chỉ tăng watermark khi nhận E2E ACK, Data PDR không bị `E2EAckPacket` làm phồng, và gói trong blackout bị dọn khi quá hạn TTL.

## 2. Khảo sát 5 bài báo Q1 (2024–2026) về Hybrid MARL cho UAV & 3 bẫy Gate 4

| Công trình | Venue / Năm | DOI / URL | Cơ chế xử lý hành động lai |
| :--- | :--- | :--- | :--- |
| **HA-MADDPG** (Task Offloading & Trajectory) | IEEE TMC (2026) | `10.1109/TMC.2026.3712259` ([IEEE CSDL](https://www.computer.org/csdl/journal/tm/2026/11/11603311/2i0MS3EbQJ2)) | Gumbel-Softmax khả vi cho offloading rời rạc + quỹ đạo liên tục |
| **MAHPPO-RSP** (Z. Bai et al.) | IEEE TWC (2024) | `10.1109/TWC.2024.3458194` ([ResearchGate](https://www.researchgate.net/publication/384479565_An_MA-HPPO_Approach_for_Multi-UAV_Data_Collection)) | Actor 2 nhánh rời rạc (on/off comm, sensing) + liên tục (power, motion) |
| **Latent-Space TD3** | IEEE TMC (2024) | [IEEE CSDL (10381761)](https://www.computer.org/csdl/journal/tm/2024/09/10381761/1TsSnDzy9oc) | Embedding table + cVAE ánh xạ về không gian tiềm ẩn liên tục |
| **CT-MASAC-BAC** (Y. Peng et al.) | IEEE IoTJ (2024) | `10.1109/JIOT.2023.3329346` ([ResearchGate](https://www.researchgate.net/publication/375221342_UAV-based_Emergency_Communications_An_Iterative_Two-stage_Multi-agent_Soft_Actor-critic_Approach_for_Optimal_Association_and_Dynamic_Deployment)) | MASAC quỹ đạo liên tục + Branch-and-Cut cho association rời rạc |
| **MAHPPO-AM** (T. Sun et al.) | arXiv (2025) | [arXiv:2509.23905](https://arxiv.org/html/2509.23905v1) | Shared encoder + Softmax Action Masking + Gaussian 3D trajectory |

**3 bẫy bắt buộc kiểm thử ở Gate 4 (đã viết sẵn trong `gate4_hybrid_marl_test_template.py`):**
1. **Dấu 2 thành phần entropy & `tanh` Jacobian:** Soft Bellman Target dùng `min_q - alpha_m * log_pi_m - alpha_d * log_pi_d` (tuyệt đối không nhầm dấu với `dist.entropy()` và không thiếu `-sum log(1 - tanh^2(z) + eps)`).
2. **Target entropy động theo mask khi blackout:** Khi blackout chỉ còn $K_{\text{valid}} = 1$ (`NO_SEND`), entropy bằng `0`. Nếu giữ hằng số $\bar{\mathcal{H}}_d = 0.6\ln 6$, auto-tuning sẽ đẩy $\alpha_d \to +\infty$. Bắt buộc dùng $\bar{\mathcal{H}}_d(s) = 0.6 \ln K_{\text{valid}}$ (hoặc chỉ update $\alpha_d$ trên các mẫu có $K_{\text{valid}} \ge 2$).
3. **Gradient flow qua cả 2 head:** Kiểm tra `grad_norm(motion_head) > 0` và `grad_norm(dest_head) > 0` bằng Per-Agent Exact Expected Q trên 6 lựa chọn, không gọi `.detach()` lên vector xác suất `probs`.
