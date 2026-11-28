# V118 — Phán Quyết Dứt Khoát Về Action V117 (Thực Nghiệm Tự Nhiên) & Kịch Bản Canonical Sạch

**Trạng thái:** `NOT SCENARIO_FROZEN` (chờ chạy lại 1 lượt GPU gate hậu-V113 trên Vast.ai `cuda:0` cho lõi sạch này; **30/30 tệp `uav_network/algorithms/` giữ nguyên byte-identical**).  
**Hành động mã nguồn V118:** Gỡ bỏ hoàn toàn lớp phủ thử nghiệm CPU-only của V117 (`a_evidence 0–3`, `a_relay 0–7`, `experimental_network_action_view 569` chiều) khỏi nhánh kịch bản canonical, quay về nền sạch V115 (đã có vá `C37` fail-closed CPU, `C38` ABI vận tốc chuẩn hóa `[-1, 1]`, `C42` launcher `run_v115_canonical_ds_baseline.py`, và 4 YAML tách biến `C20`). Toàn bộ bằng chứng tóm tắt V117 được lưu tại `outputs/v117_audit_summary/`.

---

## 1. Bằng chứng thực nghiệm tự nhiên (KHÔNG chèn backlog nhân tạo) bác bỏ `a_evidence` và `a_relay`

Thử nghiệm của V117 chỉ chứng minh được tác dụng nhờ **chèn nhân tạo 80 ô dirty** (`synthetic_backlog: true`) và **HELLO giả lập**. Khi chạy thực nghiệm CPU tự nhiên trên **V117** (3 seeds `44, 45, 46`, 300 bước, 6 UAV quét `62,5%` bản đồ $1\text{ km}\times 1\text{ km}$, lưu tại `outputs/v118_audit/01_natural_mission_backlog_and_relay_verdict.json`):

| Kịch bản tự nhiên | `max_dirty` / `p95_dirty` (trần gói 62) | Tỷ lệ bước `dirty > 62` | Kết quả `a_evidence` (`1=DS, 2=AoI, 3=Comb` vs `0=Auto`) | Kết quả `a_relay` (ép chọn relay vs `0=Auto`) |
| :--- | :---: | :---: | :--- | :--- |
| **Base (`v103`)** | `15 / 8,0` | **`0,00%`** (0/1.800) | **`ΔTP = 0, ΔDelivery = 0, ΔReward = 0,0`** (3/3 seeds — **100% dead action**) | **`ΔTP = 0`**, **giảm `-78` đến `-102` gói (`-5,3%`)**, tăng timeout `+25–29%` |
| **RF-only (`v111`)** | `74 / 60,0` | **`2,98%`** | `ev1` (DS) & `ev3` (Comb): **`ΔTP = 0, ΔDelivery = 0, ΔReward = 0,0`**; `ev2` (AoI): **`ΔReward = -0,011` đến `-0,134`** (âm) | `multi_route` chỉ có **`4,3–4,6%`** bước; **nhân đôi `mac_ack_timeout`** (`87–111` $\to$ `171–234`), giảm delivery `-1` đến `-19`, **`ΔTP = 0`** |

## 2. Tại sao KHÔNG port `a_evidence` và `a_relay` sang NVIDIA Warp GPU ở V118?

1. **Tự đẻ thêm 9 xung đột mới (`C56–C64`, gồm 4 P0 mới)** mà không đóng bất kỳ P0 cốt lõi nào từ V112/V113.
2. **Phá vỡ yêu cầu baseline `MATD3` & `MASAC` gốc:** `a_comm (3) × a_evidence (4) × a_relay (8) = 96` tổ hợp rời rạc trên không gian liên tục `(x, y)` không thể học bằng `MATD3`/`MASAC` gốc (bắt buộc phải độ chế sang hybrid parameterized MARL như MAPDQN / Gumbel-Softmax).
3. **Phá vỡ trần `< 500` dòng và làm chậm GPU Warp `4096×900`:** Sửa `mission_v79.py` (416 dòng) sẽ vượt trần 500 dòng; chạy top-62 DS sort trong `inject_v67` gây register spill và warp divergence nặng cho một action có lợi ích tự nhiên bằng `0,0`.

---

## 3. Bảng phán quyết cuối cùng: Giữ / Sửa / Bỏ cho kịch bản Canonical

| Thành phần | Phán quyết | Lý do ngắn gọn |
| :--- | :--- | :--- |
| **Action liên tục 2D `(x, y)`** (`act_dim = 2`, ABI `[-1, 1]`) | **GIỮ** | Chuẩn Zhang 2024 Eq. (1–2), tương thích 100% `MATD3`/`MASAC` gốc (`C38` đã vá). |
| **Truyền thông cục bộ cố định (`event_driven_comm_action`)** | **GIỮ** | Phù hợp kết luận tách tầng của Fu et al. (IEEE IoTJ 2025 — TL-RATS), pure-GCS (`PEER=false, MULE=false`). |
| **`tx_power_w = 0.1 W` cố định** | **GIỮ** | Tham số radio chuẩn UavNetSim-v1; PA RF chiếm `< 0,003%` tổng năng lượng. |
| **`a_evidence (0–3)`, `a_relay (0–7)`, obs `569` chiều (V117)** | **BỎ khỏi canonical** | Thực nghiệm tự nhiên chứng minh vô nghĩa/gây hại và phá chuẩn `MATD3`/`MASAC` gốc. |
| **`PEER` / `Data Mule` trong kịch bản chính** | **BỎ khỏi mặc định** | Vast.ai V112 đã chứng minh PEER gây giảm `-32,67 pp` GCS delivery dưới RF-stress (`C09/C34`). |
| **`w_p2p=0`, `lambda_b=0`, `ds_aoi_eta=0`, `enable_accel_energy=false`** | **KHÔNG viết vào phương trình active** | Tham số đang tắt; chỉ nêu ở phần giới hạn/ablation. |

---

## 4. Lệnh chạy cuối trên Vast.ai `cuda:0` để chốt `SCENARIO_FROZEN` và mở Phase 2

```bash
# 1. Xác nhận full pytest (593+ tests) & 4x adverse stress 4096x900 trên V118 sạch
python3 -m pytest -q tests/ research/zhang_xue/tests/
python3 scripts/gpu/v112_adverse_action_stress_gate.py --device cuda:0 --envs 4096 --steps 900

# 2. Sau khi khóa SCENARIO_FROZEN và hoàn tất S25 (replay VRAM FP16 + bỏ trùng state):
python3 scripts/gpu/run_v115_canonical_ds_baseline.py --algorithm matd3 --envs 64 --steps 300 --output outputs/phase2_matd3.json
python3 scripts/gpu/run_v115_canonical_ds_baseline.py --algorithm masac --envs 64 --steps 300 --output outputs/phase2_masac.json
```
