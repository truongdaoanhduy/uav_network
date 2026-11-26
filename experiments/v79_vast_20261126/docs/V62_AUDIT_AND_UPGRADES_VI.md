# Báo cáo Kiểm định Đa góc nhìn v6.1 (đối chiếu v5.8) & Bản nâng cấp v6.2

Ngày: 25/11/2026  
Phạm vi: Kiểm định toàn diện gói **v6.1** (`uav_v61_seven_radio_step_by_step_20261125`) và đối chiếu với các phán quyết trong **v5.8**, triển khai toàn bộ bản vá cốt lõi **v6.2** tuân thủ nghiêm ngặt quy tắc `check_project_policy.py` ($\le 500$ dòng/tệp).

## 1. Thẩm định các phán quyết v5.8 & bản sửa v6.1 (Đúng hay Sai?)

| Hạng mục | Nhận xét / Bản sửa v5.8–v6.1 | Phán quyết phản biện Q1/A* | Giải pháp đã triển khai trong v6.2 |
| :--- | :--- | :--- | :--- |
| **AoI EWMA end-to-end** | Chẩn đoán EWMA chưa nối end-to-end; kết luận phải phình wire format khỏi 8 byte. | **Chẩn đoán ĐÚNG, kết luận kiến trúc SAI.** Nhờ tính nửa nhóm của hàm mũ trong [Xue et al., Electronics 2026](https://www.mdpi.com/2079-9292/15/9/1875) Eq. 9–10, cặp hiệu dụng $(\tilde{n}^+,\tilde{n}^-)$ + `gen_ts_s` vừa khít 8 byte bằng fixed-point `UQ8.8`. | Triển khai `encode_uq88`/`decode_uq88`/`fuse_ewma_provenance_uq88` xuyên suốt `ledger.py`, `delivery.py`, `observations.py` (`aoi_ewma_end_to_end=True` trong `make_v62_config()`). |
| **Reward `info_gain`** | Nhận xét `1/prev_total` chưa phải DS entropy, nhưng v6.1 chưa sửa. | **ĐÚNG và còn SÓT 2 bug:** `delivery.py:75` tính `prev_total` *sau* khi ghi đè ledger (thành `post_total`), và thưởng dương cả khi xung đột làm tăng entropy. | Thay bằng **Deng Entropy** ([Deng, 2016](https://econpapers.repec.org/RePEc:eee:chsofr:v:91:y:2016:i:c:p:549-553)) với sàn đơn điệu `peer_h_min[rx, cid]` bảo đảm $\sum_t \Delta I_{\text{Deng}} \le 1.0$ (chống reward farming tuyệt đối). |
| **Safety Shield** | Thêm `trajectory_safety_violations` + cờ `safety_failure_terminate`. | **Chẩn đoán ĐÚNG, sửa CHƯA ĐỦ:** `flight_safety_energy.py` còn dead code phanh tường (`155–165`), nhân sai `*0.5` phanh tòa nhà (`177`), thiếu phanh cặp UAV, và `transition.py` vẫn cho UAV vừa va chạm sensing ăn điểm. | Sửa phanh đủ 4 bức tường, bỏ `*0.5` tòa nhà, thêm bất biến phanh cặp UAV `v_close²/(2*a_max)` cho cả 2 UAV (pass test đối đầu 25 m/s từ 200 m), và chặn sensing khi có `safety_violation`. |
| **Actor Adapter 522d & RF head** | Sửa feature `#9` (`alive`) + gắn 7-bit mask = 522d; bỏ qua đầu ra `controls[:, 2]`. | **Wiring 522d ĐÚNG, vứt RF head SAI về MARL** (gây nhiễu entropy MASAC và phương sai Critic MATD3). | Bổ sung `decode_tx_power_w` (`[0.05 W, 0.50 W]`), nối `tx_power_per_radio[tx]` vào CCA/RX/SINR (`channel.py`), mô hình PA tuyến tính `rf_pa_dc_power_w` (`3.0 W` tại `0.5 W`, `0.75 W` tại `0.05 W`), và phạt airtime theo công suất phát. |
| **HOL Queue trên Warp** | Dùng `_promote_queued_slot` hoán đổi `head <-> chosen`. | **Không có data race nhưng còn 3 BUG:** (1) preempt gói đang retry/freeze và reset `retries=0` vô hạn; (2) swap phá thứ tự FIFO so với CPU; (3) copy cứng 62 entries (992 B) mỗi lần promote. | Thêm guard chống preempt `active_slot >= 0` trong `v56_warp_event_kernels.py`, thiết kế `layout_v62.py` dùng `q_order`/`q_used` zero-copy (`Q=32, P=6`). |
| **Dự toán VRAM Warp** | v5.8 bóc trần v5.7 cấp phát thật `635,931 B/env` thay vì `315,710 B/env`. | **v5.8 BẮT LỖI ĐÚNG 100%.** | Tạo `zhang_xue/gpu/layout_v62.py` nén dtype thật (`int16`/`int8`/`uint8`/`float16`, giữ `gain/rx_sig/rx_interf` ở `float32` tránh underflow 2.4 GHz) xuống **`384,549 B/env`** (`0.733 GiB` / 2048 envs, giảm **39.5%**), tính trực tiếp từ `SHAPES_V62`. |
| **Lỗi MAC CPU & Parity Warp** | Chưa phát hiện các lỗi biên trong `delivery.py`, `scheduler.py`, `runtime.py`, `v57_warp_phy_kernels.py`. | Phát hiện thêm 5 lỗi cụ thể ảnh hưởng parity & độ tin cậy định tuyến. | Sửa off-by-one `MAX_HOPS` (`delivery.py`), trả gói mất route về store-carry-forward khi `WAIT_ACK` timeout (`scheduler.py`), khóa `bool` trong config (`configuration.py`), đồng bộ CCA `-82 dBm` (`6.31e-12 W`) & reset `tx_airtime` (`runtime.py`), dùng `wp.floor` + `max(z_lo, z_hi) >= 0` (`v57_warp_phy_kernels.py`). |

## 2. Kết quả kiểm định & Governance v6.2

- **Policy Governance (`scripts/check_project_policy.py`):** **0 vi phạm** trên toàn bộ cây mã nguồn (100% file $\le 500$ dòng, UTF-8, AST hợp lệ, không file phẳng ngoài danh sách legacy).
- **Kiểm thử hồi quy CPU (`unittest`):** **68 PASS, 0 FAIL, 1 SKIP** (do container đóng gói không cài `warp-lang`), bao gồm đủ 5 test mới trong `research/zhang_xue/tests/test_v62_audit_and_upgrades.py`.
- **Chạy gate Tesla T4 (chỉ chạy trên máy có CUDA/Warp):**
  ```bash
  PYTHONPATH=.:research/zhang_xue/src python research/zhang_xue/scripts/v62_t4_parity_and_scale_gate.py \
    --allow-experimental --devices cuda:0 cuda:1 --sizes 32 256 1024 2048
  ```

## 3. Tài liệu tham chiếu chính

- Zhang et al., *IEEE Transactions on Intelligent Vehicles* 9(2), 2024: https://www.semanticscholar.org/paper/Enhancing-Multi-UAV-Reconnaissance-and-Search-DDPG-Zhang-Lin/c3962200ccb4224ae66a28a26f9bd424f4e1a7f9
- Xue et al., *Electronics* 15(9), 1875, 2026: https://www.mdpi.com/2079-9292/15/9/1875
- Zeng, Xu & Zhang, *IEEE Transactions on Wireless Communications* 18(4), 2019: https://www.researchgate.net/publication/324328894_Energy_Minimization_for_Wireless_Communication_With_Rotary-Wing_UAV
- Yates et al., *IEEE JSAC* 39(5), 2021: https://arxiv.org/abs/2007.08564
- Yong Deng, *Chaos, Solitons & Fractals* 91, 2016 (Deng Entropy): https://econpapers.repec.org/RePEc:eee:chsofr:v:91:y:2016:i:c:p:549-553
- Bianchi, *IEEE JSAC* 18(3), 2000 (IEEE 802.11 DCF): https://www.researchgate.net/publication/3234321_Performance_analysis_of_the_IEEE_80211_distributed_coordination_function
- UavNetSim-v1: https://github.com/Zihao-Felix-Zhou/UavNetSim-v1
- NVIDIA Warp Documentation: https://nvidia.github.io/warp/
