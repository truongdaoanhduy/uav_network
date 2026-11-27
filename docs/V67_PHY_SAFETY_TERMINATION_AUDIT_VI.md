# v6.7 — kiểm định lại từ ZIP v6.5 và v6.6; sửa logic PHY, sensing, termination

**Ngày:** 2026-11-25. **Đầu vào gốc:** `uav_v65_audited_datamule_masac_warp_20261125(1).zip` người dùng tải lên. **Mốc hợp nhất:** `code.zip` v6.6 (167 source files, kiểm tra ZIP hợp lệ). **Output:** ZIP mã nguồn v6.7 độc lập; không thay đổi repository GitHub hay tài khoản Kaggle, không dùng `key.txt`.

## Kết luận phản biện và trạng thái

| Ưu tiên | Vấn đề/đề xuất cũ | Kết luận có thể kiểm chứng | Thay đổi v6.7 |
|---|---|---|---|
| P0 | CCA −82 dBm được dùng làm ngưỡng khóa thu (`communication/channel.py`) | **Sai tầng giao thức.** CCA xác định busy/idle cho MAC; decode tại receiver quyết định qua SINR. Source `UavNetSim/phy/channel.py` tách `_deliver` (SINR) và `is_busy_for` (CCA). | Sửa acquisition sơ bộ thành `received_power >= noise_power * sinr_threshold` (trong điều kiện chưa có nhiễu), giữ CCA cho DIFS/backoff. Interference vẫn được kiểm tra tại cuối frame. **Chưa đạt hoàn toàn channel-event parity.** |
| P0 | Một UAV vi phạm an toàn khiến **toàn swarm** không được sensing | Logic thiếu local credit: `if ... or safety_violation` chặn tất cả UAV; lỗi một UAV có thể làm mất quan sát của UAV còn hoạt động. | Chỉ bỏ sensing của UAV liên quan đến `boundary`, `building`, hoặc hai đầu `pairwise`; vẫn phạt và có thể terminate theo cờ safety chung. Thêm `unsafe_sensing_uavs` vào info. |
| P1 | `np.any(battery<=0)` luôn kết thúc episode | Không phải bug cú pháp nhưng **định nghĩa mission bị hardcode**: hỏng một UAV có đồng nghĩa thất bại toàn nhóm hay không phải được khai báo trước khi so thuật toán. | Thêm `battery_termination_mode: any_depleted | all_depleted`. Default `any_depleted` giữ tương thích. Profile v6.7 dùng `all_depleted`; vẫn dừng khi hoàn thành tất cả target hoặc safety-terminal. |
| P1 | v6.6 kết luận 2km không nhận 68,25% cells do CCA | Con số 68,25% đúng với điều kiện `CCA AND SINR` trong CPU **cũ**, nhưng không phản ánh khả năng giải mã PHY theo UavNetSim. | Khảo sát mới tách `no_direct_decodable_link` và `no_direct_cca_sensed_link`, gắn nhãn hai đại lượng không thay thế nhau. |
| P1 | Full Warp GPU data-mule MAC đã sẵn sàng | **Chưa đúng**; `WarpCompressedRadioNetwork` chỉ cấp phát trạng thái, chưa có `step` ACK/TTL/4D origin. | Giữ `--gpu-gate` fail-closed, không thêm kết quả GPU giả. |
| P2 | Bỏ `NO_SEND`, `SHARE_PEERS`, các ngưỡng reward không chứng minh hiệu quả | Không đủ evidence từ 4 bước smoke hay topology test để xóa. Lợi ích phụ thuộc offered-load, churn, connectivity, reward, queue. | Giữ hai action phục vụ matched ablation. Không tuyên bố có tăng hiệu quả RL nếu chưa có hội tụ và nhiều seed. |
| P2 | Cố định Ptx = 0.1W; bỏ RF policy head, accel-energy reward | Thích hợp làm baseline ít chiều action, nhưng không phải định lý tối ưu ở mọi cấu hình. | Profile v6.7 vẫn fixed Ptx, continuous_dim=2, `lambda_b=0`, `w_p2p=0`. Các mô-đun optional lịch sử vẫn là mã cấu hình, không được xem là contribution chính. |
| P2 | Đổi `ds_mf=ds_me=0.8`, `confirm_th=0.90` ngay | Đây là câu hỏi calibration/ROC, không tự suy ra từ UavNetSim. | Giữ nguyên để tránh trộn nhiều biến trong cùng bản vá. Cần ablation 0.90/0.99 và false confirm. |

## Đối chiếu UavNetSim trực tiếp

**Nguồn mã chính thức đã truy xuất:**

1. [utils/config.py](https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/master/utils/config.py): `TRANSMITTING_POWER=0.1`, `CCA_THRESHOLD_DBM=-82`, NF 7dB, noise density −174 dBm/Hz, `MAX_QUEUE_SIZE=200`, `MAX_RETRANSMISSION_ATTEMPT=5`.
2. [utils/ieee_802_11.py](https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/master/utils/ieee_802_11.py): IEEE 802.11b 2 Mbps, BW=22 MHz, required SNR=6 dB, SIFS=10 μs, slot=20 μs.
3. [phy/channel.py](https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/master/phy/channel.py): `_deliver` dùng SINR để quyết định thành công; `is_busy_for` dùng CCA để phát hiện channel busy. Interference trong `_evaluate` được weighted theo phần trăm thời gian chồng lấn — **custom CPU v6.7 hiện sử dụng worst-case overlap**, nên không thể nói full parity.
4. [UavNetSim-v1 (Zhou et al., 2025)](https://arxiv.org/abs/2507.09852) và [NVIDIA Warp docs](https://nvidia.github.io/warp/v1.18/user_guide/runtime.html): simulator gốc là event-driven, Warp graph không tự động port Python queue/ACK/SimPy thành GPU kernels.

**Hệ quả vật lý được tái hiện bằng test:** noise ≈4,3896×10⁻¹³ W; SINR decode 6 dB đòi tín hiệu khoảng 1,7475×10⁻¹² W; CCA −82 dBm = 6,3096×10⁻¹² W. Tín hiệu `8×noise = 3,5117×10⁻¹² W` **dưới CCA nhưng trên ngưỡng SINR**: trong điều kiện không nhiễu, frame HELLO vẫn được giải mã và cập nhật tuyến. Test trước bản vá thất bại; sau sửa thành công. Điều này **không** nói tất cả tín hiệu dưới CCA đều giải mã được; tín hiệu dưới ngưỡng SINR vẫn thất bại.

## Khảo sát bản đồ sau sửa (mỗi ô 1 UAV, link trực tiếp về GCS, không interference)

| Bản đồ / lưới | Ô không đủ SINR (decode fail) | Ô dưới ngưỡng CCA tại GCS | Lưu ý |
|---|---:|---:|---|
| 1 km / 20×20 | 161/400 = 40,25% | 161/400 = 40,25% | Giá trị bằng nhau do hình học baseline này, không phải do CCA = RX threshold |
| 2 km / 40×40 | 635/1600 = 39,69% | 1092/1600 = 68,25% | **457 ô** có SINR lý thuyết đủ nhưng power dưới ngưỡng CCA |
| 3 km / 60×60 | 2079/3600 = 57,75% | 2995/3600 = 83,19% | Chưa mô phỏng việc packet qua relay và RLT QoS |

Các số là output của `scripts/maintenance/academic_scenario_audit.py` trên hình học CPU hiện tại, **không phải PDR**, và cũng không là kết quả kiểm định thuật toán đã học. `no_direct_receivable_link_cells` bây giờ là bản tương thích tên cũ với định nghĩa SINR-only; dùng `no_direct_cca_sensed_link_cells` để phân tích MAC carrier sensing.

## Thiết kế ablation cần tách riêng

Có bốn matched-profile mới:

- `research/zhang_xue/configs/scenario_v67_datamule_1km.yaml`
- `research/zhang_xue/configs/scenario_v67_no_peer_1km.yaml`
- `research/zhang_xue/configs/scenario_v67_datamule_2km.yaml`
- `research/zhang_xue/configs/scenario_v67_no_peer_2km.yaml`

Mỗi cặp tại một kích thước chỉ khác `enable_peer_dissemination` (ON/OFF). Các cấu hình đều chứa `battery_termination_mode: all_depleted`, `w_p2p=0`, `P_tx=0.1W`, `continuous_dim=2` tại learner. Chế độ `any_depleted` cũ vẫn có thể chọn qua YAML hoặc constructor; so sánh bất kỳ kết quả nào khác termination mode phải báo rõ confounder.

## Gate và trạng thái trung thực

```bash
# Không cần GPU hoặc Kaggle, và không đụng file token
PYTHONPATH=.:research/zhang_xue/src pytest -q tests/ research/zhang_xue/tests/
bash setup.sh --check-only
bash setup.sh --cpu-gate-v67 --seed 44      # MATD3 + MASAC 4 transitions + 1 optimizer update
python scripts/maintenance/academic_scenario_audit.py --sizes-m 1000 2000 3000
python scripts/maintenance/verify_source_manifests.py --check
bash setup.sh --gpu-gate                   # cố ý exit=3: BLOCKED
```

Đã chạy CPU learner gate v6.7 với 6 UAV: MATD3 và MASAC đều đúng continuous_dim=2, destinations=3, fixed Ptx0.1, 4 transitions, 1 optimizer update, return finite. **Không** gọi đây là trained policy, evaluation Q1 hay GPU E4096 throughput.

**Việc còn phải làm (không được gọi đã sửa):** implement packet/MAC ACK/queue/TTL/multi-origin trên GPU bằng Warp, benchmark parity vs CPU và upstream UavNetSim cùng interference/overlap/seed; remeasure peak VRAM T4×2; train hội tụ nhiều seeds; compare delivery%, discovery%, coverage/AUC, AoI/GCS, PDR, timeout, queue/collision & CI. Ngay cả CPU custom đã sửa CCA cũng không đồng nhất với channel simulator upstream do khác overlap và SionnaRT.
