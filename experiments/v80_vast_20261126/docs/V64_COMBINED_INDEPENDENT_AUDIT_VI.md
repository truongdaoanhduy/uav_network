# v6.4 — Hợp nhất project-wide + Q1-Clean: kiểm định độc lập

Ngày 25-11-2026. Đây là **một cây nguồn duy nhất**, không phải hai ZIP lồng vào nhau. `v6.2` là tổ tiên, `project-wide v6.3` là nền cấu trúc, các thay đổi Q1-Clean được cherry-pick/merge có chọn lọc vào các package mới. **Không chỉnh branch `main` trên GitHub, không có training T4 được chứng minh.**

## 1. Nguồn và loại bằng chứng

- Mã upstream: [UavNetSim `utils/config.py`](https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/master/utils/config.py), [`utils/ieee_802_11.py`](https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/master/utils/ieee_802_11.py), [README và luồng SimPy](https://github.com/Zihao-Felix-Zhou/UavNetSim), [Zhou et al., 2025, arXiv 2507.09852](https://arxiv.org/abs/2507.09852).
- Nguồn học thuật tham khảo: [Zhang et al., IEEE Transactions on Intelligent Vehicles (2024)](https://www.semanticscholar.org/paper/Enhancing-Multi-UAV-Reconnaissance-and-Search-DDPG-Zhang-Lin/c3962200ccb4224ae66a28a26f9bd424f4e1a7f9), [Xue et al., Electronics (2026)](https://www.mdpi.com/2079-9292/15/9/1875), [Zeng et al., rotary-wing UAV energy](https://arxiv.org/abs/1804.02238), [Warp Memory Management](https://nvidia.github.io/warp/latest/user_guide/execution_and_performance/memory_management.html).
- Những số dưới đây về code đến từ kiểm tra **file nguồn và bài test CPU**, không phải số liệu huấn luyện; công thức truyền là mô hình đơn giản hóa đang có trong code. Không coi tên journal của nguồn là bằng chứng hệ thống này đạt chuẩn Q1/A*.

## 2. Phán quyết đối với những phát biểu trong hai ZIP v6.3

| Phát biểu đã nhận | Phán quyết dựa trên code | Xử lý v6.4 |
|---|---|---|
| `P_tx=0.1 W` khớp default UavNetSim | **Đúng về thông số** | Chính sách main Q1 cố định 0.1 W; không cho học RF head. RF learned là mã ablation/legacy, không phải run chính. |
| `CCA=-70 dBm` khớp UavNetSim | **Sai**: upstream `CCA_THRESHOLD_DBM=-82.0`. | Cấu hình chính và reference về `-82 dBm`. `-70 dBm` chỉ tồn tại trong ablation đặt tên riêng. |
| Nhiễu `4e-11 W` là chuẩn 802.11b upstream | **Sai**: upstream tính `-174 dBm/Hz + 10log10(22MHz) + 7dB` = khoảng **−93.58 dBm**, tức **4.39e-13 W**. | Dùng cùng phép tính trong reference/Q1; bản `4e-11` chỉ được dùng cho thí nghiệm khác biệt đã gắn nhãn. |
| LoS truyền được đúng `249.1 m` ở 0.1 W | **Sai trong profile PHY chuẩn**: với Friis trong repo, 1 km LoS trống vẫn > ngưỡng SNR và CCA. | Có regression vật lý test 1 km. Không suy diễn multi-hop từ con số 249 m. |
| Chỉ `continuous_dim=2` thì train MATD3/MASAC sẵn sàng | **Sai trước khi vá**: `canonical_continuous` xóa `v_y` và MASAC bỏ log-prob của motor Y. | Sửa end-to-end actor output, masks, critic action, entropy; CPU gate chạy 2 thuật toán với RF cố định. |
| Actor 3 action thì `zero-shot` qua N-UAV chắc chắn hoạt động | **Chưa chứng minh**: chiều joint critic là `N*518`; policy independent / identity khác `N`. | Trên CPU có thể tạo mô hình cho N mới; không được quảng bá **chuyển một checkpoint** giữa N mà chưa có test load và rollout. |
| Compact layout GPU giảm VRAM ~40% | **Chỉ trên giấy**: `gpu/runtime.py` hiện đọc `gpu/layout.py` (635931 byte/env), không gọi `make_shapes_v63`. | `setup.sh --gpu-gate` fail-closed. Không dùng ngân sách VRAM của layout chưa active làm kết quả. |
| Bản CPU packet MAC đã 'bằng' UavNetSim | **Chưa có parity event trace** (SimPy logic, queue, routing, traffic, collision, retransmission không so 1-1). | Chỉ nói **tham số 802.11b đồng bộ một phần**; còn gate packet-trace. |
| `SHARE_PEERS` là data-mule relay evidence | **Không đúng**: `get_dirty_entries_non_oracle` chỉ đọc ledger của nguồn `src`; các peer không tái gửi dữ liệu do người khác thu. | Giữ peer-sharing để thử truyền tin phân tán, thêm config tắt + mask action/zero reward để ablation, không gọi là data mule. |
| Lưới 40×40/60×60 = benchmark quy mô lớn tương đương | **Chưa hợp lệ về độ khó**: 5 building và hai vùng blackout vẫn ở tọa độ tuyệt đối cũ, GCS cố định. | Mở rộng CPU bookkeeping/packet codec, nhưng benchmark diện tích lớn chỉ là **sensitivity**, không suy luận fair-scaling trước khi thiết kế spatial densities mới. |
| Bỏ `r_airtime`, hiệu chỉnh `m_f=m_e=0.8`, bỏ accel power | **Hợp lý cho baseline tối giản**, nhưng không phải định lý. | Baseline chính dùng `lambda_b=0`, `paper_fixed`, `enable_accel_energy=false`; giữ mã experiment tùy chọn, cần ablation. |

## 3. Lý do số 249 m không đáng tin và mâu thuẫn vật lý gốc

Từ mã `zhang_xue/communication/common.py` (đường truyền LoS đơn giản):

- `g(d) = (c/(4π f))² / d²` với `f=2.4 GHz`;
- `P_rx=0.1 * g(d)`, `N = 4.389577e-13 W`, ngưỡng SINR `6 dB`;
- tại `d=1000 m`, `P_rx≈9.88e-12 W`, **cao hơn** ngưỡng CCA `−82 dBm≈6.31e-12 W` và ngưỡng giải mã `N*10^(6/10)≈1.75e-12 W` (chưa tính NLoS/interference).

Trong **chính mô hình CPU này**, đường truyền LoS tốt giữa hai điểm xa có thể thành direct-link. Để chứng minh relay thực sự cần thiết, phải quan sát **route length distribution, direct-GCS fraction, packet-delivery ratio, average hops, delivery latency/AoI, collisions, NLoS outage** trên cấu hình thực — không tự nâng nhiễu hoặc hạ CCA rồi gọi đó là chuẩn upstream.

## 4. Cấu trúc và config sau hợp nhất

- `uav_network/algorithms/matd3`, `/masac`, `/baselines`, `/core`, `/configuration`: hyperparameters, encoder/critic, gradient/replay trong folder chuyên trách.
- `research/zhang_xue/src/zhang_xue/{environment,communication,belief,mobility,gpu/kernels,learners,compat}`: không trộn nguồn phiên bản phẳng lại với production.
- `configs/learners/{matd3,masac}_cpu_gate.yaml`: dùng `continuous_dim=2`, `network.power_control=fixed`, `network.destination_mode=compact_3`, N=6. Legacy learned TX code không phải baseline.
- `research/zhang_xue/configs/scenario_7radio.yaml`: baseline Q1 1 km, N=6, 20x20, 15 targets, 0.1 W, CCA=-82 dBm, BW22MHz-equivalent noise; 900 s, e_max=180 kJ.
- `research/zhang_xue/configs/uavnetsim_reference.yaml`: scenario đầy đủ với các thông số PHY/MAC 802.11b upstream **được chọn**; không phải UavNetSim thực thi.
- `research/zhang_xue/configs/research_multihop_ablation.yaml`: thay CCA/nhiễu chủ ý, KHÔNG còn được gọi là upstream reference.
- `research/zhang_xue/configs/ablation_no_peer.yaml`: giữ action width 3, nhưng khóa `SHARE_PEERS` bằng mask và không thưởng peer; chạy với cùng seeds để đánh giá giá trị thực.
- `setup.sh`: cài đặt riêng khi yêu cầu, chuẩn bị dữ liệu seed ngoài Git, policy/test, CPU gate; GPU gate chặn khi Warp Q1 chưa được nối.

## 5. Gates chính xác và các blocker công bố paper

| Gate | Nội dung | Tiêu chuẩn |
|---|---|---|
| G0 | Config hợp lệ, parameter match, 500-line policy, secret/artifact gate, source checksum | PASS trên cây source cuối trước khi phát hành |
| G1 | CPU environmental transition, TX fixed, MASK, collision safety smoke, DS/AoI/packet codec | Test assertions phải PASS trên bộ version hợp nhất |
| G2 | MATD3/MASAC: N=6, 2D XY, 3 route action, real optimizer step | `bash setup.sh --cpu-gate` cho kết quả finite với weight update count >=1 |
| G3 | UavNetSim parity: trace packet queue/backoff/ACK/retransmit/drop ở cùng source traffic, topology, seed | **CHƯA**; đối chiếu `UavNetSim` gốc thay vì chỉ đối chiếu thông số |
| G4 | CPU/Warp parity belief, EWMA, routing, MAC và action với N=6/8, grid20/40 | **CHƯA**; active GPU runtime hardcoded 6 UAV/7 radio, shape cũ |
| G5 | T4×2, 2048 env/GPU, VRAM peak + p50/p95 steps/sec, GPU grads/eval | **CHƯA**; không khởi training trước G3/G4 |
| G6 | Network ablations (`NO_PEER`, `NO_NETWORK`, `NO_AOI`, fixed vs calibrated belief), 30 seeds/eval fairness | **CHƯA**; phải có control experimental evidence mới kết luận đóng góp |

**Vấn đề công bố quan trọng:** Xuất hiện *intervention* bằng Safety Shield, 3D sensing occlusion, mạng PHY đơn giản và chế độ belief DS có thể gây lệch so Zhang và UavNetSim. Cần công bố thành phần tự sửa, thí nghiệm đối chứng, giới hạn prior, tỷ lệ success/discovery/delivery/AoI cùng seeds. Đừng gộp kết quả benchmark 1km 15 target vào benchmark cũ 3km 35 target.

## 6. Chạy kiểm định trong ZIP

```bash
bash setup.sh --dry-run
bash setup.sh --check-only
bash setup.sh --prepare-data --seed 44 --episodes 4
bash setup.sh --test
bash setup.sh --cpu-gate
# Cố ý BLOCKED cho đến khi Q1 2D+compact3 CPU/GPU parity thật sự hoạt động:
bash setup.sh --gpu-gate
```

Tại folder gốc, thử số UAV mới **trên CPU**, phải cập nhật **cả hai** config:

```bash
PYTHONPATH=.:research/zhang_xue/src python research/zhang_xue/scripts/v61_cpu_actor_learning_gate.py \
  --algorithm matd3 --transitions 4 \
  --learner-config configs/learners/matd3_cpu_gate.yaml --learner-set world.num_uavs=8 \
  --scenario-config research/zhang_xue/configs/scenario_7radio.yaml --scenario-set num_uavs=8
```

`SHA256_MANIFEST.json` và `SOURCE_SHA256_MANIFEST.json` giúp so byte từng file nguồn trong ZIP. **Không** đẩy checkpoints/keys/Kaggle account sang Git hoặc ZIP. Đừng lấy một ``pytest PASS`` làm chứng nhận GPU, Q1 hoặc parity simulator.
