# Kiểm định độc lập v6.6: logic nghiên cứu, truyền thông, map, reward và Warp

**Ngày:** 2026-11-25. **Đầu vào:** ZIP v6.5 do người dùng cung cấp. **Đầu ra:** source ZIP độc lập, không thay đổi repository người dùng. **Nguyên tắc:** RED→GREEN→regression; giữ nguồn lịch sử; đúng hợp đồng <=500 dòng/file; mọi chỉ số phân biệt phép đo trực tiếp, ước tính, và giả thuyết. Các nhận định về paper không xác nhận thứ hạng Q1/A* nếu không có chỉ mục năm tương ứng.

## 1. Kết luận khoa học, ưu tiên theo tác động

| Mức | Thành phần | Cái sai hoặc xung đột đang có | Quyết định v6.6 | Kiểm tra tiếp trước paper |
|---|---|---|---|---|
| P0 | Warp v6.5 `WarpCompressedRadioNetwork` | Chỉ cấp phát SoA; không có `step()` packet/MAC, không có dynamic end-to-end packet/ACK. 379.848 byte/env không đại diện cho full environment. | Giữ mã để phát triển nhưng **GPU gate BLOCKED**, không train hay nói GPU parity PASS. | Port packet-state machine và ACK/TTL/provenance; compare per-event trace với CPU; chạy T4. |
| P0 | Multi-origin watermark trên Warp | `ack_wm_origin_{pos,neg,ts}[src,dst,origin,cell]` chỉ tồn tại CPU; GPU compressed schema thiếu các trường. | Dự toán thêm 806.400 byte/env ở 6 UAV/20×20; không tuyên bố giảm VRAM 40,3% cho đầy đủ data-mule. | Thiết kế sparse provenance/ACK, kiểm tra equivalence, peak allocation đo trên T4. |
| P0 | `resolve_compact3_actions_np` | Chọn peer gần nhất dù peer không cần dữ liệu, khác CPU `_resolve_comm_dest`. | Thêm đối số `dirty_dest[e,src,dst]` và `can_tx[e,src]`; chỉ dùng kết quả như peer parity khi cung cấp đủ cả hai. Nhánh cũ diagnostic tương thích ngược. | Gắn các mảng thực này vào Warp-kernel *sau* khi implement network step. |
| P1 | Map scaling 2–3 km | Tòa nhà chỉ tăng radius theo √scale, GCS và spawn ở vị trí 1km, không phải normalized canonical geometry. | Tỷ lệ hóa cùng hệ tọa độ XY và bán kính, giữ bit-exact ở 1 km. | Báo cáo hai loại so sánh: geometry-similar và RF difficulty; không mặc định task ngang độ khó. |
| P1 | `scenario_7radio.yaml` vs `scenario_v65_datamule.yaml` | `w_p2p=0.10` + no-data-mule đối lập `w_p2p=0`, data-mule on; CPU gate mặc định chỉ chạy profile cũ. | Tạo profile peer-ON/OFF matched 1/2km, thêm `--cpu-gate-datamule`. | Chạy trained policy cùng seeds và episode budget, confidence interval. |
| P1 | UavNetSim PHY/MAC equivalence | Chỉ chung vài hằng PHY; CPU Q64, v65 Warp Q32, upstream `MAX_QUEUE_SIZE=200`; channel model/route khác. | Profile reference Q200; giữ smoke Q64; GPU gate ghi rõ mismatch. | Replay event log nguồn-vs-mô phỏng và load-sensitive collisions/ACK/backoff/TTL. |
| P1 | Bằng chứng data-mule | Trước đó kết luận peer hữu ích từ số ô mất 1-hop; không chứng minh packet chuyển tiếp thật. | Test 2-contact topology: nguồn → peer → GCS, peer ON nhận 1 gói còn OFF nhận 0. **Đây là test khả năng, không phải kết quả học RL**. | Chạy policy learned ablation và kiểm tra diện tích giao hội liên kết thật. |
| P2 | `NO_SEND` và `SHARE_PEERS` | RF cost thấp + `lambda_b=0`, `w_p2p=0` có nguy cơ action suy biến trong traffic thấp. | **Chưa xóa** trước khi đo: bỏ mất điều kiện truyền thông có thể che giấu bottleneck; bật/tắt peer để có causal ablation. | PDR, GCS delivery/AoI, delay, retry, collision, queue occupancy với offered-load sweep; nếu action thừa thì bỏ sau bằng chứng. |
| P2 | `confirm_th=0.90` | Khác threshold 0.99 từng được yêu cầu cho giả thuyết trước; một con số khác làm thay đổi precision/recall, không phải bug cú pháp. | Giữ nguyên trong hai nhánh ablation (không âm thầm đổi khoa học). | Calibrate ROC/false-confirm với Pd=0.85, Pf=0.03, then compare 0.90 vs 0.99. |
| P2 | AoI reward | `r_gcs_aoi` chỉ được trả **một lần** lúc target true-positive được GCS xác nhận (`gcs_tp_latch`), không reward cho refresh sau đó. | Không đổi thưởng nếu chưa có mục tiêu long-term AoI; ghi rõ chỉ tối ưu freshness *at first confirmation*. | Nếu paper tuyên bố maintain freshness, sửa mục tiêu + tests and ablation. |
| P2 | Belief-map baseline | `paper_fixed mf=me=0.8`, `ds_fusion_mode=sum`, EWMA counts là phần kết hợp mô hình khác Zhang; không được gọi tái lập Zhang bit-exact. | Giữ fixed 0.8 baseline để so sánh; ghi rõ extension packet/AoI + provenance. | Ablate `paper_fixed` vs sensor calibration; inspect posterior, negative/false observations, FP reset. |
| P2 | Motion v_max15, 900s/180kJ | Hợp lý làm pilot, chưa chứng minh mô hình năng lượng động lực bay/ràng buộc phanh trong thử nghiệm benchmark. | Giữ physics config không thay đổi để tránh biến số. | Sweep battery/trajectory vs safety interventions; kiểm tra cruise power và GCS delivery latency. |
| P2 | Độ khó khi tăng map | 1 km:15 targets; nếu giữ 15 trên 2 km mật độ giảm 4 lần, không còn cùng density. | Ablation 2km đặt 60 targets, 40×40, 50m/cell. | Tăng theo diện tích đồng thời kiểm tra sensor radius/obstacle count/RF reach. |
| P2 | Static 7-radio kernel & locality | GPU PHY chứa 7-radio, 20×20, cell50m, canonical obstacle; CPU hỗ trợ N=1..30 nhưng GPU không. | GPU gate BLOCKED; không tuyên bố 6→8 zero-shot hay E4096 Warp parity. | Parameterize kernels plus independent CPU/GPU conformance tests. |

## 2. Số liệu bản đồ đã đo từ mã CPU v6.6

Điều kiện: 1 UAV tại tâm mỗi ô, alt=50m; GCS alt=15m ở (50,50)m * scale; fixed TX 0.1W, noise 4.389577e-13W, SNR>=6dB, CCA RX lock>=-82dBm; 5 hình trụ và 2 vùng Manhattan-blackout. Hàm riêng `scripts/maintenance/academic_scenario_audit.py` kiểm tra direct PHY; không chạy packet simulator hay đo relay.

| Kịch bản | Ô | Tỷ lệ không đạt SNR | Tỷ lệ không thể RX theo SNR+CCA | Ghi chú |
|---|---:|---:|---:|---|
| 1km, 20x20 | 400 | 161/400 = 40.25% | **161/400 = 40.25%** | blackout 50 ô, NLoS 137, giao 26 ô (không cộng trực tiếp) |
| 2km, 40x40 | 1600 | 635/1600 = 39.69% | **1092/1600 = 68.25%** | không đúng khẳng định cũ ">70% ở 2km" |
| 3km, 60x60 | 3600 | 2079/3600 = 57.75% | **2995/3600 = 83.19%** | mất direct link không đồng nghĩa mất multi-hop |

**Cảnh báo:** Ghép chung SNR/CCA có nghĩa *theo implementation CPU này*; upstream UavNetSim không nhất thiết dùng CCA như điều kiện lock RX. Điều kiện CCA nên được nghiệm chứng trực tiếp qua topology/trace UavNetSim (lý thuyết 802.11: CCA carrier sensing không tự động là receiver sensitivity). Không gọi phép quét này là PDR 60% hoặc bảo đảm data-mule có gain.

## 3. Thử nghiệm có thể lặp lại

`tests/test_v66_academic_logic_contract.py` có test 2-contact: ban đầu không có route GCS ở cả source và peer, source gửi cell42 tới peer, sau đó thay topo để peer tiếp xúc GCS; peer cầm origin0 gửi sang GCS. Kết quả trong packet simulation: `ledger_pos[GCS,origin0,cell42]` là 256 với true data-mule, 0 khi disabled. `data_delivered_gcs` là 1 vs 0. Việc **thay tọa độ trực tiếp** trong test này chỉ chứng minh radio custody khi contact xuất hiện, **không** là hành trình bay physically feasible (không flight speed/safety simulation).

CPU learner smoke với 1km/2km × data-mule/no-peer × MATD3/MASAC: cả 8 cấu hình chạy 4 transition, đúng 2D+compact3, fixed 0.1 W, mỗi thuật toán 1 optimizer update và số liệu hữu hạn. Những reward ở 4 transition **KHÔNG** dùng để kết luận sự vượt trội của thuật toán/policy hay tính thống kê.

Bộ 4 profile: `research/zhang_xue/configs/scenario_v66_{datamule,no_peer}_{1km,2km}.yaml`. Mỗi cặp chỉ khác flag `enable_peer_dissemination` nên có thể dùng same-seed evaluation sau training. Vì actor action mask thay đổi theo policy, cần đánh giá cả shared weights và trained-separately controls.

### Gate CPU
```bash
bash setup.sh --check-only
bash setup.sh --test
bash setup.sh --cpu-gate
bash setup.sh --cpu-gate-datamule
python scripts/maintenance/academic_scenario_audit.py
python scripts/maintenance/verify_source_manifests.py --check
```

### Gate GPU — chủ động FAIL
```bash
bash setup.sh --gpu-gate
# Expect exit code 3 until v65 has an actual packet-step/ACK/TTL/ledger implementation.
```

**Không dùng Kaggle account 2** khi GPU engine chưa implement; không đọc hoặc sao chép `key.txt`/secrets. Trước khi chạy T4x2 cần lắp đầy đủ event state machine (queue/multi-origin watermark/packet ACK/TTL), rework GPUSoA, kiểm tra bit-level parity bằng trace, rồi profiling runtime/VRAM 512→2048→4096 môi trường. Không coi việc import Warp hay tạo buffers là end-to-end training.

## 4. Paper và tài liệu đối chiếu (primary sources)

1. [UavNetSim official GitHub / config.py](https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/master/utils/config.py) — TX0.1W, CCA−82dBm, IEEE 802.11b 22MHz/2Mbps, thermal density −174dBm/Hz, NF7dB, Q200, 5 retries. Đây là **mặc định tại nguồn được đối chiếu**, không có bằng chứng custom CPU hoàn toàn tương đương UavNetSim SimPy/SionnaRT.
2. [Zhou et al., UavNetSim-v1 (2025)](https://arxiv.org/abs/2507.09852) — packet MAC, routing and simulator, benchmark context. Conference preprint **không tự động là Q1 journal**.
3. [Wei et al., An Accurate IEEE 802.11 Packet Delay Model for Single-Hop UAV Networks (2025), IEEE TVT](https://doi.org/10.1109/TVT.2025.3554551) — CSMA/CA, ACK, queue, partial connectivity, collisions matter to channel scheduling; do not assume airtime alone captures congestion.
4. Zhang et al. (2024), *Enhancing Multi-UAV Reconnaissance and Search Through Double Critic DDPG With Belief Probability Maps*, IEEE Transactions on Intelligent Vehicles, user-supplied PDF — m_f=m_e=0.8 baseline, DS fusion, exploration; paper assumes real-time global information sharing, unlike this packet custody + AoI extension.
5. [Xue et al. (2026), AoI- and DS-Enhanced Cooperative Search for Multi-UAV](https://doi.org/10.3390/electronics15091875) — freshness-weighted belief/evidence fusion, blackout and buffering; our custom EWMA/UQ8.8 packet design is **not** a verbatim implementation.
6. [SchedNet, ICLR 2019](https://arxiv.org/abs/1902.01554) và [IC3Net, ICLR 2019](https://arxiv.org/abs/1812.09755) — communication scheduling/gating useful only when policy actually pays a cost or creates congestion/coordination benefit; avoid branding ad-hoc `NO_SEND` as a contribution.

**Nguồn học thuật chưa truy cập được trong lượt này:** Consensus thông báo hết quota tháng (reset 2026-12-01); Liner yêu cầu phương thức thanh toán. Không sử dụng kết luận chưa xác minh từ hai connector này. Q1/A* classification cần xác minh bằng SCImago/JCR năm xuất bản trước khi viết paper, không suy từ tên tạp chí.

## 5. Checklist gate trước khi train / viết kết quả chính thức

- [x] Giữ mọi source text <=500 dòng/file, config/manifest kiểm tra được khi giải nén không có Git.
- [x] Tái lập unit test hiện tại; giữ hành vi diagnostic cũ của compact3; thêm test receiver-specific dirty.
- [x] Data-mule multi-origin P2P→GCS functional test có no-mule control.
- [x] 1/2/3 km PHY survey phân biệt SNR và CPU RX lock (CCA).
- [x] CPU MATD3/MASAC 4-transition smoke cùng đúng main data-mule profile.
- [x] Ablation profiles matched 1km/2km, giữ seed và density.
- [ ] UavNetSim CPU event parity đối chiếu cùng topology/traffic/seed.
- [ ] GPU `WarpCompressedRadioNetwork.step()`/ACK/multi-origin/queue/TTL, với trace parity.
- [ ] CUDA T4×2 memory/speed benchmark (allocated peak + throughput + replay + training).
- [ ] Converged-policy ablations với 30–100 eval episodes/seeds; PDR/AoI/delay/delivery/coverage/AUC và 95% CI.
- [ ] Căn chỉnh threshold confirm=0.90/0.99 theo false-positive và paper objective.
