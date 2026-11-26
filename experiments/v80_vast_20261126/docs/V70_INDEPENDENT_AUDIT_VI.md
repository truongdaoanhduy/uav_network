# Kiểm định độc lập v7.0 — từ ZIP v6.9, code thật, log thật, giới hạn khoa học

**Ngày kiểm định:** 25/11/2026. **Nguồn gốc:** ZIP người dùng tải lên `uav_v69_unified_warp_exec_qa_logs_20261125.zip`, SHA-256 `458c0b7bf1f1cb61df20e33104461e230c4877d485e00f81ab27ea6f3e56b1e3`. Không sửa repo GitHub mặc định và không đọc/đóng gói `key.txt`. ZIP mới giữ mã nguồn, test, QA, tài liệu và lịch sử báo cáo; không bao gồm CUDA results chưa chạy.

## 1. Phán quyết từng nhận định v6.9

| Nhận định trong v6.9 | Phán quyết độc lập | Bằng chứng và giới hạn |
|---|---|---|
| v6.9 có mã Warp reset/PHY/sense/inject/step | **Có tên hàm và source** | `gpu/runtime_v67.py`, `gpu/kernels/*_v67.py`; chưa chứng minh CUDA kernel compile/run. |
| `NumPySoARefV67` kiểm chứng Warp packet parity | **Sai nếu hiểu packet-event parity** | `inject_and_deliver_ideal()` truyền lý tưởng trực tiếp, không thực thi CSMA/CA, collision, ACK timeout, PHY, routing. Chỉ chứng minh luồng dữ liệu lý tưởng 2 contact. |
| `ack_origin_gen` nhỏ hơn naive 4D 75% | **Đúng về phép đếm byte** | 6×7×6×400×2 = 201.600 B/env; naive pos/neg/ts 806.400 B/env; tổng SoA schema = 480.686 B/env. Đây là **ước tính backing store**, không phải measured VRAM. |
| Watermark `gen` tương đương 100% CPU trong EWMA | **Có điều kiện, chưa chứng minh tổng quát** | Không phân biệt hai snapshot cùng `gen` nhưng khác effective counts; CPU watermark cũng giữ pos/neg/ts. Muốn chính xác cần sequence tăng mỗi update hoặc bất biến một mẫu/cell/generation, cùng chính sách ACK. |
| Warp có thể train 4096 env T4×2 | **Chưa** | Source Warp mới chỉ được gọi tại preflight/test; loop MATD3/MASAC vẫn dùng CPU packet env. Không có 4096 parallel actor/reward/replay/optimizer/eval Warp. |
| UavNetSim v1 tương đương hệ thống này | **Chưa** | CPU là hiện thực tuỳ biến các ý tưởng MAC/routing; Warp là port riêng, Q=32 (so với UavNetSim `MAX_QUEUE_SIZE=200`). Không có trace so từng event và Sionna RT đầy đủ. |
| B17 giải mã UQ8.8, B18 sửa quan sát, B09 metrics | **CPU tests hỗ trợ** | Các regression v6.9 có kiểm tra và fresh pytest có PASS; chưa đủ để tuyên bố cải thiện policy sau huấn luyện. |

**Phát hiện P0 mới đã sửa source (CUDA chưa xác minh):** (1) `_finish_tx_v67` xoá `rx_lock` trước `_deliver_data_v67`, rồi tra sender bằng `-1` gây đọc sai packet; nay truyền `tx_node` rõ ràng; (2) mặc định RX lock dùng CCA dù CPU dùng SNR; nay `rx_lock_on_cca=False`; (3) nhiều SEND liên tiếp ghi đè pending E2E cho cùng destination; nay giữ slot tới ACK/expiry; (4) kernel event cap overflow bị bỏ qua trong host; nay raise và chặn transition. Các regression dùng kiểm tra cấu trúc source cho kernel khi thiếu Warp; **không phải CUDA pass**.

**Các sửa nhỏ có test thực:** thiếu `import numpy as np` trong `kernels/reset_v67.py`; NumPy reference reset để lại ledger/trạng thái cũ; validate đầu vào cell ID, detected, action, PHY, time; memory preflight fail-closed; kiểm tra cặp YAML theo **effective defaults**. `wp.atomic_add` chuyển từ ternary array expression sang nhánh tĩnh để giảm rủi ro biên dịch.

## 2. Thành phần kịch bản thực tế

| Thành phần | v6.9/v7.0 thật sự có gì | Chưa đúng hoặc cần làm rõ |
|---|---|---|
| **Map** | Bản đồ 2D, lưới 50m/cell; 1km=20×20, 2km=40×40, 3km stress=60×60 | Không phải 3D occupancy-map; thêm lưới làm tăng kích thước dữ liệu, không tự làm bất biến difficulty. |
| **UAV** | 6 UAV và một GCS, policy điều khiển XY; altitude 50m cố định; giới hạn vận tốc 15m/s, gia tốc 5m/s² ở v67 | Không phải policy bay 3D tự do, không altitude-dependent Pd/Pf. Model shield cần đánh giá trajectory collision và khả năng khả thi (viability). |
| **Mục tiêu** | 15 static targets/1km², 60/4km², 135/9km²; Pd=.85, Pf=.03; confirm=.90 | Target density giữ 15/km² nhưng UAV/km² giảm 6→1.5→0.667; phải coi là stress khác nhau. False confirm vẫn cần precision/recall. |
| **Belief/DS** | `mf=me=.80`, unknown mass, EWMA AoI Q8.8, `ds_fusion_mode=sum`, optional sensor likelihood calibration | Chính v67 profile sử dụng `paper_fixed`, không phải Bayes-calibrated theo Pd/Pf; vacuous mass (0,0,1) không đồng nghĩa Bayesian prior=0. |
| **Reward** | local info gain, GCS TP/FP, first-confirm freshness AoI, blackout/safety/energy penalties; `w_p2p=0` trong cặp v67 | Không phải time-average AoI objective theo định nghĩa chuẩn; reward còn liên hệ ground truth (hợp lệ trong simulator để chấm reward, không phải oracle input actor). |
| **Mạng** | CSMA/CA, SNR/SINR + CCA, HELLO DV, multihop, MAC ACK, E2E ACK, TTL, dedup, Data-Mule, queue, RF energy | Chính sách MAC CPU và Warp chưa chứng minh tương đương; Q=32 GPU vs CPU 1..200 vs upstream 200, Warp scalar TX vs CPU per-UAV TX. |
| **RL** | MATD3 / MASAC shared core, adapter actor local, centralized training state; CPU 4-transition smoke kiểm tra backward | Chưa training hội tụ; v6.9 Warp chỉ là **radio subsystem**, không phải environment GPU đầy đủ. |
| **GPU/SoA** | `WarpCompressedRadioNetworkV67` với cấu trúc mảng nén, kernels nguồn | Cần CUDA compile, event trace parity, device-wide VRAM, benchmark từng bước, batch actor/reward/optimizer/training. |

## 3. YAML conflicts và nguyên tắc thực nghiệm

- `scenario_7radio.yaml` lịch sử bật `w_p2p=.10`, chưa bật Data-Mule; không gọi nó là cùng baseline với v67 có `w_p2p=0`.
- `scenario_v66_*` mặc định `battery_termination_mode=any_depleted` trong khi v67 chọn `all_depleted`: **khác điều kiện kết thúc**, so sánh cross-version gây confounding.
- Cặp `scenario_v67_{datamule,no_peer}_{1km,2km}.yaml` đã được chương trình `paired_scenario_gate.py` xác nhận chỉ khác `enable_peer_dissemination`. Cặp v70 3km giữ cách thiết kế đó nhưng là **stress profile mới**, chưa có kết quả học.
- 1km vs2km vs3km thay đổi không chỉ diện tích: số target 15/60/135, FOV 50m không đổi, số UAV 6 không đổi, mật độ UAV giảm 4×/9×; năng lượng, thời gian và connectivity không bất biến.
- `NO_SEND` và `SHARE_PEERS` action là **chính sách truyền có giới hạn**, không chứng minh là SchedNet/IC3Net vì hai nghiên cứu đó có cơ chế học gate/scheduling riêng.

## 4. Các lỗi khoa học/logic chưa được coi là FIXED

1. **P0 Full Warp rollout**: chưa chuyển collision shield, target sensing, DS fusion, mission reward, actor observation, route mask, termination, reset bất đồng bộ, batch replay/update vào GPU. Không thể báo 4096-env RL training thành công.
2. **P0 Packet-event parity**: thiếu trace theo `timestamp_us, tx, rx, power, CCA, rx_lock, SINR, retries, route, packet_seq, ACK, queue_len, TTL, radio_energy`; phải so CPU cùng tham số, seed và sự kiện, không chỉ cuối episode.
3. **P1 Memory 30 UAV**: `ack_origin_gen` O(U²(U+1)cells) làm 30 UAV×40²×2048 env ≈195,14 GiB *chỉ network SoA*, T4 không đủ; preflight đã chặn theo free VRAM trước cấp phát nhưng không giải quyết độ phức tạp.
4. **P1 Cont. AoI vs first confirm**: reward AoI hiện thưởng một lần khi nhận TP tại GCS, không tối ưu tuổi của mọi thông tin theo thời gian. Nếu thay objective, phải định nghĩa variant/ablation, không tự sửa reward khiến paper mục tiêu đổi.
5. **P1 DS belief calibration**: cần ablate `.90`/`.99` và Bayes-likelihood mapping; ngưỡng tương đương `k=2`/`k=3` **chỉ** khi các phép đo dương liên tiếp, BBA=.8 cố định, không decay và không đo âm. Ví dụ m(T)=1−0.2^k: k=2→.96, k=3→.992.
6. **P1 Sensor/mobility**: cao độ cố định không mô hình hoá đầy đủ tránh vật cản 3D/điều khiển z; xét riêng chống va chạm swept-path và energy termination.
7. **P1 Learned TX GPU**: CPU đã nhận power riêng từng UAV, Warp step còn một scalar TX. Không chạy GPU policy học TX trước khi có vectorized PHY/MAC/energy parity.
8. **P1 E2E watermark same-timestamp**: watermark chỉ theo generation; hai cập nhật cùng second với khác UQ8.8 counts không làm dirty. Đây là giới hạn thiết kế; không được mô tả là “100% tương đương”.
9. **P2 Normalization metrics**: PDR (packets delivered/enqueued) không thay thế discovery, precision, false-confirm, coverage AUC, communication energy, latency và return; phải ghi từng metric độc lập.
10. **P2 GPU allocator**: `torch.cuda.max_memory_allocated()` chỉ tính phần PyTorch quản lý, bỏ qua allocations của Warp; CUDA script mới dùng `torch.cuda.mem_get_info` *before/after* nhưng vẫn cần theo dõi tối đa theo thời gian / cuBLAS / kernel cache.

## 5. Nguồn đối chiếu có thể kiểm chứng

1. **Zhang et al., IEEE Transactions on Intelligent Vehicles 9(2), 2024**, *Enhancing Multi-UAV Reconnaissance and Search Through Double Critic DDPG With Belief Probability Maps*: DS belief + uncertain/target maps, DCDDPG, multi-objective reconnaissance. DOI/abstract: https://ieeexplore.ieee.org/abstract/document/10388472/authors . Đây **không** phải mặc định MATD3/MASAC + UavNetSim; cần nói rõ phần kế thừa và phần thêm mới.
2. **Xue et al., Electronics 15(9), 1875, 2026**, *AoI- and DS-Enhanced Cooperative Search for Multi-UAV Systems Under Spatially Structured Communication Constraints*: https://www.mdpi.com/2079-9292/15/9/1875 . Là paper liên quan reward/evidence freshness, không thay thế kiểm định event simulator.
3. **UavNetSim primary upstream**: https://github.com/Zihao-Felix-Zhou/UavNetSim ; config gốc tại https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/master/utils/config.py . Chứng cứ `MAX_QUEUE_SIZE=200`, `TRANSMITTING_POWER=0.1W`, CCA -82dBm, MAC/PHY khác nhau.
4. **NVIDIA Warp chính thức**: https://nvidia.github.io/warp/v1.18/user_guide/runtime.html . GPU kernel compile/launch, `wp.array` allocations và device-specific launch; CUDA results không suy ra từ source presence.
5. **SchedNet (ICLR 2019)**: https://arxiv.org/abs/1902.01554 ; **IC3Net (ICLR 2019)**: https://arxiv.org/abs/1812.09755 . Tham chiếu ablation learned communication chứ không phải code parity với chính sách heuristic compact_3.
6. **IEEE TNSM 2026 — Joint MARL and Message-Passing for Resilient Multi-UAV Networks**: https://ieeexplore.ieee.org/abstract/document/11322791 . Khác vấn đề tối ưu reliability/network learning, có ích khi định vị đóng góp ngành mạng.

**Quartile Q1/A\***: không suy ra chỉ từ tên IEEE/MDPI; cần xác minh venue, năm xét hạng và taxonomy cụ thể. Paper tốt không chứng minh code của đề tài đã được kiểm chứng.

## 6. Cách chạy các gate đúng nghĩa

```bash
bash setup.sh --check-only
bash setup.sh --qa --seed 44
python scripts/maintenance/paired_scenario_gate.py
python scripts/maintenance/verify_source_manifests.py --check
python scripts/gpu/warp_packet_cuda_gate.py --envs 4 --device cuda:0
```

Đọc `outputs/qa_evidence/{results.json,pytest_junit.xml,logs/*,SUMMARY.md,red_green/*}`. Các lệnh trên nếu không có CUDA sẽ cho **BLOCKED (exit 3)**. *Blockers không phải test FAIL hay test PASS.* Tất cả regression Warp CPU ở đây chỉ chứng minh các predicate/shape/cấu trúc source; phải chạy thật trên T4 để kết luận kernel compile và packet trace.
