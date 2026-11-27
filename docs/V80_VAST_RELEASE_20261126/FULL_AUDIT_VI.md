# KIỂM TOÁN KỸ THUẬT V80 — VAST GPU — 26/11/2026

## Mục đích và giới hạn kết luận
Báo cáo này đối chiếu V80 do người dùng cung cấp với mã nguồn đang chạy trên Vast.ai, có thử GPU thật trên RTX3060 CUDA:1, kiểm thử CPU, test RED/GREEN, trace kernel, và theo dõi MATD3/MASAC. Mục tiêu là tìm logic conflict rồi sửa từng lỗi có bằng chứng. Không đánh đồng chạy được với hội tụ, bài kiểm thử riêng kinematic với CPU–Warp toàn nhiệm vụ, hay custom Warp MAC với UavNetSim SimPy.

Trạng thái cuối: V80 đã sửa **8 lỗi runtime/logic xác nhận**, với **402 tests passed, 5 skipped, 100 subtests**; **6 GPU regression tests passed**; project policy 0 vi phạm; **284 root + 121 research SHA256 checksums PASS**. Còn **nhiều Gate khoa học CHƯA được chứng minh**.

## Scenario thực chạy, không được nhầm với dự án 3km/3D khác
- File cấu hình: research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml
- UAV=6, GCS=1; khu vực 1000 × 1000 m; lưới DS 20 × 20, ô 50m; điều khiển 2D tại độ cao cố định 50m; 15 target tĩnh, 5 tòa nhà trụ và 2 vùng blackout.
- v_max=15m/s; a_max=5m/s², dt=1s; khoảng cách UAV tối thiểu 50m; mặc định horizon=900 bước. Vị trí khởi tạo 90+65*u theo X và 110+30*(u%2) theo Y.
- Sensor Pd=.85, Pf=.03; DS m_f=m_e=.8, confirm=.90, eta=.02; evidence count fixed point UQ8.8; đây là baseline chưa hiệu chỉnh theo Bayes.
- TX=0.1W; noise=4.389577e-13W; ngưỡng SNR=6dB; deterministic ray-cylinder building attenuation 12dB/building, custom Warp CSMA/CA. Không gọi là UavNetSim gốc hay đã được ITU chứng nhận.
- Policy decentralized actor + centralized critic MATD3/MASAC; E4096/B8192 chỉ smoke 3 bước, 2 optimizer updates. GPU0 đang bị chiếm ~11.7GB; mọi CUDA test chạy trên GPU1.

## Tám sửa lỗi có test trước/sau và lý do
1) Torch→Warp dtype: V80 gốc truyền torch.int32 vào wp.from_torch, bị RuntimeError khi chạy GPU. Đổi đúng kiểu wp.int32. Log 03 FAIL; 09 PASS.
2) Kernel code generation: shield 66 candidate + backward codegen mặc định làm compile >85–120s. Khai báo wp.set_module_options(enable_backward=False,max_unroll=2), vì physics transition không chạy backward. Sau sửa module compile 2.57s (trace 06).
3) Sensing ABI mismatch: launch _sense_visible_v79 thiếu failed, gây 13 args khi kernel cần14. Đã truyền đủ và theo đúng thứ tự (trace 06/07).
4) Termination buffer key: V80 dùng terminal_reason, trong khi V79 alloc termination_reason. Đổi toàn bộ tham chiếu tương ứng (trace 07/08).
5) Kinematic clamp order: GPU cap tốc độ trước gia tốc, CPU ngược lại. Chuyển sang cap gia tốc trước, giới hạn tốc độ sau. Cùng input 4world x16 bước, max abs velocity delta từ 8.228706m/s xuống 1.416596e-6m/s (log 15/16). Không tuyên bố toàn trạng thái luôn tương đương.
6) dt_s: tham số cho phép .5 mặc dù network step cố định 1e6us; thêm fail-closed validation chỉ hỗ trợ dt_s=1s đến khi refactor đồng hồ. Unit TDD RED 17, GREEN 18.
7) Network overflow: trước đó chỉ kiểm tra mỗi 10 steps, có thể bị auto-reset xóa. Check mỗi bước trước reset, test tiêm sự kiện overflow ở step1. Log 19/20; E4096 final guard 3/3 steps.
8) P0 fail-closed position commit: mã gốc đánh dấu failed=1 nhưng vẫn update pos theo chuyển động không an toàn (test trước: x=8 -> x=17). Đã thêm early return của _move_v80 khi phys_fail, giữ nguyên pos và vel world đã kết thúc, tránh việc một safety warning thành state unsafe được commit. TDD RED 29, CPU PASS 32, CUDA PASS 34. Đây là rollback terminal, KHÔNG làm quỹ đạo ban đầu invalid thành an toàn và không thay thế feasible shield.

## Kết quả chạy đúng mã V80 đã sửa
| Test | MATD3 | MASAC | Ý nghĩa |
|---|---:|---:|---|
| E4096/B8192 horizon2, 3 vector steps | PASS, 2 updates | PASS, 2 updates | Chạy được 4.096 parallel world và replay/learner |
| E64/B128 horizon900, 300 steps | 299 updates | 299 updates | Stress ngắn, chưa học hội tụ |
| Episode safety terminated trong 300 steps | **38** | **3** | Cần nghiên cứu vì không đạt 0 |
| Cờ boundary | 35 | 1 | Cờ chồng nhau có thể xuất hiện |
| Cờ UAV-UAV | 6 | 2 | Không phải số va chạm thực ngoài đời |
| Target success (15/15) | 0 | 0 | Không có bằng chứng policy hoàn thành |
| Full suite CPU | 402 passed; 5 skipped; 100 subtests | chung code | Tập test software |
| New CUDA parity/fail-closed | 6 passed | chung code | Sự kiện/đối chứng cụ thể |

Trước vá fail-closed, V80 sau sửa clamp có MATD3 41, MASAC 3 safety terminal. Trước sửa clamp V80 từng ghi 1/1, nhưng đó KHÔNG là đại diện cho source final. V79 (cơ chế safety conservative khác) từng ghi MATD3 63, MASAC 117; **không so sánh 63→38 hay 117→3 là giảm tai nạn tương ứng**, vì tiêu chuẩn terminate V79 có braking envelope prediction còn V80 mặc định xét swept physical collision.

Bài đối chứng zero action 4world × 903 bước trước patch fail-closed: tại bước900 có 4 truncation, bước901 clear sau reset; 0 safety failure, 0 overflow. Đây là một đối chứng horizon/reset, KHÔNG chứng minh policy RL có thể vận hành 900 bước an toàn.

## Kiểm toán từng thành phần và các Conflict còn mở
**Kịch bản / Map:** 50m fixed altitude nghĩa là 2.5D mission (2D control + 3D building RF), không phải UAV 3D trajectory learning. Spawn gần nhau hơn dự phòng phanh cực đại (khoảng thêm 21.59m trong khi d_phanh=22.5m ở 15m/s khi phanh 5m/s²) — cần ablation spawn 72/100/120m và khả năng khởi tạo an toàn trong nhiều hướng.

**Safe RL shield:** 66 candidates và 3 vòng coordinate descent là cách xấp xỉ, không tự cung cấp chứng chỉ invariance. 38 và 3 terminated mới là minh chứng còn xung đột với policy actions. Bản vá commit freeze thế giới bị fail giúp trạng thái simulator không vượt thêm giới hạn sau khi phát hiện, nhưng mô hình vẫn cần invariant proof theo dynamics, giả định sensing và worst-case UAV-UAV. Không thay ngầm safe_d=50m hoặc tắt kết thúc do safety để làm đẹp số liệu. Cần CBF/QP/viability comparison với điều kiện vật lý phù hợp.

**Observation:** _observe_core_v79 có own pose/energy/queue/GCS, 10x10 semantic belief map; không đưa đầy đủ vị trí 5 peer và NFZ nguyên thủy vào actor. Có thể thêm thông tin từ gói HELLO (không lấy oracle future position), age/validity masks và NFZ relative feature rồi chạy ablation. Kiểm định Dec-POMDP.

**DS / AoI:** Chính sách hiện dùng mass fixed .8 và eta=.02 để làm phai count evidence của mục tiêu tĩnh. AoI tin gửi đến GCS KHÔNG tương đương mục tiêu đã tìm bị biến mất. ETA=0 hiện bị từ chối do nguy cơ bão hòa UQ8.8; cần thiết kế số đếm, priors, mass calibration Bayes, posterior map parity, tách freshness khỏi spatial belief. Không gán định lượng 2.01x overconfidence nếu chưa có test toán học cùng điều kiện.

**Reward:** _reward_v79 dùng local ledger để cộng loc và GCS ledger cho delivered. Đây là team-local information shaping, không được tự động gọi là policy oracle leakage vì reward trong CTDE có thể tập trung; tuy nhiên objective nghiên cứu về MẠNG cần ablation reward GCS-only/local/potential-based và khai báo liệu GCS có thấy mỗi detection không. Lambda_b=0, w_p2p=0 ở baseline, airtime có thể đã tốn pin nhưng không được reward thêm tiền kênh; chạy ablation có traffic/energy monitoring. Lệch raw critic_loss MASAC/MATD3 không tự động là bug, cần per-critic normalisation.

**PHY/MAC:** YAML noise 4.389577e-13W tương ứng gần −93.58dBm, khác ví dụ UavNetSim v1 ~4e-11W (−73.98dBm) ~19.60dB. Băng thông/noise figure khác nhau có thể là lựa chọn; phải đặt khớp khi benchmarking, không sửa mù quáng. Attenuation 12dB/building là deterministic ray-cylinder heuristic, không phải mô hình ITU P.2109 building entry loss tương đương. Chạy PDR/SINR/delay/AoI vs obstacle sweeps, packet event traces, queue/ACK/retry. Công bố chính xác "GPU-native testbed lấy cảm hứng từ UavNetSim".

**NVIDIA Warp:** E4096 world đồng thời và minibatch B8192 đã chạy được ở 1 GPU, nhưng không chứng minh chạy 2 GPU. GPU0 đang chiếm bộ nhớ; Warp CUDA P2P không khả dụng trên host. Host sync mỗi vector step thông qua .item() và JSONL có thể gây overhead; có thể vectorize diagnostics trên GPU, log ít hơn mỗi 50–100 steps nhưng PHẢI giữ overflow check mỗi step.

**Training/Scientific evaluation:** Chưa có 20k episodes, chưa có checkpoint resume, final held-out eval 50–100 episode, discovery/delivery/coverage/AUC, t-test/CI đa seed, target delivery time, hay khảo sát reward saturation toàn pipeline. E4096 3 steps chỉ xác nhận integration.

**Ablation conflict:** ETA0 YAML rejected vì UQ8.8 overflow; highnoise và commreward YAML chưa được GPU hỗ trợ do battery_termination_mode=any_depleted so với all_depleted. Chưa chạy các ablation này; không bypass validate.

## Tài liệu tham khảo có kiểm chứng
- Zhang et al. 2024, IEEE Transactions on Intelligent Vehicles, belief maps: https://ieeexplore.ieee.org/abstract/document/10388472/authors
- Wang, Ames, Egerstedt 2017, IEEE Transactions on Robotics, barrier certificates: https://ieeexplore.ieee.org/abstract/document/7857061
- Alshiekh et al. 2018 AAAI, RL Shielding: https://mlanthology.org/aaai/2018/alshiekh2018aaai-safe/
- Ng, Harada, Russell 1999 ICML, policy-invariant reward shaping: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- Yates et al. 2021 IEEE JSAC AoI survey: https://dspace.mit.edu/entities/publication/ca039de7-e561-4a0f-bd3a-b92d0c436020
- Zhou et al. UavNetSim v1, ICCC 2025 conference source: https://arxiv.org/abs/2507.09852 ; https://github.com/Zihao-Felix-Zhou/UavNetSim
- NVIDIA Warp module options documentation: https://nvidia.github.io/warp/latest/api_reference/_generated/warp.set_module_options.html
- ITU-R P.2109-2, BUILDING ENTRY LOSS: https://www.itu.int/rec/R-REC-P.2109-2-202308-I/en

Không phải mọi tài liệu trên là bài tạp chí Q1/A*. Zhang (IEEE T-IV) và Wang (IEEE T-RO) là tạp chí IEEE; Shielding AAAI là hội nghị, UavNetSim v1 là hội nghị ICCC 2025; ITU là khuyến nghị kỹ thuật. Không dùng "Q1/A*" như nhãn cho toàn bộ.

## Phương pháp xác minh từ file ZIP
Sau khi giải nén, chạy tools/verify_release.py để kiểm checksum các thành phần. Xem matrix ISSUES_CODE_LOG_SCIENCE.csv (24 issues) và CODE_EVIDENCE_WITH_LINE_NUMBERS.md để xem code nguyên văn kèm số dòng; xem evidence/raw_vast_logs cho toàn bộ log từng lần thất bại/sửa test. Dữ liệu Kaggle V77/V78 là ARCHIVE LỊCH SỬ khác source V80; không dùng làm bằng chứng V80 đã training hội tụ.

## Thứ tự tiếp theo (chưa triển khai)
1. P0 viability & adversarial action grids cho shield, làm sao luôn chọn được safe acceleration bounded action hoặc fail without hazard.
2. P0 environment CPU–Warp full parity có controlled packet clocks + sensor RNG.
3. P0 event parity với original SimPy UavNetSim + network traffic ablations.
4. P1 DS eta0 counter-safe & dynamic Bayes calibrations; reward GCS-only, local, traffic costs.
5. P1 long-running checkpoint/resume + held-out eval. Chỉ sau khi các gates khoa học đóng mới báo paper-ready.

Đây là bản báo cáo ghi nhận chứng cứ hiện có, không tuyên bố hết các conflict hay hoàn thành training.
