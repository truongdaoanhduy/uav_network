# Cơ sở khoa học, xung đột kịch bản và thứ tự triển khai Warp

Đọc cùng `V72_AUDIT_AND_FIXES_VI.md`. Các phương án thí nghiệm dưới đây là khuyến nghị có lý do, không phải điều kiện bắt buộc chung của mọi tạp chí/hội nghị. Không có kết luận hội tụ hay chất lượng Q1/A* từ một lần smoke.

## 1. Nguồn gốc đã đối chiếu và điều gì được phép suy ra

| Nguồn chính | Nội dung liên quan | Áp dụng vào dự án / điều không được đánh đồng |
|---|---|---|
| [Zhang et al., IEEE T-IV 2024](https://doi.org/10.1109/TIV.2024.3352581), PDF gốc người dùng đã lưu; §III, Eq6–16, §V.B | DS bắt đầu vacuous; BBA positive/negative; Eq16 dùng MAX counts; ví dụ mf=me=.8 và τ=.95 | `.8` có căn cứ baseline; provenance SUM là thiết kế khác. DCDDPG kết hợp local/central critic không phải twin-critic MATD3 |
| [Xue et al., Electronics 2026](https://www.mdpi.com/2079-9292/15/9/1875), §3.1.3–3.2, Eq9–15 | Freshness-weighted buffered evidence, global mass decay, hai điều kiện xác nhận; AoI penalty và uncertainty gain | EWMA count-power của dự án không phải bản tái lập toàn bộ công thức này. Fixed Pd/Pf và continuous velocity cũng khác paper |
| [Yates et al., IEEE JSAC 2021](https://arxiv.org/html/2007.08564v1), §I–II, Eq1 | Age tại nơi nhận và time-average age | Phân biệt event freshness bonus, current map age và time-average; nêu initial age cho ô chưa nhận |
| [Zeng, Xu, Zhang, IEEE TWC 2019](https://arxiv.org/abs/1804.02238), §II.B, Appendix A | Propulsion power level-flight theo tốc độ + communication energy | Mô hình tốc độ dùng được cho planar flight; acceleration extension và tham số airframe trong code cần công bố/hiệu chuẩn riêng |
| [UavNetSim source](https://github.com/Zihao-Felix-Zhou/UavNetSim/tree/c2703f24eb58cdb557a5f67ded5c7b0368a964ed), `utils/config.py`, `utils/ieee_802_11.py`, `mac/csma_ca.py`, `phy/phy.py` | PHY params, retry/backoff, packet queues, channel/energy implementation | Commit đã đọc cố định `c2703f24…`; trùng vài tham số không chứng minh simulator tương đương |
| [UavNetSim-v1 paper, ICCC2025/arXiv](https://arxiv.org/html/2507.09852v1) | Simulator các tầng mạng, MAC, mobility, energy, PDR/delay | Căn cứ xây test mạng độc lập; đây không phải bằng chứng paper Q1/A* |
| [TarMAC, ICML2019](https://proceedings.mlr.press/v97/das19a.html) | Học nội dung/đích message qua attention, multi-round | compact3 chọn lúc gửi/loại đích ở application, không tái lập attention protocol này |
| [IC3Net, ICLR2019 source](https://github.com/IC3Net/IC3Net) | Học cổng giao tiếp và message liên tục | Không gọi action PEER qua CSMA là IC3Net hoặc chứng minh ưu thế so với nó |
| [Isaac Gym, NVIDIA](https://arxiv.org/abs/2108.10470), abstract/full method | Physics và policy trên GPU, truyền buffer trực tiếp | Cơ sở tránh roundtrip CPU ở rollout; không cần đổi UAV simulator thành Isaac Gym |
| [Warp1.18 PyTorch interop](https://nvidia.github.io/warp/v1.18/user_guide/interoperability/pytorch.html) | to_torch/from_torch chia sẻ storage; CUDA stream/capture | Zero-copy view không thay thế việc port CPU logic; phải quản lý stream/lifetime |
| [Isaac Lab multi-GPU](https://docs.nvidia.com/isaaclab/features/multi-gpu-and-multi-node-training) | Distributed training và config theo rank | Tham khảo orchestration; không tự làm MATD3/MASAC hiện tại thành DDP |

Đã dùng công cụ truy xuất GitHub, Context7, Exa, Parallel, Octen, Firecrawl, alphaXiv, Undermind và tìm kiếm web ở phần phù hợp. alphaXiv không tìm được chính xác Zhang; Undermind nhận diện paper nhưng không tải được PDF Zhang, nên dùng bản PDF gốc đã lưu của người dùng. Consensus hết hạn mức; Liner báo yêu cầu thanh toán; không lặp lại hay mua dịch vụ. Đây là giới hạn truy xuất, không phải bằng chứng khoa học.

Chưa xác minh bảng xếp hạng hiện hành theo từng ngành/năm. Q1 là thuộc tính journal/category/year; A* thường dùng cho conference ranking, không phải chứng chỉ đúng code hoặc xếp hạng từng paper. Các nguồn trên được chọn theo nội dung gốc và relevance.

## 2. Những nhận xét trước đó cần đính chính

### DS decay không tương đương theo cách báo cáo cũ viết

Code giữ effective positive/negative counts với trọng số exponential, rồi đưa vào canonical count-power DS. Xét một positive, mf=.8 và age-weight w=.5:

- Code: `mT = 1 - (1 - .8)^.5 = .5527864`.
- Discount trực tiếp khối lượng DS: `mT = .5 * .8 = .4`.

Hai kết quả khác nhau. EWMA count aging có identity O(1) trước lượng tử hóa; không vì vậy mà toàn DS fusion giống discount mỗi measurement mass. V72 sửa docstring/claim, giữ thuật toán chính. Muốn source-faithful mass discount phải lưu lịch sử/tổng hợp phù hợp và kiểm tra sai số; đây là biến thể khoa học khác.

### “BBA phải lấy từ Bayes, threshold phải .99” không phải kết luận tự động

`sensor_likelihood_to_bba` dùng likelihood ratios, không phải Bayes posterior từ prior. Với Pd=.85/Pf=.03: mf=.964706, me=.845361. Nếu đổi main sang chế độ này mà giữ threshold.90, một false positive đơn lẻ có thể đủ xác nhận. Vì thế chưa đổi mặc định `.8/.8` hay `.90`.

Phương án: chạy matched variants `.90/.95/.99`, fixed BBA, likelihood-ratio BBA và capped likelihood-ratio BBA; đánh giá precision/recall/false-confirm cùng delivery/discovery. Threshold cao thường đổi trade-off miss/delay, không tự luôn tốt hơn.

Vacuous DS ban đầu là `(mT,mF,mU)=(0,0,1)`, không phải Bayesian prior `P(target)=0`. Đặt prior Bayes bằng0 sẽ làm việc cập nhật bằng Bayes thông thường bị kẹt tại0; không áp dụng nhầm yêu cầu “prior0” của nhánh cũ.

### AoI và reward cần chọn nhiệm vụ rõ ràng

AoI tại nơi nhận: `Δ_c(t)=t-u_c(t)`, `u_c(t)` là generation mới nhất thực sự đã nhận. Time average là tích phân hoặc xấp xỉ lấy mẫu theo thời gian. First-confirmation bonus vẫn là event shaping.

Main giữ objective tìm kiếm/giao kết quả. Option `continuous_aoi_penalty_w>0` nay dùng current map age; trước sửa dùng running average nên double-average và đổi trọng số qua thời gian. Không bật option này trong main một cách ngầm định.

Clamping local information gain về không âm có thể thưởng việc làm mới belief sau decay nhiều lần. Với nhiệm vụ persistent monitoring điều đó có thể có chủ ý; với static one-shot search cần ablation first-visit/min-ever uncertainty hoặc potential shaping có định nghĩa rõ. Không tự sửa reward objective chỉ vì thấy một chu kỳ có reward dương.

`per_target_norm` hiện chỉ scale một số positive terms; FP/hold/safety không tự có cùng chuẩn hóa. Muốn reward normalized toàn diện phải định nghĩa weights và matched experiment riêng, không so map sizes theo raw return.

### “Phải thêm MAPPO/QMIX mới được Q1/A*” là phát biểu quá mạnh

Communication controls hợp lý trước tiên: No-Peer, Always-Send, Periodic/Round-Robin, learned scheduling, và bật/tắt AoI/DS/MAC khi cùng nhiệm vụ. MAPPO là một đối chứng bổ sung hữu ích nếu câu hỏi nghiên cứu cần; QMIX cần xử lý khác cho continuous motors. Không thêm implementation thứ ba chưa kiểm định chỉ để có tên baseline.

## 3. Xung đột cấu hình và logic dễ gây kết luận sai

| So sánh | Vì sao bị nhiễu | Cách xử lý |
|---|---|---|
| v67 datamule vs no_peer cùng1km/2km; v70 cùng3km | Chỉ nên khác peer dissemination | Gate đã so effective config; giữ cặp matched và seed nhiệm vụ |
| v66 vs v67 | any_depleted vs all_depleted đổi mission termination | Không ghép thành cùng ablation |
| scenario_7radio vs main | w_p2p=.1, Data-Mule off và nhiều legacy defaults | Chỉ coi là profile lịch sử/smoke |
| research_multihop_ablation vs reference | CCA−70 và noise4e−11 khác CCA−82/noise4.3896e−13 | Ghi là RF-stress variant |
| CPU vs Warp | Queue64/200 vs32; nhiều pending vs một pending/đích; RNG/dedup/selection khác | Pin cùng assumptions hoặc định danh protocol khác trước so throughput/reward |
| 1 vs2 vs3km | Target density15/km² giữ, UAV density6→1.5→.667; radio/travel/FoV tương đối đổi | Gọi area stress; báo từng metric và scale riêng |
| Pool observation | Grid20/40/60 đều nén10×10 | Physical cell vẫn50m nhưng mỗi pooled bin rộng100/200/300m; kiểm tra mất chi tiết |
| Old3D memory vs source mới | Altitude-sensitive sensor/TX/3km/CBF của nhánh cũ không còn main | Không tái đưa thông số cũ vào nhánh này |
| Confirm và FoV | Measurement phải có visibility tại thời điểm sensing; GCS nhận muộn không cần UAV còn trong FoV | Không thêm điều kiện FoV-at-GCS-confirm từ nhánh cũ |
| Fixed TX head và MASAC | Motor2D phải giữ vy; entropy−2 và replay masks tương ứng | Giữ algorithm semantics, test action adapter; không thêm RF head ngầm |
| Central safety shield | Shield biết positions/buildings thật dù actor chỉ local | Nêu giả định tracking/safety service, test local-sensing variant nếu muốn execution tự trị hoàn toàn |

Các file YAML effective nằm tại `outputs/qa_evidence_v72/effective_configs/`. Hình `figures/scenario_maps.png` dùng GT chỉ cho audit, không phải actor input hoặc kết quả policy.

Pin/RF: main hover model+avionics=183.49W, ideal endurance chưa RF≈981s; ở15m/s≈153.55W. Đây là tính toán từ tham số code, không số đo máy bay. PA fixed .1W TX dùng1W DC trong budget khi phát; RX/idle circuitry, fading calibration, battery reserve và dynamic turn costs vẫn cần kiểm tra nếu nhắm triển khai thật.

## 4. Tại sao Warp4096 env hiện chưa sẵn sàng

Hiện radio có Warp kernels, nhưng environment state, motion/shield, multi-cell sensing, belief features, reward/done/reset và RL adapter vẫn là NumPy/Python. `wp.to_torch()` cho một buffer không tự chuyển các phần này lên GPU. Runtime host methods còn tạo `wp.array` mới từ NumPy và có `.numpy()` trong debug guards/snapshot; đó là API kiểm định, chưa phải captured rollout API.

Nghiệm tối ưu cần đo; không mặc định graph capture hay DDP giải quyết mọi nút thắt. Một thread/env xử lý tuần tự7radio có thể hợp lý ở nhiều env; bottleneck cần profiler chứ không suy từ số vòng lặp.

| Cấu hình mỗi GPU | Network backing tại2048env | Diễn giải |
|---|---:|---|
| 6 UAV/grid20 | .918GiB | Chưa gồm full env, model/update, replay và overhead |
| 6 UAV/grid40 | 2.854GiB | Cần đo headroom thực |
| 6 UAV/grid60 | 6.081GiB | Replay/activation có thể chiếm phần còn lại; chưa được nói “chạy tốt” |
| 12 UAV/grid40 | 15.855GiB | Không vừa budget T4 usable≈15GiB, ngay cả network-only |
| 30 UAV/grid40 | 195.147GiB | Dense provenance watermark không khả thi trên T4 |

V72 backing default481141B/env, tăng455B vì hop dedup. `ack_origin_gen` scale gần `O(U²(U+1)C)`; phải tính theo schema và actual free VRAM. Hai T4 không cộng thành một address space32GiB. Các số trên là byte estimates, không NVML measurements.

## 5. Thứ tự tiếp theo và điều kiện thông qua

| Gate | Làm gì | Evidence cần có trước khi đi tiếp |
|---|---|---|
| G0 — source/CPU | Full pytest, manifests/policy, environment & learner smoke | Exit0, JUnit, effective YAML; V72 cung cấp |
| G1 — Warp CPU | Kernel execution, TTL/ACK/reset/fault tests, application fixture | Payload/count/gen đúng; V72 cung cấp; không gọi là event parity |
| G2 — CUDA nhỏ | 4env trên từngT4, kernel payload gate và generation/reset tests | CUDA logs tênGPU/driver, device-wide memory, không fault; hiện BLOCKED |
| G3 — packet fidelity | Định nghĩa trace schema: time, TX/RX, packet identity, seq, origin/cell/gen, ACK, retry, queue, expiry | Controlled fixtures: direct/multihop/no-route→contact/lostACK/hidden-terminal/blackout/queue-full. Pin RNG/event tie-break hoặc so phân phối với sai số đăng ký trước |
| G4 — full device env | Port motion, swept safety, FoV+Pd/Pf, DS aging, reward, obs/masks, done/partial reset | So từng field theo seed/input đối với CPU; không host scan hàngnghìn env |
| G5 — RL nối thật | Device actions→env→obs/reward→replay→sample→MATD3/MASAC update→target→eval | 32env finite gradients, controls/masks đúng, TD3 delay/SAC entropy giữ nguyên, terminal/truncation đúng |
| G6 — scale/benchmark | 32→256→1024→2048/GPU, rồi phân chia4096 total | Nsight/timers, envsteps/s, H2D/D2H bytes, whole-device free/peak, replay/optimizer costs |
| G7 — hiệu quả nghiên cứu | Matched baselines, ≥nhiều training seed, paired eval layouts/seeds khác train | Curves theo total transitions, CI, FP/delivery/AoI/PDR/delay; chưa có trong V72 |

G4 nên có modules riêng cho `motion`, `sensing_belief`, `observation_reward`, `vector_environment`, cùng input/output device arrays đã preallocate. Warm compile trước capture; dùng cùng CUDA stream cho Warp và Torch, giữ storage lifetime; error flags phải được gate/mask rõ. RL ở đây không cần gradient đi xuyên simulator qua `wp.Tape`.

Multi-GPU có thể chọn một learner + rollout worker, hoặc sharded learner/replay theo rank. Nếu dùng NCCL/DDP, phải quyết định sync actor/critics/entropy/target/optimizer schedule và seed theo rank. Chỉ bọc một class bằng DDP không đủ. Đo một GPU trước khi chọn kiến trúc.

Không triển khai full GPU env vội trong bản sửa này khi protocol/reference còn khác và CUDA không kiểm thử được; một bản port chưa parity sẽ làm thay nhiệm vụ mà khó nhận ra. Các giao diện/gate hiện có là điểm khởi đầu kiểm định, không được dùng làm báo cáo đã hoàn thành G2–G7.

## 6. Kaggle account2 và bước chạy cụ thể

Đã đọc nhãn `account_02` trong `key.txt` được phép và gọi introspection bằng SDK Kaggle chính thức. API trả `active=false`; không gửi GPU job, không dùng account khác, không xuất token vào ZIP/log/commit. Hai bản khóa cũ chứa cùng token này.

Khi token account2 hoạt động: dùng script runner trong ZIP để thực hiện gate nhỏ trên `cuda:0` và `cuda:1`; metadata Kaggle dùng GPU T4×2 (`NvidiaTeslaT4`) theo [tài liệu Kaggle chính thức](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md). Runner không submit competition, không upload public dataset và không tự chạy training.

Lệnh tiếp theo sau khi giải nén/cài package:

```bash
python scripts/gpu/warp_packet_cuda_gate.py --device cuda:0 --envs 4 --uavs 6 --grid 20
python scripts/gpu/warp_packet_cuda_gate.py --device cuda:1 --envs 4 --uavs 6 --grid 20
python scripts/gpu/packet_contract_gate.py --device cuda:0 --uavs 6
```

Nếu gate fail thì giữ raw log, tái hiện trên số env nhỏ rồi sửa, không tăng2048/4096 trước. Nếu chỉ application fixture PASS thì vẫn chưa đóng G3 packet-event parity.
