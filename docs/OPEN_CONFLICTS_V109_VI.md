# V109 — DANH SÁCH XUNG ĐỘT VÀ CỔNG CHƯA ĐẠT (BỔ SUNG 28/11/2026)

**TRẠNG THÁI TỔNG:** `NOT SCENARIO_FROZEN`. Đây là phụ lục KIỂM TOÁN, KHÔNG phải bản sửa thuật toán hay kết quả huấn luyện. Toàn bộ V109 code và evidence giữ nguyên so với ZIP gốc; chỉ thêm phụ lục xung đột và ma trận bản bổ sung. Không push Git, không đóng các scientific gates bằng giả định.

**Phạm vi:** DS-only V109 active, 6 UAV, 15 static targets, 1 km², 20×20 grid, 900×1 s, fixed UAV altitude 50m, fixed TX 0.1W, CPU NumPy + NVIDIA Warp GPU. Hai cấu hình active là `research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml` và `scenario_v105_ds_rf_stress_1km.yaml`.

## 1. Ưu tiên xử lý — không đánh đồng thiếu bằng chứng với bug đã xác nhận

| Ưu tiên | Mã | Phân loại | Kết quả kiểm toán và cổng cần vượt |
|---|---|---|---|
| **P0 (freeze)** | **C04** | Độ tin cậy DS | Repeated observations không độc lập; DS có thể quá tự tin. Cần mission-level paired CPU/Warp, provenance, false GCS confirmation dưới correlated sensing và kiểm tra hợp lệ thống kê |
| **P0 (freeze)** | **C07** | Sai biệt mô phỏng mạng | DCF/CSMA/CA trên Warp là surrogate; chưa chứng minh tương đương packet events với UavNetSim SimPy upstream. Cần upstream-vs-GPU matched workloads/traces: backoff, ACK, interference, collisions, TTL, retry, queue, routing |
| **P0 (freeze)** | **C18** | Decision gate | Freeze chưa có cùng lúc DS reliability, matched network, nhiều hạt giống, safety/network end-to-end; không tự ghi `SCENARIO_FROZEN` |
| **P0 (giới hạn PASS)** | **C14** | Khả năng chịu tải CUDA | 4096 env ×900 step Base/RF stress PASS với `gentle_patrol`, không chứng minh action bất lợi, comm contention, long-route or worst-case motion. Cần action-policy stress matrix |
| **P1 (training blocker)** | **C19 (MỚI)** | Shape + semantic interface mismatch | Adapter cũ hardcode `RAW_DIM=514`, `DISC_DIM=7` và không nhận active 549/3. Nếu đưa raw549 thẳng vào canonical learner, `obs[9]` bị xem là alive (thực ra là khoảng cách GCS) và 3 phần tử cuối HELLO bị hiểu nhầm là comm mask; cần contract + semantics tests |
| **P1 (training blocker)** | **C11** | Action hybrid | Active action `(vx,vy)` continuous + `compact_3` discrete (idle/GCS/peer); pure continuous MATD3/MASAC không tự xử lý discrete branch. Chốt nonlearned comm policy hoặc explicit hybrid MARL fair comparator trước training |
| P1 | C05 | Ngưỡng DS | `.90` active; `.95/.99` chưa chứng minh tốt hơn theo GCS-level FP/TP/latency; thử paired 3 thresholds, cùng seeds/coverage |
| P1 | C08 | Khả đạt mission | Được phép hình học ≠ reachable under speed/safety, battery, RF delivery; cần offline upper bound route/cell reachability, không đưa oracle vào actor |
| P1 | C09 | PEER/MULE utility | 2×2 ablation cũ 3 seeds×120 steps cho số trung bình giống nhau. Cần workload có P2P packet activity, queue pressure, blackout, custody, >=5 paired seeds/CI |
| P1 | C12 | Năng lượng | Rotary-wing Zeng-inspired steady-level model + own TX airtime, accel term disabled; battery 180kJ; cần conservation và physical/sweep calibration |
| P1 | C13 | RF sensitivity | 12dB loss/building là hệ số scenario, không phải field-measured wall loss; RF sweep against known propagation baselines |
| P1 (đã ghi nhận) | C15 | Định nghĩa observation | `semantic` pooling (max mT/min mF/mean mU) không còn tổng 1; actor 10×10 là *feature summaries*, KHÔNG phải DS BBA; giữ nhãn rõ, validate CPU/Warp parity |
| P1 (trạng thái active) | C10 | TX action | TX cố định 0.1W, không có TX learnable head. Đây là lựa chọn thí nghiệm, không tự coi là defect; cần PDR-energy Pareto nếu muốn đưa power vào action |
| P1 (trạng thái active) | C16 | DS-only | Active GPU từ chối Bayesian; một nhánh Bayesian legacy còn trong source CPU compatibility, không được trình bày là đang chạy active |

## 2. C19 — phát hiện bổ sung có test tái hiện

**Source:** `research/zhang_xue/src/zhang_xue/environment/core.py:16-18`, `environment/observations.py:99-153`, `learners/actor_adapter.py:17-52,112-128`, `research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml`.

- Active `obs_schema_v2=true` => mỗi actor có `14 + 5×10×10 + 35 = 549` float; 6 agents => central state từ flatten local views có `3294` floats. `comm_action_mode=compact_3` => action mask 3 bits/actor.
- Legacy adapter: `RAW_DIM=514`, `DISC_DIM=7`, `ACTOR_POLICY_DIM=522`, và expect `(6,514),(6,7)`; state legacy encoded `6×522 = 3132`, không đồng nhất state active `3294`.
- Thử import module và gọi `encode_actor_input(np.zeros((6,549)), np.ones((6,3)))` cho `ValueError: Expected local 7-radio actor observations (..., 6, 514)`. `encode_actor_input_scalable` cũng `ValueError: Expected local actor observations shape (..., N, 514)`.
- Ngoài lỗi shape: `uav_network/algorithms/baselines/canonical_continuous.py` có helper `event_driven_comm_action_torch(obs)` đọc **3 chiều cuối của tensor** làm comm mask và dùng `obs[..., 9]` như alive mask. Trong active raw observation 549 chiều, **3 chiều cuối là HELLO peer telemetry**, **index 9 là khoảng cách chuẩn hóa đến GCS**; alive/radio bit là index 5. Tương tự canonical MATD3/MASAC update dùng `obs[..., 9]` làm alive. Vì vậy sửa mỗi shape mà không version hóa semantic features sẽ tạo lỗi âm thầm nếu chuyển thẳng raw549 vào canonical learner. Helper zero-oracle đã tồn tại, nhưng wiring với active feature schema chưa được xác nhận.
- **Chưa sửa** vì user chỉ yêu cầu bổ sung xung đột. Điều kiện GREEN: adapter có contract chính thức cho 549/3, giữ 35 HELLO features, đúng alive-mask semantic, tách route mask khỏi HELLO tail, không leak oracle state, no-op TX head absent, `joint state` shape explicit, route-mask fail-closed, tests CPU/Warp, rồi mới sử dụng cho MARL.

## 3. Chi tiết xung đột khoa học/nghiên cứu vẫn OPEN

### C04 — DS evidence phụ thuộc (P0)
- Code: `belief/dempster_shafer.py`; `environment/transition.py`; GPU kernel corresponding.
- V109 synthetic test **50k trials một cell**, threshold .90, 900 negative-world observations: rho=0 FPR ~0.00092, rho=.7 ~0.03616. Không gán rho=.7 là dữ liệu thực, không diễn giải thành mission FPR.
- Nguyên nhân: quy tắc DS lặp p/n counts đánh mạnh vào confidence khi observations tương quan theo thời gian/đường bay.
- Gate: log (UAV, cell, timestamp, bearing/pose), ablate independence vs correlation; report GCS-delivered FPs, TP recall, latency, uncertainty under same flight seeds; model assumptions cited, no hand-tuned discount without evidence.

### C05 — confirmation threshold (P1)
- `.90` được giữ ở V109; `.99` không giải quyết hết correlated false evidence và có thể giảm timely TP.
- Gate: paired threshold `.90/.95/.99`, đủ 900 s, quantify GCS precision, recall, arrival latency, dropped/late packet, seed-paired CI. Không sửa threshold chỉ dựa vào analytical one-cell first passage.

### C07 — UavNetSim parity (P0)
- Project `communication/channel.py`, `communication/scheduler.py`, `communication/lifecycle.py`, `gpu/kernels/mac_v67.py` và `gpu/kernels/phy_v67.py` dựng **mạng packet surrogate** lấy cảm hứng từ UavNetSim. **Không phải upstream UavNetSim được chạy trực tiếp trên GPU**.
- Upstream pinned reference: `https://github.com/Zihao-Felix-Zhou/UavNetSim` commit `c2703f24eb58cdb557a5f67ded5c7b0368a964ed`; SimPy process scheduling / CSMA/CA semantics khác GPU event kernels.
- Gate: exact sample matched topology + packets + seeds/time units; export packet IDs and timing of queue/enqueue/DIFS/backoff/CCA/TX/ACK/retry/drop/route/delivery; compare event sequence or explicit allowed tolerances. Chưa được khẳng định packet-level identical.

### C08 — dynamic feasibility (P1)
- V108 geometric S14 indicates 392 legal/viewable cells nhưng không chứng minh hoàn thành 15-target mission trong time/battery/network.
- Gate: offline reachability solver for obstacles, stop-turn acceleration, >=50m clearance, 180kJ, blackout, delivery; upper bound possible cells and reasons infeasible; oracle used only offline scoring, not actor obs.

### C09 — PEER × MULE ablation chưa phân biệt (P1)
- RF-stress S20 4 arms, 3 seeds×120 steps, GCS TP/cov/reward/AoI averages identical; cannot infer benefit or harm.
- Gate: traffic loads and custody transfers to exercise peer pipeline, `peer_new_cells_rx`, `data_delivered_gcs`, queue occupancy, route failures, PDR, AoI, >=5 paired seeds; confidence intervals and documented achieved treatments.

### C10/C11/C19 — action research decision vs integration bug
- TX=.1W fixed: **deliberate controlled variable**, not learner action; keep until counterfactual RF sweep.
- `compact_3`: action mask idle/GCS/peer exists in environment; decide fixed decentralized communication rule or separate hybrid actor. Do not quietly train 2D continuous network with unmatched random discrete side input.
- Old adapter 514/7 is a **proven shape incompatibility** (C19). Canonical continuous helper for route choice DOES exist, but naïvely fed raw 549 would misread HELLO tail as route mask; canonical alive index 9 mismatches active index 5. Needs explicit semantic adapter tests before baseline.

### C12/C13 — parameter provenance (P1)
- Propulsion uses Zeng-inspired rotary-wing equation and 15W avionics, fixed RF TX PA bookkeeping; zero acceleration-energy flag in active config. These are modeled—not measured—constants.
- RF multi-obstacle `G= (c/(4π f))² / d² * 10^(−kL/10)` with k intersected 3D cylinders and L=12dB/building; no field material calibration. Sweep L, noise, CCA, SNR, power with PDR/energy/delivery metrics.

### C14/C15/C16 — evidence limits / semantic labels
- V109 CUDA Base/RF 4096×900 gentle patrol PASS; not a proof of safety or network correctness under all admissible actions.
- 5×10×10 actor pooled map is **not** itself a normalized DS triple even though underlying 20×20 per-cell `(mT,mF,mU)` is normalized.
- Active DS-only GPU. CPU compatibility branch Bayesian exists but is not the active experimental estimator; don't mislabel paper.

### C18 — freeze checklist (P0)
- **Freeze only after**: C04 mission reliability, C07 upstream network differential, C09 causal PEER/MULE utility, C08 reachability, C14 policy/action stress, C11+C19 interface validation; independently inspect evidence artifacts. Never bypass `v99_freeze_evidence_decision.py` to force PASS.

## 4. Những mục đã FIXED — không coi là OPEN

- **C01:** test V72 stale spawn 65m vs active 120m corrected (no geometry source change).
- **C02:** incompatible V95 `shared_coverage_reward=true` + `first_visit` test corrected; config fail-closed preserved.
- **C03:** PEER OFF / requested MULE ON had false 6× ACK obs scaling; effective MULE now `PEER AND MULE`; RED/GREEN tests and full pytest PASS.
- **C06:** repeated DS ignorance hovering reward replaced by `first_visit` active novelty.
- **C17:** incorrect sanitizer test name corrected in V109 runbook, corrected CUDA memcheck 6 passed / 0 errors.

## 5. Thứ tự thực nghiệm đề nghị (phụ lục, không phải đã chạy)

1. Chốt data contract C11+C19 (549 floats, 3 comm actions, non-oracle critic), RED/GREEN adapter test. Giữ MATD3/MASAC nguyên bản đến khi được phép training.
2. DS mission-level correlated-evidence test C04+C05 và validate sensor origin/time; không sửa threshold/masses dựa trên cảm giác.
3. Chạy upstream UavNetSim vs Warp network event differential C07, gắn nhãn surrogate nếu chưa đạt parity.
4. Chạy workload network stress 2×2 >=5 seeds cho C09; tách packet generated, delivered, E2E ACK, PDR và latency.
5. Safety/energy/reachability offline + policy stress GPU C08/C12/C13/C14.
6. Đánh giá C18 fail-closed sau khi evidence đã có; lúc này mới xem xét `SCENARIO_FROZEN` và canonical learner baselines.

## 6. Bằng chứng trong ZIP giữ nguyên

`outputs/v109_audit/REPORT_VI.md`, `outputs/v109_audit/CONFLICT_MATRIX.csv` (ma trận 18 vấn đề gốc), V108/V109 RED/GREEN, original+patched full pytest, CUDA memcheck, Base/RF full-horizon logs, DS Markov 50k synthetic Monte Carlo. Bổ sung **`outputs/v109_audit/CONFLICT_MATRIX_V109_1.csv`** chứa cả C19. Thông tin source và test chỉ phản ánh những gì đã kiểm tra, không tuyên bố một experiment chưa chạy là PASS.
