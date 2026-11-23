# M2 / Correlated DS / Cooperative SPRT / UavNetSim — Audit & Native implementation

**Ngày:** 2026-11-21. **Người dùng đã chốt:** DS 1B; candidate 2B; SPRT 3C, alpha=beta=0.05; sensor tương quan *cả theo thời gian và giữa UAV*; cooperative pooling không bắt buộc hai UAV; event-ID fusion 4C-1; 5D reward; observation-only Critic; Q4B correction/retraction; M2 multi-layer map; GCS chỉ phát cho UAV trong tầm, các UAV tự học relay. Không có tự động hạ 50m.

## A. Những xung đột xác nhận được khi audit

1. Legacy `gpu/bayesian.py` dùng tọa độ target thật để sinh sensor (hợp lệ) **và** dùng `has`/target IDs để reset false confirmation, tạo `s.visible` cho target thật và bỏ qua false reports (không hợp lệ với 2B+SPRT). **M2 Native xử lý:** protocol không nhận `target_id`; sử dụng cell, event, decision; simulator truth chỉ được cấp cho sensor/score.
2. Legacy `communication/gpu/reports.py`, `streaming.py`, `mac.py` khóa report storage theo true target ID, chưa có evidence/correction/map sync. **M2 Native xử lý:** message-envelope riêng qua pinned real UavNetSim, per-message IDs, packet size, bounded queue, duplicate suppression. **GPU chưa xử lý.**
3. Legacy generation không bảo đảm một target/cell. **M2 Native xử lý:** reject/resample scene trùng cell; explicit scene bị từ chối. Legacy checkpoints giữ nguyên.
4. DS Dempster, SPRT chuẩn giả định nguồn có thông tin độc lập có điều kiện. **M2 Native xử lý:** cả sensor simulator lẫn likelihood decoder có một biến bias tiềm ẩn chung cho mỗi cell (gây tương quan giữa UAV và theo thời gian). Receiver replay các events theo `(tick, source, event_id)` và tính lại posterior bias dưới hai giả thuyết.
5. Backend truyền một chiều UAV->GCS khi chỉ có true target report. **M2 Native xử lý:** GCS gửi đúng `M2Envelope` qua UavNetSim tới một UAV trong tầm; các UAV chuyển tiếp tiếp bằng destination action. Không có broadcast lý tưởng.
6. `MATD3/MASAC` đòi `observation[..., 9]` là `active` và `[-(n+1):]` là destination-mask. **M2 Native xử lý:** giữ đúng prefix/index này; đã smoke-test cả hai learner hành động, replay và cập nhật gradient thực.
7. Buffer evidence truyền liên tục sẽ đầy, reward delivery có thể thưởng lặp khi hai UAV xác minh cùng cell. **M2 Native xử lý:** FIFO có ưu tiên correction, evict bounded buffer; positive delivery thưởng tối đa một lần trên mỗi target cell.
8. Obstacle map không thể được suy ra từ binary target sensor. **M2 Native xử lý:** mô hình sensor phân loại footprint obstacle *riêng*, với Pd/Pf và nuisance riêng; đây mới là layer occupancy 2D, **chưa phải full 3D obstacle reconstruction**.

## B. Phương pháp toán học thực sự được cài đặt

### B1. Observation của mỗi cell

Với camera tại h, footprint bán kính `r=h*tan(FOV/2)`, `q` là tỉ lệ diện tích cell trong FoV; LOS tới cell center của camera target được kiểm tra bằng geometry.

- `p0 = 1-(1-Pf(h))**q`
- `p1 = q*Pd(h) + (1-q)*p0`
- q=0 hoặc không có LOS hợp lệ -> **không quan sát**, không báo empty.

Hai phương trình hiệu dụng này là giả định mô hình hóa occupancy-level cho các cell partial FoV, không lấy nguyên xi từ Zhang 2024. Cần kiểm tra với occlusion thực tế (center-ray approximation).

### B2. Correlated sensor giữa UAV và thời gian

Một bias sign `B_c in {-1,+1}` được sample đúng một lần mỗi cell cho toàn bộ mission, độc lập với target occupancy. Khi sensor nhìn cell:

`P(z=1 | H_j, B_c=b, h, q) = p_j + b*rho*min(p_j,1-p_j)`, với `j∈{0,1}`.

Hai sign equiprobable; trung bình theo sign cho đúng `p_j` (giữ calibration marginal Pd/Pf). Sensor sample Bernoulli conditional. Vì nhiều UAV chia sẻ cùng bias, phép đo có tương quan dương cả giữa các UAV và theo thời gian; không phải CNN/image detector.

### B3. Conditional likelihood cho DS + SPRT

Receiver lưu hai posterior trên bias sign: `w_b^(H1), w_b^(H0)`. Với sự kiện mới k:

`P_jk = sum_b w_b^(Hj) * P(z_k=1 | Hj, b, h_k, q_k)`;

`ell_k = z_k*log(P_1k/P_0k) + (1-z_k)*log((1-P_1k)/(1-P_0k))`;

`w_b^(Hj) <- w_b^(Hj)*P(z_k | Hj,b) / sum_{b'} w_b'^(Hj)*P(z_k | Hj,b')`.

Evidence simple-support từ likelihood có điều kiện: nếu z=1, `mf=1-P0k/P1k`; nếu z=0, `me=1-(1-P1k)/(1-P0k)`. Đây là lựa chọn thiết kế likelihood->DS, **không phải phương trình được Zhang 2024 chứng minh và không bảo đảm DS mass là posterior đã calibrated**.

DS cập nhật bằng Dempster Eq.(6)-(10) Zhang 2024, có `m(F)+m(E)+m(U)=1`. Candidate khi m(F)>=0.95. SPRT **khởi từ S=0 sau event tạo candidate**; lịch sử trước candidate chỉ dùng trong conditional likelihood; không cộng lại vào SPRT. Khi S>=log(19) verified, S<=-log(19) rejected, còn lại pending. Alpha=beta=.05 là ngân sách nominal per candidate; conditional/adaptive policy cần đánh giá thực nghiệm mới biết false-confirmation thật.

Để thực hiện Q4B, mỗi node replay lại sự kiện theo observed time nếu nhận out-of-order, và phát `DecisionNotice` mới với revision tăng khi trạng thái thay đổi. Đây là **revision-enabled cumulative LLR** chứ không phải Wald SPRT cổ điển dừng vĩnh viễn ở lần quyết định đầu tiên. Không được trích dẫn error bound của Wald như một chứng minh cho cơ chế retraction này.

### B4. Communication protocol và bản đồ GCS

Mỗi nguồn phát `Event(event_id,origin,tick,cell,layer,positive,p1,p0,rho)` hoặc `DecisionNotice(message_id,source,cell,revision,status,tick)`. Packet chứa một hoặc nhiều sự kiện (ước tính kích thước JSON + header <= 1024 byte); UavNetSim xử lý CSMA/CA, PHY, queue, ACK, retry, TTL. API Native sử dụng message-object metadata trong quá trình reassembly như cơ chế report legacy; **không phải mô phỏng byte-level JSON corruption**.

- GCS ban đầu target/obstacle map đều `(0,0,1)`, coverage=0.
- Map mỗi node chỉ đổi sau `node.ingest` từ sensor tại node hoặc radio-packet đã hoàn tất tới node đó.
- Events immutable + dedup theo event_id; history replay theo observed time; delayed messages không biến thành independent observations.
- GCS downlink chỉ dùng radio link khả dụng; relay ngoài vùng bắt buộc policy UAV chọn destination; không auto-sync từ memory.
- Receiver lưu link-contact timestamp sau khi **thực sự nhận** packet; báo cáo có version; packet cũ không được ghi đè correction mới.
- Buffer per node bounded; các packet evidence cũ có thể bị loại và không bao giờ tới GCS. GCS map vẫn lưu các evidence đã nhận trước đó.

### B5. Multi-layer World Map

- target DS `(F,E,U)`;
- obstacle footprint DS `(F,E,U)`;
- coverage (`observed event count`);
- freshness (`last observed`/`last arrived`);
- verification/candidate/claim history;
- network contacts và packet buffers.

Lớp obstacle hiện mô tả **occupancy footprint trên bản đồ ground 2D**, chưa có geometry mesh/height reconstruction. Ground-truth target/obstacle bitmaps chỉ nằm trong simulator sensor và evaluator; `state()` của centralized Critic nối các local observations, không có true target/obstacle ID/XY.

### B6. Reward

- candidate .15/lần xuất hiện đầu tiên của một cell;
- verification +1 / -1 theo đúng-sai (oracle chỉ trong reward);
- GCS nhận đúng report +50 tối đa một lần mỗi true-target cell;
- mỗi false verified report -10;
- tiến độ Brier target+obstacle tại GCS (chỉ khi radio mang event tới GCS);
- step/propulsion/network penalties theo config.

Hệ số trên là **prototype sanity defaults, không phải calibrated final 5D reward**. Không giữ bonus descent, không ép hạ xuống 50 m. Actor vẫn có 4 continuous dimensions XYZ+TX và 1 discrete destination.

## C. Những gì đã test và chạy thật

- Tests new: sensor marginals + cross-agent correlation; conditional LLR; DS/CoopSPRT; out-of-order replay, source dedup; target & obstacle maps; versioned correction; bounded queue; real UavNetSim UAV->UAV->GCS and GCS->UAV; no direct GCS map copy; Actor action-mask ABI; MATD3/MASAC replay + 2 gradient updates; no forced descent.
- Existing DS Warp unit/parity and Native network telemetry/GPU packet regression: `88 passed, 20 subtests passed` (ở một đợt trước sửa policy ABI; cần chạy lại cuối cùng).
- Real 35-target/25-obstacle/6-UAV Native 3D smoke 6 steps: source scene no duplicate cells, no crash, GCS nhận evidence.
- **60-step diagnostic** 6 UAV / 35 target / 25 obstacle / 1000x1000m / grid 25m / seed 44, *scripted route/frontier heuristic*, không phải learned Actor. `GCS received events 1024`, `received radio packets 373`, `PDR 90.10%`, `GCS coverage 11.625%`, `target Brier ~0.23685`, `obstacle Brier ~0.22451`. **No true target verified/delivered in this short diagnostic; do not claim search success.**
- Figures and evaluator NPZ at `~/Downloads/UAV_M2_CORRELATED_COOP_20261121/`.

## D. Các phần CHƯA xong — không được ghi là production-ready

1. **Warp E4096-U8-B8192 GPU protocol integration**: Stage1 Warp DS kernels có kiểm thử riêng nhưng M2 correlated sensor, distributed event buffer, packet decoder, GCS downlink, report statuses **chưa được nối** vào `environment/gpu/vector.py`, `communication/gpu/mac.py/reports.py`, hay checkpoint/Eval/W&B. Native UavNetSim single live instance is NOT a substitute for 4096 GPU envs.
2. **Convergence / learned routing / learned altitude**: Có real Actor/Critic CPU update smoke; **chưa train MATD3/MASAC tới hội tụ**; diagnostic policy dùng heuristic, không dùng checkpoint đã học trên nhiệm vụ M2.
3. **Statistical calibration**: cần nhiều seeds và kiểm định alpha/beta per-candidate dưới shared nuisance, selection conditioning, delayed messages và Q4B retractions. Không được khẳng định error guarantee 5%.
4. **Sensor map fidelity**: obstacle model là 2D noisy footprint, chưa reconstruct heights/3D meshes. Occlusion model chỉ xấp xỉ line to cell center. Real correlated optical/obstacle detector statistics chưa calibrated với dữ liệu thực.
5. **Throughput**: Python per-cell event logs và full-history replay theo nhận packet là O(history), chưa thích hợp GPU 4096. Cần sparse bounded provenance, checkpointed sufficient-statistics theo latent shared-bias và Warp kernels, đo VRAM/throughput trên T4x2 trước full training.
6. **Inter-node contradictory claims**: GCS theo dõi version từng `(source,cell)`, nhưng chưa có chính sách quyết định một kết luận authoritative giữa các UAV mâu thuẫn. Cần chuẩn hóa policy/report SLA trước paper result.
7. **Training scripts**: `experiments/training.py` hiện tạo legacy `make_vector` và `warp_packet` không tạo CooperativeM2Env; muốn 20k episodes phải tạo riêng M2 backend và config/runner mới. Không trỏ notebook cũ sang source mới và tuyên bố tương đương.

## E. Chạy lại

```bash
cd /home/aduy/Documents/NCKH/uav_network_coopmap_20261121
python -m pytest -q tests/test_correlated_m2_protocol.py tests/test_m2_uavnetsim_integration.py tests/test_m2_mission_native.py tests/test_ds_evidence_map.py tests/test_ds_evidence_warp.py tests/test_network_telemetry.py tests/test_gpu_packets.py
python scripts/research/run_cooperative_m2_native_demo_20261121.py --steps 60 --seed 44
```

Nhánh cô lập từ commit `9c7b081`, đã copy snapshot source đang sửa từ `/home/aduy/Documents/NCKH/uav_network_study_1200` để smoke test đúng phiên bản hiện tại; original dirty working tree và mọi notebook/W&B/checkpoint vẫn không bị sửa. Không khẳng định source commit sạch đầy đủ tương đương worktree đang có: các thay đổi chưa commit từ source phải được bao gói riêng khi phát hành.

## F. Kết quả kiểm thử hoàn tất và minh bạch phạm vi

- Sau điều chỉnh action-mask ABI và reward dedup: **93 passed + 20 subtests** trong targeted regression `test_correlated_m2_protocol`, `test_m2_uavnetsim_integration`, `test_m2_mission_native`, `test_ds_evidence_map`, `test_ds_evidence_warp`, `test_network_telemetry`, `test_gpu_packets`; bổ sung kiểm tra learned-action/replay và test correction ngoài bộ trước. Nên chạy lại để lấy số cuối cùng.
- **Toàn bộ tests/** (khi đã mang `scripts/` snapshot hiện tại vào worktree): **1793 passed, 105 skipped, 27 failed, 1982 subtests passed** trong 370.92 s. Lịch sử `docs/superpowers/CURRENT_TASK.md` của worktree gốc cũng đã ghi nhận 27 failure còn tồn tại trước đó (trong nhóm checkpoint/recovery, W&B scripts, structural source). Không kết luận mọi test failure hiện tại giống byte-for-byte với baseline khi chưa đối chiếu từng case. Không tuyên bố full repo green.
- Bản đồ GCS Native demo chỉ phủ 11.625% sau 60 bước với policy diagnostic, **không chứng minh learned routing hoặc improved target search**. Brier target=0.23685095, obstacle=0.22451284; GCS verified reports=0. Đây là kết quả trung thực của demo, không phải benchmark hoặc kết quả Q1.
- `M2Radio.step` hiện có GCS downlink chọn peer theo chính sách luân phiên trong số node khả dụng (phần node GCS chưa có learned controller). Tất cả quyết định relay của **UAV** vẫn lấy từ action; diagnostic script sử dụng heuristic, chưa phải Actor đã train.
- Prototype `CooperativeM2Env` có limited `max_scan_cells` và bounded buffering, phù hợp kiểm chứng Native. Không được thay nó vào training E4096 và mong cùng throughput.

### Đối chiếu 27 lỗi lịch sử

Đã đối chiếu `tests/` full-suite failure names với `outputs/baseline_rootcause_20261120/test_status.json` của nguồn gốc. **Cả 27/27 tên case hiện tại trùng đúng 27/27 case lịch sử** (không có tên failure mới). Điều này không chứng minh mọi thay đổi đều chính xác, nhưng xác nhận không xuất hiện tên case regression mới trong full suite hiện có. Khi chạy `pytest tests/` thời điểm 2026-11-21, summary là `1793 passed, 27 failed, 105 skipped, 1982 subtests passed`. File log tạm: `/tmp/m2_full_test_suite_with_scripts_20261121.log`.
