# V119 — Kiểm toán và vá KỊCH BẢN DS-only trước khi khóa; CHƯA vào Phase2

## 0. Giới hạn khoa học & nguồn bằng chứng

- **Nguồn mã gốc:** ZIP `UAV_V118_CLEAN_CANONICAL_SCENARIO_AND_NATURAL_AUDIT_20261128.zip` do người dùng cung cấp (SHA256 `dc90bb619e6f37198341b7568aa3e746b1d902631d715cabfd4e7e51c07c1a9a`). **V119** chỉ sửa ABI motor Warp, khóa launcher, bổ sung test/orchestrator và tài liệu; algorithm baseline vẫn nguyên vẹn.
- **Nguồn kết luận V117 action:** V118 `outputs/v118_audit/01_natural_mission_backlog_and_relay_verdict.json`: 3 seed 44–46 ×300 bước, không chèn backlog, RF-only 2,98% UAV-steps backlog >62, base 0%. Có kết quả tổng hợp, **không có raw packet-trace, trajectory và lệnh tái hiện V117 gốc** trong ZIP V118. Không được suy rộng thành action dead trên mọi kịch bản.
- **GPU:** Vast RTX3060 12 GiB; Warp 1.18.0, Torch 2.7.1; script stress reset per episode chứ không giữ 4096 world liên tục 900 bước. Các log `outputs/v119_reaudit/vast_raw/` từ lần chạy remote thật; không sáng tác số.
- **Quyết định:** `SCENARIO_FROZEN = false`: stress pass chứng minh hợp đồng implementation trong phạm vi test, không giải quyết giả định DS độc lập / parity upstream SimPy / ngoại suy reward RF.

## 1. Mô hình hiện hành — cái nào thật, paper nào là nguồn cảm hứng

### 1.1 Bản đồ, điều khiển, camera

Bản đồ 1000×1000m, lưới 20×20 (50m/ô), 6 UAV, 15 mục tiêu cố định, 5 công trình hình trụ, 2 blackout, horizon900 bước và altitude UAV50m. `GCS_POS_M=(50,50,15)` là **ăng-ten tại trạm cố định độ cao 15m**, KHÔNG phải UAV GCS bay. XY policy gồm hai số chuẩn hóa `u_x,u_y∈[-1,1]` (không thêm TX, độ cao hay route action). Máy bay có `vmax=15m/s`, `amax=5m/s²`, dt=1s và shield 66 candidates.

Đường chuẩn lệnh cần nhất quán CPU và Warp:

```math
\bar u = \operatorname{clip}(u,-1,1),\quad
q = \frac{\bar u}{\max(1,\|\bar u\|_2)},\quad
v_{cmd}=15 q,\quad
\Delta v = \operatorname{clip}_{\|\cdot\|_2\le5}\left(v_{cmd}-v_t\right),\quad
v_{nom}=\operatorname{clip}_{\|\cdot\|_2\le15}(v_t+\Delta v).
```

Lý do không dùng `clip(u_x)*15, clip(u_y)*15`: vector (1,1) nằm ngoài đĩa hành động vật lý và làm lệch hướng tăng tốc sau phép giới hạn `amax` nếu `v_t≠0`. **Red/green**: control=(1,1), vprev=(5,0) tạo sai khác 0.5084864826 m/s V118 so với CPU; V119 parity Warp CPU 3.48e-7 m/s và CUDA 3.59e-7 m/s. **Đây là kiểm tra nominal candidate, không phải so toàn bộ safety shield CPU/GPU.**

Hệ công trình (5 hình trụ) và blackout (2 tâm) là tình huống giả định, không mô phỏng bản đồ/LoS LiDAR hay địa hình thực nghiệm. Độ cao cố định loại bỏ bài toán phối hợp độ cao-băng thông và giảm công tính toán; không thể tuyên bố đã benchmark 3D UAV. FoV50m, detection `Pd=.85,Pf=.03` là tham số giả định để tái lập, không phải camera được hiệu chuẩn.

### 1.2 DS ONLY, không Bayesian active

Mỗi origin-cell lưu số lần quan sát dương tính và âm tính `p,n` không âm, cùng timestamp/ACK watermark; code `belief/dempster_shafer.py` dùng canonical masses của positive/negative observation với `mf=me=.8` và lũy thừa `p,n`. Công thức kết hợp từ số đếm độc lập (minh họa):

```math
 A=(1-m_f)^p,\quad B=(1-m_e)^n,\quad Z=(1-A)B+A(1-B)+AB,\\
 m(T)=\frac{(1-A)B}{Z},\quad m(F)=\frac{A(1-B)}{Z},\quad m(\Theta)=\frac{AB}{Z}.
```

Công thức trong source tính bằng `log1p/expm1` để hạn chế triệt tiêu số. V119 **không thay đổi** DS: `belief_inference_mode=dempster_shafer`, `static_evidence_mode=raw_u16_count`, `ds_fusion_mode=sum` qua nguồn gửi, `ds_aoi_eta=0`, confirm `m(T)≥.90` và `m(Θ)≤.2`. Dempster's rule quy ước evidence độc lập là **giả định**, chưa được chứng minh với sensor noise tương quan thời gian. Denœux 2008 đề cập kết hợp nondistinct evidence; **cần thí nghiệm correlated noise** trước khi tuyên bố false positives hợp lý.

DS SUM của V119 **KHÁC** Zhang et al. IEEE T-IV 2024: Zhang dùng nhóm belief DS, không có cơ sở để khẳng định công thức `SUM` theo origin của V119 là nguyên bản Zhang. Cũng không nên ghi V119 là dùng cơ chế discount tuổi DS của Xue vì `ds_aoi_eta=0` (AoI reward và telemetry vẫn có thể active, không đồng nghĩa DS freshness weighting active). Bayesian code legacy tồn tại để tương thích ablation, nhưng **không được gọi bởi YAML active**. Muốn loại tuyệt đối code Bayesian khỏi repository là dự án cleanup tách biệt, không làm âm thầm khi còn các regression dùng nó.

### 1.3 Mạng, RF, AoI và reward

`SevenRadioCsmaCaNetwork`: 6 UAV radio + 1 radio GCS, CSMA/CA queue Q64, TTL10s, MAC ACK/E2E ACK, HELLO mỗi 0.5s và route TTL2s, DATA tối đa62 origin-cell entries (8B/entry), E2E dedup, custom frame `70+8k B`, 2Mbps. `PEER=false` nghĩa **không truyền belief sang peer cho DS map**, KHÔNG có nghĩa luồng DATA đến GCS không thể qua next-hop UAV. `MULE=false` disable forwarding dữ liệu các origin vào peer ledger/relay protocol theo cấu hình tương ứng. TX power 0.1W cố định, không learned action. `a_comm` quyết định bởi `event_driven_comm_action` chứ không learned ở canonical XY-only policy.

V119 là **mô phỏng MAC/PHY lấy cảm hứng UavNetSim-v1** chứ không phải bản Warp SimPy chính thức. UavNetSim-v1 paper mô tả 1024 B payload và noise 4e-11W, trong khi base V119 đặt noise ~4.389577e-13W và compact DATA frame nhỏ hơn, do đó capacity/retry/delivery có thể lệch lớn (~19.6dB noise sensitivity). RF-only noise 4e-11W được dùng làm ablation độc lập, không được tráo đổi thành canonical nguồn.

Reward phải đọc từ `environment/configuration.py`, `environment/rewards.py` và `gpu/...` tại runtime; có r_tp, r_fp, w_loc_team, w_aoi, coverage first_visit... Không viết `w_p2p,lambda_b,ds_aoi_eta,continuous_aoi_penalty_w,enable_accel_energy` vào công thức **active** vì các knob đó đều bằng 0 hoặc false. Tuy nhiên thành phần AoI/first-confirm thưởng đang active; không đánh đồng `ds_aoi_eta=0` với “không còn AoI trong toàn nhiệm vụ”. Reward GCS TP/FP có ground-truth để chấm điểm, cần kiểm chứng true target không rò vào local observation/action selection.

## 2. Quyết định giữ/sửa/bỏ, không thêm action thiếu utility

| Phần | Quyết định | Chứng cứ/hạn chế |
|---|---|---|
| DS-only raw counts, threshold .90 | **GIỮ tạm** | Stable numerics; correlated FP và mass calibration chưa giải quyết, KHÔNG phải freeze đầy đủ |
| XY normalized action2 + feasible shield | **SỬA C38** | Parity red/green motor nominal; full CPU–GPU world parity tiếp tục mở |
| `run_v115_canonical_ds_baseline.py` | **SỬA C42/C48/C51** | Cờ `--allow-experimental-physics` bắt buộc, SHA256 2 YAML đã pin; không lặng lẽ train RF/PEER/Bayes |
| Pure-GCS, TX0.1W | **GIỮ** | Đúng scope trajectory-only canonical continuous; routing DATA vẫn có thể multi-hop |
| V117 `a_evidence,a_relay` | **KHÔNG ĐƯA VÀO canonical** | Natural3seed×300step cho utility 0 hoặc âm, hybrid action cần baseline khác; không đủ chứng minh vô nghĩa mọi thế giới |
| Reward/DS AoI discount theo tuổi | **KHÔNG BẬT TUỲ TIỆN** | `ds_aoi_eta=0` active; phải có thí nghiệm, không dùng AoI-priority nhân tạo |
| UavNetSim packet timing & realistic PHY | **CHƯA DÁN NHÃN REPRODUCTION** | Chưa đối chiếu event-level/packet-level upstream; RF calibration synthetic |
| MATD3/MASAC files | **GIỮ NGUYÊN** | Không thêm hybrid action; sau khi scenario freeze mới đánh giá code canonical và S25 replay |

## 3. Gate và nghiệm thu: không được cộng bừa PASS

GATE A: SHA256 manifests, source policy, thuật toán file identity. GATE B: CPU unit + FULL pytest GPU. GATE C: motor action parity (V118 red, V119 green nominal). GATE D: 4 × 4096 env ×900 step trên CUDA, 4 trường hợp (base random; RF random; RF boundary; RF churn), **mỗi trường hợp có `--output` và JSON status riêng**. GATE E: số liệu mission tự nhiên nhiều seeds trong 900 bước, delivery/coverage/FP/AUC/mean return trên PHY baseline và RF stress. GATE F: SimPy event-level parity có cùng traffic/seed/PHY/phản xạ; vấn đề packet length & noise phải tách biệt. GATE G: sensor correlation và DS mass calibration, map/battery reachability. GATE H: baseline interface dim & no hidden oracle reward leaks.

**Chỉ khi A–H chứng minh được scope công bố mới được thay `SCENARIO_FROZEN=false`.** Nếu muốn công bố mô hình chỉ là `UavNetSim-inspired surrogate`, có thể khóa *engineering scope* mà không giả vờ đã pass SimPy parity, nhưng phải đổi tên claim đúng mức.

### Lệnh chạy tái hiện (Vast GPU, từ gốc repo V119)

```bash
PYTHONPATH=research/zhang_xue/src:. python scripts/maintenance/verify_source_manifests.py --check
PYTHONPATH=research/zhang_xue/src:. python -m pytest -q tests/ research/zhang_xue/tests/
PYTHONPATH=research/zhang_xue/src:. python scripts/gpu/v119_four_way_cuda_gate.py --device cuda:0 --envs 4096 --steps 900 --seed 44 --output-dir outputs/v119_gpu_four_way
# Hiện vẫn CHƯA nên train. Chỉ khi các gate khoa học đã khóa mới mở Phase2:
PYTHONPATH=research/zhang_xue/src:. python scripts/gpu/run_v115_canonical_ds_baseline.py --algorithm matd3 --allow-experimental-physics --envs 64 --steps 300 --output outputs/phase2_matd3.json
PYTHONPATH=research/zhang_xue/src:. python scripts/gpu/run_v115_canonical_ds_baseline.py --algorithm masac --allow-experimental-physics --envs 64 --steps 300 --output outputs/phase2_masac.json
```

**GHI CHÚ:** Bản thân `--allow-experimental-physics` chỉ là xác nhận đọc V83 scope, không khẳng định PHY thực tế, không tự động bật training trong lần kiểm toán này. `--dry-run` cho wrapper cũng yêu cầu acknowledgement để xác minh sự đồng ý tường minh. `v119_four_way_cuda_gate.py` có cờ `--dry-run` xem 4 lệnh không launch GPU.

## 4. Toàn bộ vấn đề xung đột

Xem `outputs/v119_reaudit/06_FULL_CONFLICTS_C01_C72_WITH_CURRENT_STATUS.csv` và `.json` (72 bản ghi). C01–C67 được giữ **không xóa lịch sử**; V119 bổ sung C68–C72. C56–C64 là **vấn đề của nhánh experimental V117 đã RETIRED khỏi canonical**, không nên báo chúng đều FIXED. C38/C42/C48/C51 đã vá có bằng chứng test; C04/C07/C08/C13/C18/C23/C25/C27/C28/C33/C35/C39/C65/C66/C71/C72 tiếp tục mở theo từng mức. Full source+logs/bảng nguồn paper có trong archive; **không Git push**.


## 5. PHỤ LỤC V119 — Paired mission tự nhiên CUDA, 5 master seeds × 4 worlds × 900 bước

Lần chạy thực tế trên Vast RTX 3060 và **kernel V119 sau vá C38**, dùng `scripts/gpu/v112_paired_scenario_gates.py`, với seeds `[44,45,46,47,48]`, `4 worlds/seed`, horizon `900` (`first-terminal` cho paired case). Có **6 kịch bản, mỗi kịch bản 20 worlds cùng master seeds**, thay đổi theo từng yếu tố; policy hành trình là **non-oracle heuristic cố định**, KHÔNG phải actor được MATD3/MASAC huấn luyện. Raw **44,902 B** `outputs/v119_reaudit/vast_raw/14_v119_natural_5seeds_6cases_900.json`, SHA256 `07b35d1dac574bc6e3d084cacb9a26f0662904bf884c81aecbf916380c739197`; raw log cùng tên `.log` sha256 `756138a5af5477bd9e0cf6f7136cf3ed83de46647a5039c4ab4e2593a5a435a7`. File summary.csv chỉ được tính bằng cách parse raw JSON này, KHÔNG dựng từ bảng thủ công.

| Case | GCS delivery % (mean) | Coverage % | FP-ever / world | Full 15/15 worlds |
|---|---:|---:|---:|---:|
| base (noise repo; pure-GCS) | 98.000 | 93.648 | 0.30 | 15 |
| rf_only (noise 4e-11 W; pure-GCS) | 56.667 | 65.561 | 0.00 | 0 |
| rf_peer (RF stress, PEER ON) | 24.000 | 66.173 | 0.00 | 0 |
| rf_peer_mule (RF stress, PEER/MULE ON) | 24.667 | 65.969 | 0.00 | 0 |
| base_peer (base RF, PEER ON) | 96.333 | 95.982 | 0.25 | 12 |
| base_threshold099 (base RF, DS .99) | 95.333 | 95.230 | 0.00 | 9 |

Các cặp có paired **cluster-bootstrap theo 5 master seeds** (khoảng 95% mô tả, chưa đủ bảo đảm ngoại suy):

- **Noise repo→RF stress:** −41,333 điểm % delivery, CI [−47,667; −35,000]. Không được tráo RF-stress thành canonical vì thông số nhiễu khác UavNetSim repo (mà trùng paper Table) và làm performance giảm mạnh.
- **PEER OFF→ON trong RF-stress:** −32,667 điểm %, CI [−40,333; −26,333]. Với heuristic và traffic cụ thể này, chia sẻ evidence làm tắc MAC; đây KHÔNG là định lý PEER luôn có hại.
- **Thêm MULE khi RF-stress có PEER:** +0,667 điểm %, CI [−5,000; +6,000], *không chứng minh cải thiện*.
- **PEER OFF→ON trong RF base:** −1,667 điểm %, CI [−3,667; +0,333], *không chứng minh khác biệt có lợi*.
- **DS confirm .90→.99 ở base:** −2,667 điểm %, CI [−4,000; −1,000], FP-ever 0,30→0 trong 20 worlds; chỉ là trade-off trong fixture, **không có chứng cứ ngưỡng .99 đã được hiệu chuẩn**.

Tất cả 6 case: `safety_failed_worlds=0`, `overflow_worlds=0`, nhưng đây **không loại trừ lỗi cơ chế DS, reward leakage, tương quan sensor**, hay sai khác packet-level với SimPy upstream. Bốn stress CUDA dùng reset mode, không được nhầm với paired first-terminal evaluation. Bảng raw machine-checkable ở `25_V119_PAIRED_NATURAL_900_SUMMARY.csv` và `26_V119_PAIRED_DELIVERY_DELTAS_SEED_BOOTSTRAP_CI.csv`.

### Gate E cập nhật

**Đã chạy:** 6 scenario × 20 first-terminal worlds ×900 steps dưới heuristic, đo delivery/coverage/FP/return; paired CI có n=5 cụm seed. **Chưa thể đóng hoàn toàn:** chưa có AUC/uncertainty cho mọi loại policy, chưa có environment-state CPU↔GPU full parity, chưa đo utility với actors trained, chưa hiệu chuẩn DS correlated evidence, chưa SimPy event-level comparison. Vì thế `SCENARIO_FROZEN=false` và Phase2 vẫn ở trạng thái HOLD.

