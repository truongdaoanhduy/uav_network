# V121 — Kiểm toán kịch bản UAV DS-only, sửa lỗi và điều kiện đóng (28/11/2026)

**Quyết định:** `SCENARIO_FROZEN=false`. **PHASE 1** đang kiểm toán kịch bản. **PHASE 2** MATD3/MASAC chuẩn chưa được phép chỉnh sửa hoặc huấn luyện. **Không push GitHub.**

## 1. Phạm vi phiên bản và nguồn bằng chứng

- Input gốc: `UAV_V120_CONFLICT_SOLVER_AND_SCENARIO_AUDIT_20261128.zip`; SHA256 `faa2fa23c12ad10976759dbf374dcd194f4077f59df1f0682226c3022595ef00`. Gói gốc được giữ ở `input/V120_ORIGINAL_VERIFIED.zip` (để ngoài cây source phát hành).
- Mã nguồn đầu vào: `uav_network_V120/`. Mọi sửa đổi V121 so với V120 ở `outputs/v121_scenario_audit/V120_to_V121_source.diff`; hash danh sách trong `V121_CHANGED_FILES_AND_HASHES.json`.
- Canonical YAML: `research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml`, SHA256 `3fcfe434b35c29310835c2aa99f982a96c4af58375dc5e916e1e4d017a832fc5`. RF YAML `scenario_v111_ds_rf_stress_no_peer_1km.yaml` dùng cho witness. **YAML canonical không đổi**.
- Toàn bộ 30 file `uav_network/algorithms/**/*.py` giữ nguyên byte so với V120; fingerprint theo tên và SHA256: `bed0dd34d61a814eb0594f244f8dbab6a1dbfd449289ebae41cb4aa88cccef6d`.
- Registry: 72 mục C01–C72 lịch sử được giữ nguyên từng trường và bổ sung trường đánh giá V121; 10 mục mới C73–C82 trong `outputs/v121_scenario_audit/V121_CONFLICT_REGISTRY_C01_C82.{json,csv}`. Không hiểu 82 mục là 82 lỗi còn mở.

## 2. Kịch bản đang chạy

| Mục | Trạng thái canonical |
|---|---|
| Bản đồ và thời gian | 1.000 × 1.000 m, 20 × 20 belief cells (50 m mỗi ô), dt=1 s, horizon=900 |
| UAV / target | 6 UAV, 15 mục tiêu tĩnh; cao độ UAV 50 m; GCS cố định (50,50,15) m, z là tham chiếu anten, **GCS không bay** |
| Điều khiển | Action học **chỉ** `(u_x,u_y)` liên tục 2 chiều; vận tốc max15 m/s, gia tốc max5 m/s²; shield 66 ứng viên + kiểm tra fail-closed; không CBF |
| Vật cản/blackout | 5 công trình, 2 blackout; NFZ cản bay, blackout cho UAV bay nhưng cắt đường radio theo model; cản sensing 3D được bật |
| Sensor | `Pd=0.85, Pf=0.03`, FoV 50 m; nhiễu giả dương và âm tính |
| Evidence | Dempster–Shafer **only**, initial (mT,mF,mTheta)=(0,0,1), `mf=me=0.8`, `SUM` source counts, threshold=.90, uncertainty ceiling=.20; `raw_u16_count`, eta=0 |
| Comms | Pure-GCS **application evidence**: `PEER=false, MULE=false`; **multi-hop routing đến GCS vẫn hoạt động**; HELLO/ACK vẫn có airtime; TX power cố định .1 W |
| Network | NVIDIA Warp custom UavNetSim-**inspired** DCF surrogate (CPU simulation riêng); không chạy upstream SimPy đồng nhất sự kiện |
| Energy | 180 kJ/UAV, mô hình propulsion lấy cảm hứng từ Zeng 2019; tính thành phần acceleration tùy chọn **đang tắt** |
| Reward | First-visit cell + GCS-confirmed TP + GCS FP penalty + FP hold + AoI bonus khi xác nhận + shield intervention + fail-closed; `w_p2p=0`, `lambda_b=0`, `continuous_aoi_penalty_w=0` nghĩa là các phần thưởng/phạt tương ứng không góp active |
| End | Khi đủ mọi GCS-confirmed TP hoặc all depleted hoặc safety failure; truncate khi horizon |

### Toán DS đang triển khai
Theo `research/zhang_xue/src/zhang_xue/belief/dempster_shafer.py`, mỗi ô có bằng chứng dương `p`, âm `n`, độ tin cậy `m_f,m_e`. Đặt
`A=(1-mf)^p, B=(1-me)^n, Z=A+B-A*B`.
Khi `Z>0`: `m(T)=(1-A)B/Z`, `m(F)=A(1-B)/Z`, `m(Θ)=AB/Z`. Đủ `m(T)>=0.90` và `m(Θ)<=0.20` tại **GCS** mới tính xác nhận, không yêu cầu từng UAV xác nhận trước khi phát. Dùng `SUM` ở đây là **adaptation** (khác xử lý MAX của Zhang), không gọi là công thức y nguyên paper.

### Flow thực thi CPU
`environment/transition.py`: validate action / tx power (không chấp nhận TX action khi power control=false) → enqueue dirty evidence từ bước trước → mô phỏng radio CSMA/CA, HELLO, routing, ACK → GCS update evidence and mass → safety shield và cập nhật chuyển động/energy → sensing tại vị trí mới, tích lũy local evidence → reward, GCS metrics, termination và observation. Do truyền trước khi sensing, một evidence mới **thường có thể truyền từ bước kế tiếp**, không được nhận giả tức thời. GPU nằm ở `gpu/mission_v80.py` và kernels, kiểm parity từng phạm vi.

## 3. Lỗi và sửa RED → GREEN / bằng chứng

1. **C73 validation reset semantics:** lệnh V120 dùng `--worlds` sai cho script cũ và script cũ auto-reset. Đã tạo `scripts/gpu/v121_uninterrupted_cuda_gate.py`, monitor trên GPU, reject những trường hợp thiếu 900-step/no-reset, log từng chặng. Chạy lại sau bản vá shield: **Base PASS**, **RF noise PASS**, mỗi case **4096 world×900 step, 0 auto-reset, 4096 first terminal đúng 900**; 0 physical violation/overflow/nonfinite/energy-increase/safety failure; peak VRAM 2729/2743 MiB. Đây là finite action workload, **không** là proof cho mọi policy.
2. **C74 calibration sai đơn vị:** hai file `scripts/maintenance/v120_*.py` từng gửi command được normalize≈1 vào CPU API đo m/s. Unit-test phát hiện, đã sửa scale. Hai thí nghiệm **45+45=90 run CPU**, 3 seed×300, toàn bộ raw JSON và stdout/stderr ở `calibrations/`.
3. **C75 CPU MAC race:** radio cùng chạm backoff/DIFS deadline được commit tuần tự theo index, gây bất đối xứng. Dùng ready-start snapshot theo microsecond để mọi node cùng phát; test expected collision ở receiver, frozen backoff và restart DIFS trong `tests/test_v121_network_parity.py`.
4. **C76 HELLO airtime:** CPU chưa charge bytes peer telemetry cho HELLO schema v2; sửa frame duration và test 7 node, 360us legacy, DATA 70+8k và ACK giữ contract; **không** tuyên bố parity upstream SimPy.
5. **C77 CPU DS-only:** direct config / CLI bypass vào Bayesian belief trái canonical. Lock `belief_inference_mode=dempster_shafer`, fail-closed mode khác; test `tests/test_v121_ds_only_runtime.py`. Những hàm toán Bayes legacy ngoài canonical chưa bị xóa bừa.
6. **C78 shield candidate tie:** 16/900 mẫu chung prestate sai lệch đến **2.8701257 m/s** vì FP32 CUDA chọn khác FP64 CPU ở near-equal costs. Dùng cùng tolerance số học `64×eps(float32)` cho **chi phí lựa chọn** (không nới an toàn vật lý), so sánh lại 5 worlds×180 mixed actions: **0 divergent**, max velocity **3.85813e-6 m/s**, max position **4.123e-5 m**. Không chứng minh CPU/GPU full-state parity.
7. **C79 regression fixture sai scope:** test gọi `_project_uav_v80(check_peers=0)` geometry-only nhưng lại expect đầu ra full peer-aware 3-pass. 6 subcase ban đầu FAIL; cập nhật fixture đúng API, focused **4 PASS + 9 subtest PASS**; lưu cả log thất bại.
8. **C80 metadata cũ Bayes:** sửa `docs/superpowers/CURRENT_TASK.md` và test `test_v121_active_metadata.py`.
9. **C81 khoa học:** V120 ngầm coi 2 lệnh GPU đủ `SCENARIO_FROZEN=true`; V121 **không** áp dụng suy luận đó do DS temporal dependence và nguồn RF/building không được hiệu chuẩn, không SimPy event parity, không full world-state parity.
10. **C82 feasibility:** bound thẳng nhìn trước mục tiêu là upper bound lý tưởng, **không** phải chứng minh nhiệm vụ khả thi trong mạng thật. Bổ sung `v121_constructive_mission_witness.py`: một đối chứng constructive **biết mục tiêu** điều hướng, kiểm obstacle, sensor và GCS actual delivery; 5 seed×2 profile được chạy với full per-step trace. Nếu PASS, chỉ kết luận finite witness, không đại diện learned/non-oracle policy.

## 4. Đánh giá khoa học sau khi sửa đơn vị — không được lấy V120 cũ làm kết luận mới

V121 45 DS synthetic natural missions, 3 seed ×300 mỗi arm, không reset:
| Thí nghiệm | Kết quả corrected |
|---|---|
| DS SUM FP-ever/world theo Markov rho | 0→**0.333**; 0.5→**8.333**; 0.7→**13.0** |
| SUM vs MAX tại rho=.7 | **13.0 FP-ever/world cả hai**; MAX không khắc phục tự lặp bằng chứng từng UAV |
| Threshold .90 vs .99, rho=.7 | TP-ever **11→10**; FP-ever **13→8.333** |
| `mf/me` capped 0.80 vs noncanonical 0.99, rho=0 | FP-ever **0.333→8.333** |
| 45 RF-energy runs (base,300 steps) | `enable_accel_energy=False`: battery min mean **74.12%**; True: **70.50%** (không thể còn khẳng định tác động nhỏ hơn 0.01%) |
| RF loss 0 / 12 / 20 dB, base | mean GCS PDR **.9613 / .9409 / .9240**; building loss 12dB là heuristic |
| Network substeps 1 vs2 | base PDR **.9409 vs .9403**; RF PDR **.2387 vs .2389**; không thể nói CPU MAC/Warp tương đương tuyệt đối |

Các báo động giả `FP-ever` là số ô **đã từng** bị GCS xác nhận sai trên thế giới mô phỏng, **không** phải tỷ lệ FPR hay FDR. `TP-ever` khác `TP-active` nếu xác nhận thay đổi. Chỉ có 3 seed; không được dùng như interval tin cậy có xác nhận thực địa. Khi `rho>0`, lặp observation vi phạm giả định độc lập của phép gộp DS đơn giản; chọn arbitrary discount không phải lời giải khoa học tự động.

## 5. Các thành phần không cần thêm action và các thành phần có thể vô nghĩa trong kịch bản

- **TX power action: KHÔNG**. `tx_power_w=0.1` cố định, `enable_tx_power_control=false`; không tạo hành động học vô nghĩa. Thay đổi TX power chỉ ở ablation network sau này, cần chứng minh action ảnh hưởng packet delivery/energy và có quan sát non-oracle.
- **Altitude Z action: KHÔNG**. UAV cao 50m cố định, không có vertical motion/dynamics baseline. Không thể thêm z mà giữ mô hình 2D & sensor/energy không đổi.
- **Comm action 0** nghĩa là `NO_APP_SEND`, **không** nghĩa radio-off: HELLO and ACK vẫn dùng airtime. Rule-based GCS_SEND không được gọi là actor học communication.
- **PEER_SEND** tồn tại ở interface `compact_3`, nhưng canonical bị mask do `PEER=false`; không được nói model học PEER.
- **Evidence scheduling and next-hop learning V117** đã nghỉ hưu; synthetic backlog không đủ cho natural workloads. Không đưa lại vào Phase1 hoặc baseline nguyên bản.
- **AoI eta=0**: không age-discount bằng chứng; AoI telemetry vẫn đo riêng và bonus event có thể active theo reward defaults. **`lambda_b=0`** nên không phạt airtime trong reward (dù airtime vẫn mô phỏng), không thể tuyên bố learned airtime efficiency.
- **`w_p2p=0`** nên P2P info reward inactive. **`continuous_aoi_penalty_w=0`** nên mapwide continuous AoI penalty inactive. **`enable_accel_energy=false`** nghĩa là chính sách energy hiện bỏ phần điều chỉnh acceleration; phải ghi limitation.
- **`enable_3d_sensing_occlusion=true`** có thể chặn line-of-sight từ UAV cao50m xuống target dưới mặt đất bởi building cao hơn: khác với không có 3D *world motion*; không nhầm là bay 3D.
- **First-visit reward** chỉ trả một lần cho ô; DS tăng confidence qua repeated sensing. Đây là tradeoff coverage vs evidence confirmation, cần ablation reward exploitation, không thay weight nếu chưa đo.
- **RF noise** active base `4.38957709293155e-13 W`, stress `4e-11 W`: cách nhau ~19.6 dB. Không lấy RF stress paper UavNetSim làm cơ sở chứng minh calibration base.

## 6. Bằng chứng nguồn/công thức và giới hạn attribution

| Thành phần | Xuất xứ được phép tuyên bố | Phần do dự án xây dựng/giới hạn |
|---|---|---|
| Belief, DS | Zhang et al., *IEEE Transactions on Intelligent Vehicles* (2024), https://doi.org/10.1109/TIV.2024.3352581 | `SUM` source counts, caps và threshold params, packet evidence ledger, temporal-redundancy tests là adaptation |
| Blackout, AoI, delayed sync | Xue et al., *Electronics* (2026), https://doi.org/10.3390/electronics15091875 | Vị trí 2 blackout, TTL, controller, route-gated sync, không tuyên bố cùng kịch bản Xue |
| MAC/PHY/routing | UavNetSim repository https://github.com/Zihao-Felix-Zhou/UavNetSim ; arXiv:2507.09852 | Custom Warp surrogate nén DATA `70+8k` B, fixed noise model, HELLO timing; **chưa có SimPy/Warp event equivalence** |
| Rotary-wing power | Zeng et al., *IEEE Transactions on Wireless Communications* (2019), https://doi.org/10.1109/TWC.2019.2902559 | Rotor coefficients, 180kJ, optional acceleration term là lựa chọn implementation / không được kiểm chứng từ máy bay vật lý |
| Feasible 66-candidate shield | Project-specific numerical safety projection | Không gán cho paper CBF; `fail-closed` giới hạn số trường hợp đã thử |
| MARL TD3/SAC | Fujimoto et al. 2018; Haarnoja et al. 2018 | Bản gốc multi-agent để Phase2, chưa audit/tune ở V121 |

Đọc code mới sửa và file registry để xác định chính xác từng module; không gán mọi heuristic của dự án cho paper Q1/A*.

## 7. Những gate còn mở và lộ trình ưu tiên trước baseline

**P0 khoa học / triển khai:**
1. C04/C23/C41/C66: temporal correlated sensing / DS calibration và reliability, đặc biệt tỷ lệ false GCS confirmation đã tăng mạnh. Cần nhiều seed, 900-step, mixture sensor, count bias, correlated false alarms, paired MAX/SUM, TP/FP confusion metrics. Chốt định nghĩa `ever-confirmed` vs `currently-confirmed`.
2. C07/C33/C65: nếu tuyên bố tương đương UavNetSim gốc, phải implement event-level paired trace với upstream SimPy. Nếu mục tiêu chỉ là **UavNetSim-inspired Warp surrogate**, ghi giới hạn rõ ràng và kiểm các tính chất packet-DCF independent; quyết định scope này **không biến thành equality proof**.
3. C12/C13/C32: energy, building RF 12dB và noise các giá trị synthetic; sensitivity V121 có nhưng chưa có measured calibration và chưa đủ seed để kết luận khái quát.
4. C18/C31/C81: test CPU/GPU **full observation, reward, GCS evidence, network** multi-step parity còn chưa có; không dùng shield-only parity thay thế.
5. C25/C27/C29/C71: repeated sensing và first-visit reward, non-oracle invariant nhiều bước, generalization theo map/target seeds và full natural RL-like traces.

**Thứ tự xử lý:** lựa chọn/ghi scope scientific → DS correlation & GCS reliability → mạng packet parities/PHY sensitivity → end-to-end natural missions + CPU/GPU observation/reward/network semantics → freeze engineering với tiêu chuẩn trước, mới audit baseline MATD3/MASAC vanilla và sau đó mới tuning/training.

## 8. Runbook và cách tự kiểm

Từ project root, trên Vast `/workspace/v105_audit/venv/bin/python` đã được xác minh có numpy/pytest/Warp:
```bash
export PYTHONPATH=research/zhang_xue/src:.
python scripts/maintenance/verify_source_manifests.py --check
python scripts/maintenance/check_project_policy.py --scope full
python -m pytest -q tests/ research/zhang_xue/tests/
python scripts/maintenance/v121_build_conflict_registry.py
python scripts/gpu/v121_uninterrupted_cuda_gate.py --envs 4096 --steps 900 --seed 44 --device cuda:0 --motion random --output outputs/recheck_base.json
python scripts/gpu/v121_uninterrupted_cuda_gate.py --envs 4096 --steps 900 --seed 44 --device cuda:0 --motion random --noise-power-w 4e-11 --output outputs/recheck_rf.json
```
Lệnh `verify_source_manifests --check` kiểm hash tệp nguồn; cả hai CUDA gate lưu `source_sha256`, `scenario_sha256`, GPU và seed. Test thất bại, giới hạn method và log lịch sử được bảo toàn. Kiểm tra `outputs/v121_scenario_audit/`, `outputs/v119_reaudit/`, `outputs/v120_conflict_resolution/`. **Không có bất kỳ bằng chứng nào rằng baseline MARL đã được huấn luyện ở release này.**

## 9. Kết luận nghiệm thu

**Một số lỗi engineering cụ thể đã được tái hiện, sửa và được kiểm thử thành công; runtime DS-only, CPU MAC contention và shield tính nhất quán hơn. Chưa thể coi kịch bản khoa học đã hoàn tất vì mô hình cảm biến DS correlated, độ tin cậy xác nhận tại GCS, full network/CPU–GPU parity, energy/RF calibration chưa được giải quyết hoặc giới hạn phạm vi đủ nghiêm.**

`SCENARIO_FROZEN=false`; giữ nguyên 30 file thuật toán baseline và YAML active. V121 là bản kiểm toán có chứng cứ, **không phải bản tuyên bố đã giải quyết hoàn toàn mọi conflict**.

### Kết quả regression và trace lần phát hành

Full pytest hậu sửa: **640 passed, 6 skipped, 152 subtests passed**. Kiểm thử trước đó có 6 subfail do fixture projection gọi geometry-only mà kỳ vọng multi-pass; giữ nguyên log FAIL trước khi sửa. Manifest của V121: 436 source checksums, 136 research checksums; policy PASS 0 violations. Witness actual: **10/10** (5 seeds × Base/RF), GCS 15/15, 72–149 simulation steps, dùng oracle validation planner; NOT RL.
