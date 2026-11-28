# V76 — KIỂM TOÁN TỪNG BƯỚC: MAP, UAV, DS–AOI, REWARD, UavNetSim, WARP, MARL

**Mốc đọc trực tiếp:** ZIP v75 do người dùng cung cấp ngày 2026-11-26, gồm 243 mục, giải nén vào môi trường cô lập. Mọi bản sửa v76 áp dụng trên *bản sao*, không tác động code `main` hoặc tài khoản Kaggle. **Kết luận mức chứng cứ:** V76 đã bổ sung đường truyền `q_cap=64`/MAC động thật từ CLI đến hàm tạo Warp radio, PHY CPU-reference gate và metric AoI fixed-horizon theo quy ước rõ ràng. **Không có bằng chứng về full GPU environment MARL, policy convergence hoặc thống kê tương đương hoàn toàn với upstream UavNetSim.**

## 1. Thực thể được audit và luồng logic thực tế

| Bước | Hiện có trong source v75/v76 | Vị trí code | Hạn chế/điểm dễ hiểu nhầm |
|---|---|---|---|
| 0. Đặt kịch bản | `num_uavs=6`; 1 passive GCS; map `1000×1000 m`; `grid_w=20`, cell 50m; 15 targets; H=900, dt=1s; seed | `research/zhang_xue/configs/scenario_v67_datamule_1km.yaml`; `environment/configuration.py` | Đây **không** phải branch 3km/35 targets trong lịch sử. Tham số phải báo riêng cho từng experiment. |
| 1. Khởi tạo | UAV ở z=50m, GCS ở z cố định; vật cản hình trụ được scale nếu chọn, target không nằm trong footprint/gần GCS | `environment/core.py:reset` | Không học lệnh z. Kịch bản có **2D động học + 3D geometry/occlusion**, không phải đầy đủ 3D trajectory RL. |
| 2. Động học + an toàn | Lệnh vận tốc XY, giới hạn `v_max=15m/s`, `a_max=5m/s²`, kinematic safety shield; kiểm tra swept collision và dừng nếu vi phạm | `mobility/flight_safety_energy.py`; `environment/transition.py` | Không dùng CBF; shield có fallback braking **không chứng minh vô va chạm** ở trạng thái ngoài viability kernel. |
| 3. Cảm biến | FOV 50m, Pd=0.85, Pf=0.03, 3D building occlusion có bật | `environment/transition.py`; `communication/common.py` | Một sensor tại độ cao 50m không chứng minh đa độ cao Pd/Pf. Cần sensitivity ablation và kiểm tra false-positive. |
| 4. Local DS belief | `ds_mf=ds_me=0.8`, `confirm_th=0.9`, `ds_fusion_mode=sum`, `ds_aoi_eta=.02`, `aoi_ewma_end_to_end=true`, `evidence_calibration=paper_fixed` | `belief/dempster_shafer.py`; `communication/ledger.py` | **Không tương đương nguyên bản** Zhang-MAX hoặc Xue direct-BPA. Count-power có bias định lượng; xác thực sensor likelihood thực nghiệm. |
| 5. Mạng CPU | `SevenRadioCsmaCaNetwork`: queue64; PHY LoS/NLoS + blackout; CSMA/CA + half duplex + SINR; local HELLO DV + relay; E2E ACK provenance | `communication/{network,channel,scheduler,delivery,ledger,lifecycle}.py` | Custom implementation **lấy cảm hứng** từ upstream UavNetSim, không import upstream SimPy protocol stack. |
| 6. Policy action | continuous 2D XY; discrete compact3 (`0=NO_SEND,1=GCS,2=SHARE_PEERS`); `enable_tx_power_control=false` | `environment/transition.py`, `learners/actor_adapter.py` | Radio power hiện không được agent điều khiển; không nên gọi là tối ưu TX power. Peer selection có thể dựa HELLO-local và không oracle nếu flag đúng. |
| 7. Reward | Team decrease in DS uncertainty, TP delivered GCS, AoI freshness/continuous, FP, shield, airtime, peer; `w_p2p=0`, `lambda_b=0` ở primary scenario | `environment/transition.py` | Không có **direct positive peer** hoặc airtime penalty ở baseline này. Người học vẫn có incentive gián tiếp qua GCS delivery/AoI. |
| 8. Termination | GCS confirmed all true targets, all battery depleted hoặc safety fail -> terminated; H=900 -> truncated | `environment/transition.py` | Done không giống successful delivery; report success/termination cause riêng. |
| 9. Radio Warp | SoA qcap mặc định32 / opt-in64, K=62, compressed ledger UQ8.8, `SimDimsV67` MAC timing dynamic, data-mule packet workload | `gpu/{layout_v67,state_v67,runtime_v67}.py`; `gpu/kernels/*_v67.py` | **Chỉ radio GPU**, không có Warp vectorized flight/sensing/DS/reward/reset end-to-end. |
| 10. MATD3/MASAC capacity | Replay lấp đầy bằng lặp lại 4 transition CPU thật; actor inference; đúng 2 learner optimizer updates | `uav_network/experiments/{capacity_probe,capacity_fixture}.py` | Capacity diagnostic **không phải** RL training 4096 independent GPU envs. |

## 2. Đánh giá v75: cái nào sửa đúng? Cái nào chưa đủ?

| ID | Phán quyết | Bằng chứng / nguyên nhân | V76 action |
|---|---|---|---|
| V75-1 watermark monotone | **ĐÚNG để giới hạn thưởng revisit**, không phải định lý PBRS | `team_u_prev[cid]` chỉ giảm nên tổng phần thưởng uncertainty toàn tập không thể tăng tuyến tính theo số lần già hóa khi giữ trọng số dương. Nhưng Ng–Harada–Russell (ICML 1999) yêu cầu shaping dạng \(\gamma\Phi(s')-\Phi(s)\) để bảo toàn tối ưu; one-sided watermark không đủ. | Giữ code, làm ablation *task reward vs watermark*; không khẳng định policy invariance. |
| V75-2 fail-closed | **ĐÚNG** | `zhang_max` phải `eta=0` ở CPU và Warp; chặn TX control, per-peer, 2 substeps, coalesce chưa port. | Giữ fail-closed, tăng manifest. |
| V75-3 `q_cap=64` | **ĐÚNG ở SoA, thiếu wiring** | `Q_V67=32` là mặc định; `Q_V67_MAX=127` do `active_slot: wp.int8`, qcap64 trong layout nhưng probe chưa nhận CLI. | **Đã sửa** `--warp-q-cap`, `--require-queue-parity`, truyền vào Warp constructor; RED→GREEN test. |
| V75-4 dynamic MAC | **ĐÚNG ở kernel, thiếu wiring** | `SimDimsV67` đã lưu 6 timing và kernel đọc `dims.*`; `capacity_probe.py` chỉ truyền `network_kwargs` không chứa `protocol`. | **Đã sửa** `--allow-dynamic-mac-timing` để dựng `CsmaProtocolConfig` trong actual constructor. |
| V75-5 `bba_reliability_cap: null` | **ĐÚNG** | Loader nullable cần giữ `None` thay vì suy kiểu từ default float. | Test v75 giữ nguyên. |
| V75-6 buildings/blackouts | **ĐÚNG API nhưng test còn quá dễ** | `run_packet_workload()` nhận geometry nhưng probe luôn gọi `buildings=(), blackout_centers=()` mặc định. | **Đã bổ sung** `warp_phy_cuda_gate.py` kiểm tra LoS/building/blackout theo NumPy; không buộc delivery thành công qua vật cản (vì có thể đúng là mất gói). |
| Legacy test V71 | **SAI PHÉP THỬ khi chạy v75** | Đọc string `MAX_RETX`,`CW_MIN` cũ trong khi kernel đã chuyển sang `dims.max_retx`,`dims.cw_min`. | **Đã sửa** assertions theo implementation hiện hành, không đổi MAC semantics. |
| Missing Warp Python dependency | **Môi trường kiểm thử, không phải code fail** | 2 functional tests cần `warp-lang`, nhưng sandbox tải ZIP không cài Warp. | `pytest.skipif(not HAS_WARP)` (2 test skipped ở container), vẫn chạy thực trên máy người dùng có Warp 1.18 CPU. |

## 3. Paper / reference và công thức (không suy diễn hạng A* hoặc Q1 nếu chưa tra riêng)

**R1 – Zhang et al., IEEE Transactions on Intelligent Vehicles 2024**, DOI [`10.1109/TIV.2024.3352581`](https://doi.org/10.1109/TIV.2024.3352581). Bài gốc về cooperative reconnaissance/coverage/localization với DS belief probability maps và double-critic DDPG. Biến thể kiểm toán của code *không* phải bản reproduction chính xác khi dùng DS-SUM + AoI EWMA.

**R2 – Xue et al., Electronics 2026, 15(9),1875**, DOI [`10.3390/electronics15091875`](https://doi.org/10.3390/electronics15091875). Bài về AoI-DS-enhanced cooperative search dưới spatial communication constraints, công bố các baseline PPO/MAPPO/QMIX. Không được suy ra journal Q1 chỉ từ tên tạp chí; kiểm tra SJR/JCR của năm cụ thể. Theo mô tả phương pháp bản gốc, direct BPA discount \(m'_f=w m_f\), với \(w=e^{-\eta \Delta t}\), khác count-power fusion trong code.

**R3 – Zhou et al., UavNetSim-v1**, [`arXiv:2507.09852`](https://arxiv.org/abs/2507.09852), [official GitHub](https://github.com/Zihao-Felix-Zhou/UavNetSim). Đây là **Python SimPy discrete-event platform** có multiple routing algorithms, CSMA/CA / Pure ALOHA / TDMA, interference/collision; không phải Warp SIMD/CUDA environment. Table I example: TX 0.1W, noise \(4\times10^{-11}\)W, SINR threshold6dB, 600×600×100m. Source chủ động dùng radio PHY khác thì phải ghi là *adapted*, không gọi upstream defaults.

**R4 – Ng, Harada & Russell (ICML 1999)** [paper](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf). Potential-based reward shaping (PBRS) có điều kiện toán học, khác one-sided min watermark của project. ICML là conference; xếp hạng cụ thể phụ thuộc hệ thống CORE/năm, không đồng nhất với journal Q1.

**R5 – Yates et al., AoI survey**, [`arXiv:2007.08564`](https://arxiv.org/abs/2007.08564). Phải định nghĩa chính xác tuổi thông tin, timestamp và censoring của cells chưa được giao; duration-truncated averages bị lệch khi số bước không bằng nhau.

**R6 – Zeng, Xu & Zhang, rotary-wing propulsion**, [`arXiv:1804.02238`](https://arxiv.org/abs/1804.02238). Code dùng steady-level rotor power và có thể bật acceleration augmentation riêng, không thể coi mô hình là đo đạc điện thực tế.

**R7 – NVIDIA Warp 1.18.0 official**, [PyTorch interoperability](https://nvidia.github.io/warp/v1.18/user_guide/interoperability/pytorch.html) và [runtime](https://nvidia.github.io/warp/v1.18/user_guide/runtime.html). Dùng `wp.stream_from_torch(torch.cuda.Stream(...))` và `wp.ScopedStream` cho shared stream; giữ resource trên non-blocking stream, cẩn thận CUDA graph capture và readback. Không được gọi `.numpy()` mỗi step nếu muốn hot-path không host sync; `.numpy()` **được phép trong gate validation**.

### 3.1 Bất đồng định lượng count-power vs direct BPA

Với \(m_f=0.8\), \(w=e^{-0.02\Delta t}\), \(f_{\mathrm{count}}(w)=1-(1-m_f)^w\), \(f_{\mathrm{direct}}(w)=m_f w\). Đặt \(a=-\ln(1-m_f)\). Đạo hàm của \(g(w)=f_{\mathrm{count}}-f_{\mathrm{direct}}\): \(g'(w)=a e^{-aw}-m_f\), \(g''(w)=-a^2 e^{-aw}<0\). Do đó đỉnh tại \(w^*=\ln(a/m_f)/a\approx0.434331\), \(\Delta t^*\approx41.697\,s\); \(g_{\max}\approx0.155467\). Khi \(w\to0^+\), tỷ số \(f_{\mathrm{count}}/f_{\mathrm{direct}}\to a/m_f\approx2.01180\). Đây là **hệ quả của hai công thức** và ví dụ số học; không phải kết quả benchmark đã chạy trên policy.

### 3.2 Noise PHY mismatch có thể đổi kết quả PDR mạnh

\(N_{\mathrm{scenario}}=4.38957709293155\times10^{-13}\,W\), \(N_{\mathrm{paper}}=4\times10^{-11}\,W\); \(10\log_{10}(N_{\mathrm{paper}}/N_{\mathrm{scenario}})=+19.59637\,dB\). Dưới điều kiện **noise-limited, FSPL exponent 2**, riêng thay đổi floor nhiễu có thể làm threshold range thay đổi \(\sqrt{91.12}\approx9.55\times\). **Không** suy ra PDR/range thực tế tăng đúng 9.55× trong NLoS, interference-limited, MAC contention hoặc capped links.

## 4. V76 changes đã có code và RED→GREEN

1. **`uav_network/experiments/radio_config.py`**: `require_queue_parity` explicit fail-closed, report flag. Không mặc định đổi behavior benchmark v75; người chạy phải opt-in.
2. **`uav_network/experiments/capacity_probe.py`**: cho nhận `warp_q_cap`, `allow_dynamic_mac_timing`, `require_queue_parity` và truyền *validated* `protocol` object đến `WarpCompressedRadioNetworkV67`. Mọi option ghi vào `radio_config.json`; source manifest/provenance được giữ.
3. **`scripts/gpu/run_capacity_probe.py`**: thêm 3 CLI flags; CPU functional/4096 CUDA capacity dùng chung logic, không giả lập GPU khi CUDA thiếu.
4. **`scripts/gpu/warp_phy_cuda_gate.py`**: benchmark có kiểm chứng LoS/NLoS/blackout so với reference, chạy CPU hoặc CUDA, trường `gpu_executed` phân biệt CUDA thực thi, không nhận nhầm là full RL.
5. **`tests/test_v76_{probe_contract,phy_reference,phy_gate_source,scenario_audit}.py`**: test config wiring, sớm chặn q mismatch, geometry reference, bảng issue.
6. **`environment/aoi_metrics.py` + `environment/transition.py`**: chỉ thêm báo cáo terminal-only `mean_gcs_aoi_fixed_horizon_no_future_delivery_s` (không thay reward/termination). \(m=H-t\), tổng tuổi giả định sau khi done là \(m\sum_c A_c(t)+C m(m+1)/2\). Hệ số chia toàn horizon \(HC\). Nếu chưa done thì metric `None`; minh bạch đây là no-delivery extrapolation, không phải quan sát.
7. **`scripts/maintenance/scenario_conflict_audit.py`**: xuất JSON có mã vấn đề, mức độ, nguyên nhân, file/paper và khuyến nghị; không tự gắn nhãn `parity_verified=true`.
8. **`tests/test_v71_verified_audit_fixes.py` và `tests/test_v73_capacity_diagnostics.py`**: cập nhật test cũ chống báo lỗi giả; functional tests chỉ skip khi thiếu dependency (không im lặng báo pass).

## 5. Memory sizing: chỉ *radio SoA*, không phải tổng VRAM/throughput

| Map grid, q_cap=64, E=4096 | Radio bytes/env | Radio GiB (một GPU) | Đánh giá |
|---|---:|---:|---|
| 20×20 (1km) | 597,397 | **2.2789 GiB** | Có khả năng fit radio trên T4 16GiB; chưa tính learner/replay, CUDA runtime, geometry, JIT, activation. |
| 40×40 (2km) | 1,612,597 | **6.1516 GiB** | Có thể fit 4096 radio, nhưng phải đo actual free VRAM + peak. |
| 60×60 (3km) | 3,304,597 | **12.6060 GiB** | Rất sát ngân sách; có thể OOM với reserve 3GiB hoặc actor/learner đồng cư trú. Không có định lý “bắt buộc 2×2048”; alternatives: giảm E, nén ledger, shard radio, thay K/cells. |

**Replay**: `TensorReplayBuffer` thực tế `FP32 state + obs` trùng dữ liệu, dù `StateDerivedReplayBuffer` tồn tại trong code nhưng **chưa được cắm vào capacity_probe/learner interface**. Không được báo rằng việc chuyển FP16 đã hoàn tất. Các lệnh `state=obs.reshape` cần kiểm tra **ở mọi nguồn dữ liệu**; trong CTDE nếu critic có privileged truth thì state-derived bị sai semantics. Cần chọn và benchmark sample loss/quantization (FP16 so với FP32) rồi mới chuyển.

## 6. Độ tin cậy và các gate theo thứ tự (chưa thực hiện ≠ đã sửa)

| Gate | Điều kiện PASS bắt buộc | Trạng thái |
|---|---|---|
| G0 source provenance | nguồn ZIP v75, SHA256, test logs/patch, không có secret | V76 release gate nội bộ |
| G1 CPU scientific semantics | deterministic seed, map/target/safety/reward, zero oracle, no reward farming | Baseline CPU tests; chưa proof mọi scenario |
| G2 Warp radio CPU | q64, MAC dynamic, 2-hop route + ACK, no overflow; PHY LOS/NLOS/blackout vs NumPy | **PASS thực thi trên máy người dùng** (v75 kernels) |
| G3 Warp radio GPU | T4 ×2, q64 dynamic, PHY and packet gates, free/device-wide VRAM | **CHƯA CHẠY v76 CUDA** |
| G4 CPU↔GPU radio parity | compare deterministic packet/ledger/ACK and statistical loss under realistic fading/contention with acceptance tolerances | OPEN |
| G5 full Warp env | motion+safety+energy; sensing+occlusion+DS; semantic obs+mask; AoI+reward; partial reset + queue/ACK/ledger | OPEN |
| G6 GPU online RL | >=4096 truly independent environment transitions, per-env resets, replay from GPU, gradients flow, no `numpy()` hotloop | OPEN |
| G7 training-quality | seeds ≥3, equal transition budget, val separate seeds, correct y-axis env steps, no actor tanh saturation | OPEN |
| G8 paper-grade ablation | 5 ablation families, PDR/AoI/coverage/delivery/detection, positive controls + CIs | OPEN |

**Ablation tối thiểu:** (1) Zhang MAX vs DS-SUM vs Xue Direct-BPA vs EWMA variant; (2) `w_p2p=0/positive`, `lambda_b=0/positive`, continuous AoI on/off; (3) No-Peer / Always-GCS / Always-Peer / Round-Robin / Learned; (4) `q_cap=32/64/127` và stress có traffic/load/backoff/blackouts; (5) radio noise model / NLOS geometry, 1km/2km/3km, fleet 4/6/8/12, MATD3 vs MASAC. Dùng các scenario seeds cố định và evaluation khác training; không so số return tuyệt đối nếu reward scale khác nhau.

## 7. Quy trình vận hành Kaggle account 2 (T4×2)

Notebook release `UAV_V76_KAGGLE2_WARP_Q64_DYNAMIC_MAC_GPU_GATES.ipynb` nhúng source v76 có SHA256, không có `key.txt` hay API tokens. Khi chạy thủ công trên Kaggle của account 2, bật GPU T4×2 và Internet nếu cần cài `warp-lang`. Run All gồm: source checksum -> package dependency/accelerator preflight -> CPU/Warp smoke + PHY -> genuine CUDA PHY -> 4096 env radio+MATD3 capacity -> 4096 env radio+MASAC capacity -> ZIP logs, **tất cả vẫn là diagnostic**. Không tự đặt trạng thái PASS nếu bất kỳ bước nào BLOCKED hoặc FAIL. Nếu phải hạ E/batch, ghi nhãn pilot chứ không thay ngầm.

**Ranh giới thẩm định:** đã có proof-of-execution Warp CPU q64/dynamic MAC từ máy người dùng; GPU T4 v73 `DIAGNOSTICS_PASS_NOT_FULL_RL` là dữ liệu lịch sử khác source. Việc nhúng v76 vào notebook chưa có nghĩa GPU đã chạy v76; phải có stdout/phase log/artefact trên Kaggle rồi mới được xác nhận.
