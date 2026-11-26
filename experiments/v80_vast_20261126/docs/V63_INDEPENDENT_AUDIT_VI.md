# v6.3 — Kiểm định độc lập từ ZIP v6.2, tái cấu trúc toàn project

**Ngày:** 25/11/2026. **Nguồn đầu vào:** `uav_v62_full_aoi_entropy_txpower_warp_20261125.zip`
(224.269 bytes). Không trộn thêm ZIP v5.7/v5.8 hay code huấn luyện 3 km từ nhánh cũ.

**Mục tiêu:** từng gate RED → GREEN, cấu trúc thư mục rõ ràng, config/CLI có
validation, file nguồn ≤500 dòng, data/artifact trong `outputs/`, một
`setup.sh` quản lý cài đặt, dữ liệu, test và CPU/GPU preflight.

## Gate A — Đánh giá đầu vào v6.2

- Gói gốc trong môi trường không có Warp: **115 pytest PASS, 2 SKIP, 51
  unittest subtests PASS**. Việc giải nén ZIP thiếu `.git` trước khi `git init`
  khiến một test dùng `git ls-files` báo lỗi; không phải bug thuật toán.
- Không được lấy con số "68 PASS" trong báo cáo cũ làm kết quả cho toàn bộ cây
  nguồn; tập test của hai lệnh khác nhau.
- **ĐÚNG:** PHY/MAC CPU có 7 radio, route, queue, ACK, E2E ACK, DS có
  provenance và bản EWMA lượng tử UQ8.8; một số lỗi hop, CCA, safety đã
  được vá ở v6.2.
- **SAI:** `gpu/layout_v62.py` chỉ có `SHAPES_V62`; `gpu/runtime.py` vẫn
  cấp phát theo `gpu/layout.py` (Q=64, `int32`, cùng khoảng **635.931 B/env**),
  dùng state Warp cũ và `_promote_queued_slot` còn sao chép payload. Không
  có giảm 39,5% VRAM trong runtime.
- **ĐÚNG MỘT PHẦN:** `decode_tx_power_w` và `env.step(tx_power_w=...)` đã có,
  nhưng learner mặc định `power_control='fixed'` và CPU gate trước đây không
  truyền RF action. Đầu ra thứ 3 của Actor trở thành action chết.
- **SAI:** `network_substeps=2` có trong v6.2 config, nhưng không có bất kỳ
  nhánh sử dụng nào trong transition. Về thực tế, mạng vẫn chạy một lần 1 s.
- **SAI Ở CHÚ THÍCH:** hàm Deng entropy của khung nhị phân `{T,F}` có
  H(0,0,1)=log₂(3) nhưng **Hmax=log₂(5)** ở `(0.2,0.2,0.6)`;
  trước đây docstring khẳng định nhầm miền `[0,log₂(3)]`.
- **QUÁ MỨC:** gọi gộp EWMA UQ8.8 là *exact Xue Eq. 9–10*. Lũy thừa có tính
  gộp thuận tiện, nhưng lượng tử Q8.8 ở từng cập nhật có sai số; chỉ có
  một timestamp mới nhất và phải kiểm tra watermark, saturation, source
  independence, equivalence với chuỗi raw observation. Việc gán là "paper
  parity" chưa được chứng minh.
- **CHƯA CHỨNG MINH:** safety shield bảo toàn tập khả thi trong mọi trạng
  thái, MAC event parity với UavNetSim, full CUDA graph, peak T4 VRAM,
  hội tụ MARL, W&B/HF online.

## Gate B — Refactor thuật toán (đã code và test)

```text
uav_network/
  algorithms/
    matd3/         __init__.py, learner.py, diagnostics.py, gradient_probe.py
    masac/         __init__.py, learner.py, gradient_probe.py
    baselines/     vanilla.py, vanilla_masac.py
    core/          models.py, optimization.py, precision.py, replay.py,
                   hybrid_replay.py, actions, gradient & critic utilities, checkpoints.py
    configuration/ schema.py, loader.py
    checkpoints/   __init__.py    # backward compatible import
    vanilla_baselines/ __init__.py # backward compatible import
  experiments/     evaluation, plotting, telemetry, statistics
```

- Di chuyển 20 module thuật toán, không thay hàm toán học, giữ đường import
  `from uav_network.algorithms.matd3 import MATD3` và `...masac`.
- `load_learner_config(path, overrides=[])` merge YAML có kiểm tra schema,
  kiểu thật (không cho `bool` làm `int`), giới hạn gamma/tau/LR/hidden sizes,
  không cho unknown key. Hỗ trợ ví dụ `--learner-set matd3.gamma=0.98`.
- Thực nghiệm giữ độc lập `runtime.algorithm`, hyperparameter MATD3/MASAC,
  và `network.power_control=fixed|learned`.

## Gate C — Folder cho source môi trường (đã code và test)

```text
research/zhang_xue/
  src/zhang_xue/
    environment/   core.py, transition.py, observations.py, configuration.py
    belief/        dempster_shafer.py
    mobility/      flight_safety_energy.py
    communication/ network.py, channel.py, scheduler.py, delivery.py,
                   ledger.py, lifecycle.py, common.py
    gpu/           runtime.py, state.py, layout.py, layout_v62.py,
                   mac_functions.py
      kernels/      reset.py, injection.py, events.py, phy.py,
                    actor_maps.py, scaffold.py
    learners/      actor_adapter.py, state_derived_replay.py,
                   legacy_cpu_train.py
    compat/        env.py, network.py, belief_mobility.py, warp_mac.py
  configs/         scenario_7radio.yaml, cpu_smoke.yaml,
                   scenario_v63_learned_tx_aoi.yaml
  scripts/         run_scenario.py, CPU/GPU gate scripts, legacy/ reproduction
  tests/           research integration and regression tests
```

Đã di chuyển 12 module root `src/*.py` vào thư mục chuyên biệt, đồng thời
sửa import và kiểm tra các bài test lịch sử trên **đường source mới**. Đổi
module import trực tiếp cũ sang `zhang_xue.compat.*` hoặc module canonical:
đây là breaking change đối với chương trình ngoài project còn dùng import
phẳng kiểu `from v56_warp_event_kernels import ...`.

## Gate D — TX power, config thực thi và MAC–mobility

- Đã thêm `step_from_policy_action`: `(XY, route, RF)` → decode unit-disc XY
  → W/mỗi radio → `env.step(..., tx_power_w=...)` → channel CCA/RX/SINR,
  RF airtime, battery và reward. Nếu `fixed/learned` không đồng nhất giữa
  algorithm và scenario thì **ValueError**, không silently bỏ qua output RF.
- Đã chặn RF power `NaN`, `inf`, sai shape/range, nhầm bool trong config.
- Đã hiện thực `network_substeps=2` CPU: chạy radio ở vị trí đầu bước trong
  nửa khoảng thứ nhất, cập nhật topology ở **vị trí giữa quỹ đạo đã shield**,
  rồi chạy nửa khoảng còn lại; cuối bước mới commit motion/energy như cũ.
  Đây chưa phải continuous channel update, và **Warp chưa có logic tương đương**.
- Kiểm thử CPU thực tế với seed 44, 4 transition/thuật toán, 1 optimizer
  update/thuật toán và mean RF nằm trong `[0.05, 0.50]` W. **Không phải
  training hội tụ**. MATD3 actor delayed-update đúng ở first step.

## Gate E — Setup, config/data và packaging

- `configs/data/sources.yaml` mô tả procedural data; không cần dataset ngoài.
  `scripts/data/prepare_data.py` tạo target/obstacle/seed manifest xác định
  ngoài Git (`outputs/data/`). File đó chứa *oracle truth*, chỉ phục vụ
  reproducibility, **không** được cấp vào Actor.
- Download ngoài tùy chọn cần `kind=download`, HTTPS và SHA256; sau tải sẽ
  so hash trước khi giữ file. Không lưu dataset/checkpoint/credentials lên Git.
- `bash setup.sh`: cài cả algorithm core và scenario (editable), tạo
  procedural data, kiểm tra import/policy. Các flag: `--dry-run`,
  `--install`, `--gpu`, `--prepare-data`, `--test`, `--cpu-gate`,
  `--gpu-gate`, `--check-only`, `--seed`, `--episodes`.
- `pyproject.toml` root bao gồm **toàn bộ** package `uav_network*` thay vì
  chỉ 3 package cũ; scenario package discovery gồm kernels/compat/learners.
- Source file Python/YAML/shell ≤500 dòng, tất cả code theo folder.

## Các thành phần hiện có và xung đột chưa giải quyết

| Lớp | Có trong ZIP | Xung đột/giới hạn còn lại | Hướng tiếp theo |
|---|---|---|---|
| Bản đồ | 20×20, 1.000 m, 15 target tĩnh, 5 building | Không tương thích bài toán gốc 3 km/35 target/25 obstacle/40×40; packet cell-id 9 bit chỉ mang tối đa 512 cell | Định nghĩa benchmark/profile riêng; mở 40×40 bằng wire-format mới (11 bit) và GPU buffers theo schema |
| Cảm biến | `Pd=.85`, `Pf=.03`, FOV 50 m, occlusion 3D | Độ cao bay cố định 50 m; không thực sự 3D motion, Pd/Pf theo altitude gốc chưa có | Mô hình altitude-dependent với ablation, trước khi sửa sensor code phải đổi action/energy/safety |
| DS/AoI | canonical masses, provenance, UQ8.8 exponential age | Q8.8 là *xấp xỉ*, `ds_mf=ds_me=.8` là heuristic, chưa chứng minh Bayes → DS calibrated | Kiểm chứng error bound fixed point, nguồn chéo, ACK watermark theo trace |
| Reward | TP, FP, area uncertainty, airtime, peer Deng gain, safety cost | Monotonic `peer_h_min` bảo đảm không tăng reward vô hạn **theo từng receiver-cell**, không phải toàn nhiệm vụ; `r_loc_team` và `r_gcs_tp` khác Zhang | Reward ablation/scale, oracle leakage, target-confirm precision, no-communication controls |
| Motion/energy | shield, swept collision test, battery/PA model | Không có bảo đảm formal viability cho mọi trạng thái (không thể né nếu bắt đầu quá gần và vận tốc quá cao); RX energy chưa kiểm chứng | Invariant/swept hard gate, collision rate, flight/TX/RX energy budget |
| Mạng CPU | 7 radio; discrete-event CSMA/CA, HELLO/DV, ACK/E2E ACK | UavNetSim gốc SimPy, routing/MAC khác triển khai; chưa có event-by-event equivalence | trace timestamps, frozen/retry/FIFO/custody invariants và thống kê PDR/delay |
| Warp | v61 SoA runtime, event kernels, `tx_airtime` | V62 `SHAPES_V62`, `q_order`, learned RF/AoI/info gain chỉ **thiết kế**; chưa dùng runtime | viết structs/kernels mới + parity gate trước khi công bố nén VRAM |
| MARL | MATD3, MASAC, 522d actor, joint critic 3132d, TX policy CPU | Không có online env warp 4096, replay rollout/backprop/eval end-to-end | Vectorized scenario CPU/Warp parity → 2048 env/GPU → optimizer → eval/checkpoint |
| Logging | generic metrics/plots | Không có W&B/HF production integrations trong nhánh này | run manifest, isolated seeds, frozen config, periodic 50-eval snapshots |

### Những phần KHÔNG nên sửa mù quáng

1. **Không thay `gpu.layout` thành `layout_v62` đơn thuần.** Structs Warp,
   `Q=64`, queue promotion, payload indexing và kernel store layout hiện
   phụ thuộc dtype/shape cũ. Thay mỗi bảng sẽ gây sai bộ nhớ/runtime.
2. **Không xem việc nén EWMA là chứng minh tái lập Xue 2026.** Cần phép đo
   sai số lượng tử và độc lập provenance qua packet loss, TTL, retransmission,
   peer fusion; đối chiếu raw timestamp traces.
3. **Không đưa toàn bộ true_targets vào critic/actor bằng cách dùng file
   data generated.** Chỉ phục vụ script kiểm nghiệm/đánh giá.
4. **Không dùng số liệu 1 km để chứng minh kết quả 3 km.** Reward và Pd/Pf
   khác, cần định danh benchmark riêng.
5. **Không tuyên bố QoS/PDR UavNetSim parity** chỉ từ unit test CPU cục bộ
   hoặc từ Warp 2-hop smoke. Đây là simulator khác được tái hiện lựa chọn.

## Thứ tự tiếp theo cần thực hiện để có training GPU có ý nghĩa

1. Ghi **trace CPU–Warp cùng topology và gói tin**: HELLO, DATA, MAC ACK,
   E2E ACK, retries, drop, TTL, source-time, loss SINR. Check invariants
   trước so thống kê chứ không kỳ vọng bitwise RNG.
2. Phân loại ABI device cho features v6.2; migration sang nén cần
   `wp.struct`, queue `q_order`, init/reset/inject/event và replay đồng bộ;
   chạy parity 1/32/256 env CPU và CUDA.
3. Phân tách Warp sensor/flight/safety/reward/obs, đảm bảo world độc lập và
   bốn benchmark levels, **không chỉ Warp MAC**.
4. CUDA T4 1 GPU đo real peak VRAM và throughput từ 32→256→1024→2048,
   không đặt giả định 3 GiB reserve thành đo lường; sau đó 2 process/2 GPU.
5. Nối actor TX+XY+route → GPU env → replay → MATD3/MASAC backward/update,
   định kỳ validation 50 ep/seed riêng, rồi mới bật W&B/HF/checkpoint.

## Nguồn khoa học đã đối chiếu

- Zhang et al., *IEEE Transactions on Intelligent Vehicles* 9(2), 2024,
  https://ieeexplore.ieee.org/abstract/document/10388472 — belief maps
  DS, coverage/localization, DCDDPG với centralized và local critic; thuật
  toán ở đây là **MATD3/MASAC**, không phải tái hiện DCDDPG.
- Xue et al., *Electronics* 15(9), 1875, 2026,
  https://www.mdpi.com/2079-9292/15/9/1875 — AoI, DS, constrained communication.
- UavNetSim nguồn chính thức: https://github.com/Zihao-Felix-Zhou/UavNetSim
  — PHY/MAC/routing/mobility/energy là các layer tách biệt trong Python/SimPy.
- Deng, *Chaos, Solitons & Fractals* 91, 2016,
  https://econpapers.repec.org/RePEc:eee:chsofr:v:91:y:2016:i:c:p:549-553
- NVIDIA Warp: https://nvidia.github.io/warp/stable/

## Báo cáo xác minh đã chạy (độc lập với claims của ZIP v6.2)

- `python -m pytest -q tests/ research/zhang_xue/tests/`:
  **150 PASS, 2 SKIP, 53 unittest subtests PASS** (CPU-only).
- `bash setup.sh --dry-run`, `--check-only`, `--prepare-data --seed 44
  --episodes 2`, `--cpu-gate`: thực thi thành công trên source working tree.
- `python -m pip wheel --no-deps --no-build-isolation` cho **cả hai**
  distributions; kiểm tra nội dung wheel có 29 module thuật toán và 43
  module nghiên cứu. Đây là test packaging, không phải CUDA test.
- `python scripts/maintenance/check_project_policy.py --scope full`:
  kiểm tra tất cả tracked sources ≤500 dòng, không lộ credential file.
- `python scripts/maintenance/verify_source_manifests.py --check`:
  đối chiếu source Git và SHA-256 2 cấp manifest.
- Legacy 80-step CPU runner chạy trực tiếp mà không cần đặt `PYTHONPATH`
  khi dùng đường dẫn `research/zhang_xue/scripts/legacy/`.
- **Chưa đạt/chưa chạy:** CUDA/T4, Warp GPU MAC parity, 4096-env MARL
  training/evaluation và kiểm định QoS tương đương UavNetSim.

## Tái kiểm định

```bash
bash setup.sh --dry-run
bash setup.sh --check-only
bash setup.sh --prepare-data --seed 44 --episodes 4
bash setup.sh --test
bash setup.sh --cpu-gate
# optional on actual GPU (experimental runtime layout is still v61):
bash setup.sh --gpu-gate
```

**Không có secrets/Kaggle account keys hoặc dữ liệu training lớn trong ZIP.**
Gate local chưa có CUDA NVIDIA/Warp nên không chứng minh throughput T4,
UavNetSim equivalence hoặc hiệu quả RL sau học.
