# V113 — FULL CODE + V112 HISTORY + BUG-FIX QA + SCIENTIFIC SCENARIO AUDIT

**Ngày:** 2026-11-28. **Giai đoạn:** Hoàn thiện kịch bản trước khi tạo baseline canonical MATD3/MASAC. **Trạng thái: `NOT SCENARIO_FROZEN`.** Không training learner, không sửa các tệp `uav_network/algorithms/`, không push GitHub hay đổi nhánh gốc. Đây là bản V112 cộng một fix P0 và bộ kiểm tra giả định DS, không tự nhận đã giải quyết hết 37 xung đột.

## 1. Cây chứng cứ (đọc theo thứ tự)

- Source root `uav_network_V113/`. Start with source YAML `research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml`, CPU `research/zhang_xue/src/zhang_xue/environment/transition.py`, Warp `uav_network/gpu/mission_v79.py`, network CPU communication `research/zhang_xue/src/zhang_xue/communication/`, DS `research/zhang_xue/src/zhang_xue/belief/dempster_shafer.py`.
- Entire V112 audit remains in `docs/V112_EVIDENCE_DRIVEN_SCENARIO_AUDIT_VI.md`, prior V111 `docs/V111_CONFLICT_RESOLUTION_AND_PROVENANCE_VI.md`; detailed reviewer audit `outputs/v110_reviewer_audit/`.
- V113 **37-issue** matrix `outputs/v113_validation/04_ALL_CONFLICTS_V113_37.csv` and `.json`: issue class, severity, actual source, root cause, evidence, treatment, gate.
- GPU prior paired 5 seeds and 4096-env stress **evidence summaries** `outputs/v112_validation/`; user-local raw 20 files in `/home/aduy/Downloads/UAV_V112_RAW_VAST_LOGS/`, also `/workspace/v112_validation/output/` on Vast, originally *not bundled* in the uploaded V112 ZIP. These raw V112 files are **not** within the ChatGPT artifact: do not confuse summary JSON with original stdout/raw JSON.
- New V113 direct source regression tests, DS synthetic screen outputs and Vast pytest output `outputs/v113_validation/`; remote full QA result 593 pass/6 skip/114 subtests.
- Original nested V108/V109/V110/V111/V112 archives kept outside source in ZIP folder `HISTORICAL_ORIGINAL_ZIPS/` to avoid SHA governance violations. Verify each SHA against `ARCHIVE_INPUT_SHA256.json` at ZIP root.

## 2. Nghiêm trọng P0: CPU commit physical motion dù trajectory invalid (C37)

**Phát hiện gốc:** V112 CPU sử dụng `kinematic_safety_shield` và `trajectory_safety_violations`, nhưng kiểm tra `safety_violation=True` xảy ra trước một vòng lặp **luôn** gán `self.positions[u] = old_xy + v_safe * dt` và tiêu thụ pin. Vì thế tại `old x=995 m`, `vx=15m/s`, sau acceleration clamp còn `vx=10m/s`, CPU đưa UAV tới `x=1005 m` ngoài map 1000 m, rồi mới báo `terminated=True`. Một kịch bản bắt đầu ở `x=985` cũng bị vi phạm **swept stopping feasibility**, dù next position x995 chưa ngoài map. Đây là bug logic commit, không phải thiếu paper.

**Sửa thật ở V113:** Nếu `safety_violation and c.safety_failure_terminate` (trường hợp canonical), **không** ghi cập nhật vật lý, giữ `position, velocity, battery` trước tick, chặn local sensing của world bị terminate, phát cờ vi phạm/terminal; nguồn tạo packet/MAC đã chạy trước khối này và **không rollback**, nên không claim full atomic physics-network rollback. Khi legacy `safety_failure_terminate=False`, giữ nguyên hành vi nonfatal để không phá `test_v67_other_uav_sense`. V113 không biến shield thành CBF, không chứng minh safety recurrence hay khả thi cho mọi state.

**Chứng cứ:** `tests/test_v113_cpu_fail_closed_safety.py` 2 test active terminal + no regression to safe motion, `outputs/v113_validation/02_cpu_p0_before_after.json`; local 15 tests PASS; Vast full `593 passed 6 skipped 114 subtests` với `transition.py` SHA `3de25a1d7fce9e5d7e7d95b75a56ffeb9cef6040c5b3e8221947c8628eec297c`. Lượt test ban đầu 3 fail và lý do khắc phục được lưu `06_failed_attempts_and_root_cause.md`.

## 3. Review logic/action: cái gì cần thiết, cái gì không nên thêm vô nghĩa?

| Thành phần | Kết luận về ý nghĩa & nguồn | Active hay không và yêu cầu reviewer |
|---|---|---|
| `tx_power_w=0.1` | Cố định là **đúng có mục đích**: tham số SNR, CCA, năng lượng TX. UavNetSim `utils/config.py` đặt `TRANSMITTING_POWER=0.1`; upstream không ép thành policy action. | `enable_tx_power_control=false`. **Không thêm** TX head học chỉ để tăng số action. Chỉ mở khi mục tiêu nghiên cứu thật sự là energy/SINR power control kèm ablation. |
| Continuous XY | RL action đúng: tốc độ mong muốn 2D, qua giới hạn động học `vmax=15,amax=5`, alt50m cố định; liên quan Zhang 2024 về UAV search. | Actor MASAC/MATD3 canonical sau freeze học XY; tránh claim actor học bay 3D khi alt cố định. |
| Communication mode compact3 | `NO_APP_SEND/GCS_SEND/PEER_SEND` xác định vận chuyển thông tin; **không phải action continuous của TD3/SAC**. Khi `PEER=false`, PEER action masked/vô hiệu. | Phase1 dùng heuristic truyền thông chung; Phase2 ghi rõ learned-action dimension=2, comm controller fixed. Không claim hybrid-action learner khi chưa có. |
| PEER/MULE | Đóng vai trò thông tin multi-hop chỉ khi mạng bị phân mảnh và forwarding hữu ích. Nhưng V112 bật PEER giảm GCS delivery `56.667->24.000` trong RF-stress; MULE tăng không có ý nghĩa 0.667pp (CI chứa 0). | Canonical active `PEER=false,MULE=false`. Để thành experimental ablation (negative result), không nhét tính năng phụ gây hại vào main claim. Nếu làm contribution mạng phải thiết kế dedup/flow control, test lại. |
| DS-only | Zhang 2024 DS belief mass, positive/negative evidence count. Xue 2026 về AoI, nhưng `ds_aoi_eta=0` nên **không** claim AoI-corrected DS. | `belief_inference_mode=dempster_shafer`, Bayesian branches legacy, **không active**. `mf=me=.8`, `Pd=.85 Pf=.03` không được sensor calibrate, threshold .90 chỉ scenario-specific. |
| GCS altitude/position | GCS trạm cố định, có Z=15m dùng link geometry / placement, **không là UAV biết bay**. | Không có GCS flight action. Measurement of GCS antenna altitude must be explained, not claimed search altitude. |
| Origin-ledger SUM | Giải được double-receive cùng packet origin/seq nếu ledger đúng, nhưng thay MAX counting của Zhang; độc lập qua UAV cần chứng minh. | Sự khác với Zhang là **extension tự thiết kế**, không được nói exact replication; **correlated observations có nguy cơ DS overconfidence**. |
| `first_visit` reward | Chỉ thưởng ô mới giúp chống hover farming, nhưng biến reward từ ignorance reduction thành coverage incentive. | Tự thiết kế, cần ablation so DS uncertainty gain; hiện `w_p2p=0,lambda_b=0,continuous_aoi_penalty_w=0` nên loại khỏi công thức active. |
| `rf_per_building_loss_db=12` | Stress-only physical attenuation model không phải số đo môi trường thực. | Cần sensitivity 0/6/12/20 dB. Không viết paper như model đo đạc. |
| 802.11 MAC Warp | GPU DCF-inspired scheduler giúp hàng nghìn env đồng thời, không giống event SimPy/Sionna upstream. | Cần packet airtime/frame/ACK/CW/traffic parity. Warp 1 evidence entry ~312us vs upstream default 1024B ~4328us tại2Mbps; không so trực tiếp. |
| Safety shield | Giảm action nguy hiểm, CPU bug đã fix cho fatal; không phải Formal CBF/APF. | Không claim collision-free vô điều kiện. Stress nhiều starting state, swept geometry, numeric corners. |
| Energy | Zeng 2019 rotor propulsion; avionics, battery180kJ, optional accel energy false là giả định. | Kết quả mạng có thể bị năng lượng/flight confound; cần reachability bounds và energy budget. |
| GCS delivery target score | Nên phản ánh nhiệm vụ **target confirmed AND information reaches GCS**, khác coverage đơn thuần, phù hợp multi-UAV network task. | Xuất cả discovery, delivery, coverage, AUC và delay; không nhận reward constants từ paper khi chúng do tác giả tự chọn. |
| Actor observation 553 / critic3318 | Local549+alive1+mask3; masks là contact-specific, không oracle map. | V111 đã sửa C19, giữ nguồn encode chung; test taint hiện limited fixture. |

## 4. Bằng chứng scientific V112 và điểm không thể suy rộng

- **5 seeds × 4 worlds × 900 steps** với heuristic non-oracle. Base 98% delivered; RF-only 56.667%; RF+PEER 24%; RF+MULE+PEER 24.667%; Base+PEER 96.333%; Base threshold .99 95.333%. PEER RF delta −32.667 percentage points (95% bootstrap clustered 5-seed CI [−40.667, −26.333]); MULE +0.667pp CI [−5,+6]. *Đây không phải MATD3/MASAC học.*
- C14 trước V113 4 loại action: mỗi 4096×900, 4/4 PASS, zero sticky safety/overflow/nonfinite, sampled VRAM 2.579GiB. **Không chứng minh worst case safety**. V113 GPU kernel/YAML không đổi, nhưng **chưa chạy lại tất cả 4 GPU mode trên bản V113**.
- DS thresholds trong count domain (`mf=me=.8`, no decay) 0.90 và 0.95 tương đương; 0.99 chặt hơn, nhưng correlation test V113 chứng minh không triệt tiêu FP ở rho cao.

## 5. V113 DS temporal correlation diagnostic: cách làm, công thức, kết luận

Mô hình **stress tự thiết kế** (không do Zhang/Denoeux cung cấp): mỗi lần ghé lại một cell, với xác suất `rho` lấy lại binary detection trước, còn lại lấy Bernoulli mới theo `Pf=.03` hoặc `Pd=.85`. Khung DS vacuous prior `m(T)=0,m(F)=0,m(U)=1`, masses evidence `.8/.8`. Với positive p và negative n, đặt `a=(1-mf)^p`, `b=(1-me)^n`, `D=a+b-ab`, `m(T)=(1-a)b/D`, `m(U)=ab/D` và chỉ confirm khi `m(T)>=threshold` & `m(U)<=.2`.

100000 chuỗi/case, 20 lần ghé, rho 0/.5/.9, thresholds .9/.99, Monte Carlo seeded 44+, confidence Wilson 95% từng tỷ lệ. Tỷ lệ **empty cell ever falsely confirmed** theo kết quả JSON:

| rho | threshold .90 | threshold .99 |
|---|---:|---:|
| 0 | 0.103% | 0.004% |
| .5 | 2.308% | 1.224% |
| .9 | 4.195% | 3.669% |

Đây chỉ là **single-cell diagnostic**, không phải 6-UAV mission, không phải correlated sensor thật đo đạc, không có GCS/PHY/MAC. Tăng threshold lên .99 **không giải quyết được phụ thuộc evidence**, nên C04/C05 vẫn OPEN. Thực hiện full mission injection khi sẵn sàng mà **không đổi `confirm_th` active tùy tiện**.

## 6. Paper provenance: exact original vs adaptation vs modelled assumption

| Paper / upstream | URL và điều thực sự hỗ trợ | Phần do code chủ động điều chỉnh / cần test |
|---|---|---|
| Zhang et al (2024) IEEE Transactions on Intelligent Vehicles | https://doi.org/10.1109/TIV.2024.3352581 ; DS BBA belief, positive/negative evidence, search | `SUM` origin ledger khác paper `MAX`; `mf=me=.8` borrowed, `Pd/Pf` and threshold .90 synthetic, first_visit tự thiết kế |
| Zeng et al (2019) IEEE Transactions on Wireless Communications | https://doi.org/10.1109/TWC.2019.2902559 ; rotary-wing flight power | 180kJ battery, 15W avionics, disable acceleration energy are scenario assumptions |
| Zhou et al UavNetSim-v1 | https://github.com/Zihao-Felix-Zhou/UavNetSim commit `c2703f24eb58cdb557a5f67ded5c7b0368a964ed`; radio `TRANSMITTING_POWER=0.1`, IEEE MAC reference | GPU custom approximation **NOT** upstream event implementation; 22MHz thermal noise/Base vs table RF stress are different PHY conditions |
| Xue et al (2026), *Electronics* | https://doi.org/10.3390/electronics15091875 ; AoI/DS, spatial communication blockage | Active `ds_aoi_eta=0` disables AoI decay of belief; not reproduction of Xue; venue NOT automatically top-tier A* |
| Denoeux (2008), *Artificial Intelligence* | https://doi.org/10.1016/j.artint.2007.05.008 ; non-distinct bodies of evidence require careful combination | Sticky Markov rho0/.5/.9 V113 is invented stress distribution, NOT sourced from Denoeux |
| TD3 / SAC (Fujimoto/Haarnoja 2018 ICML) | https://proceedings.mlr.press/v80/fujimoto18a.html ; https://proceedings.mlr.press/v80/haarnoja18b.html ; continuous baselines scientific foundations | Canonical MATD3/MASAC learner integration strictly later than scenario freeze; not claiming train completed |

**Cấm claim thiếu bằng chứng:** "100% correct", "guaranteed collision free", "UavNetSim identical SimPy", "PEER improves communication", "0.99 eliminates all FP", "DS masses calibrated", "algorithm has trained", "Q1/A* journal rank verified". Venue credibility ≠ default parameter correctness.

## 7. Thứ tự gate trước khi khóa kịch bản

1. **V113 QA/P0** — PATCH C37 và full Vast pytest đã **PASS**; source manifests & policy cần PASS ở ZIP cuối. No baseline edits.
2. **Chốt primary scope** — canonical pure-GCS PEER=false, MULE=false currently active; negative ablation giữ riêng. `tx_power_w=.1` fixed; 2D continuous XY only. Chọn threshold sau correlated mission-level audit, không lựa chọn từ tỷ lệ FP-ever single cell.
3. **C04/C05** — 6-UAV 900-step mission correlated sensing stress rho sweep, DS SUM/MAX ablation, source ledger dedup unit tests; measure target false confirmation & delivery not only cell rates.
4. **C07/C33** — matched PHY/MAC traffic UavNetSim SimPy↔Warp: same frame bytes, ACK and DCF and route config, offered load, RNG, packet destinations; compare PDR, queue, retries, delivery delay with paired seeds. Or downgrade claim honestly to custom UavNetSim-inspired model without strict parity.
5. **C08/C12/C13** — test feasibility to collect/deliver targets within energy/time horizon, obstacle attenuation and RF noise sensitivity and Pd/Pf/masses, plus map geometry and GCS receive reachability.
6. **Re-run GPU post-change** — 4096×900 sampled random + boundary + churn + RF stress, and explicit 4-direction boundary/collision targeted states; compile stable source SHA and configurations. Formal shields require proof not runtime only.
7. **Mark `SCENARIO_FROZEN`** only after agreed scientific gates, independent configuration snapshot and source SHA. Then Phase2 baseline native MATD3/MASAC, then tuning. No learner changes before freeze.

### Lệnh tái lập

```bash
cd uav_network_V113
export PYTHONPATH=.:research/zhang_xue/src
python -m pytest -q tests/test_v113_cpu_fail_closed_safety.py tests/test_v113_ds_correlation_screen.py tests/test_v67_phy_safety_regressions.py
python scripts/maintenance/v113_ds_correlation_screen.py --trials 100000 --visits 20 --seed 44 --output outputs/v113_validation/03_ds_temporal_correlation_diagnostic.json
python scripts/maintenance/verify_source_manifests.py --check
python scripts/maintenance/check_project_policy.py --scope full
# FULL GPU env requires torch + warp: on Vast use /workspace/v105_audit/venv/bin/python
python -m pytest -q tests/ research/zhang_xue/tests/
```
