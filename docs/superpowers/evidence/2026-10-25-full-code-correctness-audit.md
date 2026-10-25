# Full owned-code correctness audit — 25/10/2026

Mốc audit: 2026-10-25 17:50 +07. Phạm vi là toàn bộ **source do project sở hữu và track bằng Git**, config, script, test và pipeline training/cloud. `third_party/UavNetSim` không được track trong worktree này; project ghim upstream commit `04daafb815eb377409b40b285574eeb62b9a8d58` và audit wrapper/parity của project, nhưng báo cáo này không tuyên bố đọc từng dòng implementation bên thứ ba.

## 1. Inventory và cách kiểm tra

- AST parse: **100 source module**, **97 test module**, **8 script Python**; không lỗi cú pháp.
- `python3 -m compileall -q uav_network scripts tests`: pass.
- **21/21 YAML profile** load và `Config.validate()` thành công.
- Primary profile: `configs/comparison-20k-hybrid75-1kb.yaml`, SHA256 `3aa6754dedd5150cea8c2a86ce09f3712d1e365672f5af292279f2ab8b806bc6`.
- Primary workload vẫn: report/payload **1024 B**, buffer **3072 B**, 6 UAV, 35 target, 25 obstacle, horizon 1800, env4096, batch8192, replay1M, CPU replay75%, seed44, final evaluation30 episode / seed2000000044.
- MASAC/MATD3 source không được retune trong audit này. Không launch/cancel/reset cloud job.

Audit không chỉ dựa vào test: đã trace source theo domain config/world/observation/state/motion/sensing/reports/PHY/MAC/reward/termination/algorithms/replay/checkpoint/PQL/training/evaluation/tracking/cloud. Static findings chỉ được gọi là bug nếu tái hiện bằng behavior/test.

## 2. Dec-POMDP / CTDE

### Không tìm thấy leakage mới sau các fix trước

- MASAC và MATD3 execution gọi mỗi actor bằng `observation[:, i]`; global `state` chỉ đi vào centralized critic trong update.
- Native và Warp actor observation không chứa `delivered` toàn cục. `delivered` nằm trong critic state, reward, termination và metric.
- Peer/GCS link-state ngoài local radio range bị zero; destination mask cũng range/power/admission gated.
- Local report sensing/recovery không đọc remote GCS receipt; GCS delivery dedup chỉ còn ở transfer/accounting.
- `observation_version=3` chặn checkpoint semantics cũ/khác nghĩa.
- Privileged feasibility/oracle helpers nằm ở feasibility/relay-audit; learned training/evaluation không gọi chúng.

Các giả định giữ nguyên, không tự coi là bug: ideal local beacon/admission metadata trong radio range, common team reward, centralized critic privileged state, coupled multi-UAV safety transition, known GCS/global frame. Đây phải được mô tả trong paper.

Actor hiện là reactive MLP, không RNN/history controller. Đây là **policy-class restriction** trong partially observable problem, không biến actor thành centralized; nó có thể giới hạn performance so với history-based policy.

## 3. Scenario / physics / sensing / reward / termination

- Scenario generation seed-based; explicit scenario shape/finite/map/obstacle/start-safety checks có validation.
- Motion checks segment obstacle and UAV-UAV collision; dynamic-window safety có native/Warp parity tests.
- Battery funding: unaffordable move không được thực hiện; radio TX/RX debit battery; failed decode vẫn có RX/TX energy theo path tương ứng.
- Report TTL/custody/relay: relay không refresh TTL, GCS dedup metrics, report recovery yêu cầu fresh local sighting.
- Native/Warp reward component layout khớp: discovery, delivery, blocked, step, expiry, drop, raw radio energy, propulsion, normalized communication efficiency, shield correction.
- Finite horizon `max_steps` được actor quan sát và được xử lý là terminal (`terminated=True`, `truncated=False`) nhất quán giữa native/Warp. Replay lưu terminal next observation/state trước reset.

Không có correctness mismatch mới được xác nhận. Reward hiện phạt communication cost qua cả raw-joule term và normalized efficiency term; đây là reward-shaping có chủ ý, không sửa vì không có bằng chứng nó là implementation error. Với primary 1KiB, reference one-report Pmax TX energy là khoảng 0.005379 J, nên normalized term có scale khác hẳn raw `0.01*J`; cần xem như lựa chọn reward khi diễn giải kết quả, không phải bug code.

## 4. MASAC / MATD3 / PQL / replay

- MASAC: stochastic local actors, centralized twin critics, min-Q target, entropy regularization/temperature, Polyak target update; hybrid destination branch là extension riêng của project.
- MATD3: local deterministic actors, centralized twin critics, min target, target continuous noise, delayed actor/target updates; categorical next-hop + epsilon exploration là extension riêng của project.
- PQL path tự ghi `training_architecture="pql-inspired"`: giữ MASAC/MATD3 loss/model và parallelize collector/critic/policy qua bounded queues + snapshot mailbox. Primary ratio/warmup khớp BASE ở scale: 32 collection ×4096 =131072 transition và critic ratio8 tương ứng updates_per_step8.
- Replay/checkpoint giữ terminal transitions, replay/optimizer/RNG/environment state; exact resume validators và observation-version guard có mặt. Async PQL được ghi rõ không bitwise deterministic.

Không xác nhận algorithmic implementation bug mới. Trong paper không nên gọi `MASAC PQL`/`MATD3 PQL` là implementation nguyên bản của PQL; chính xác hơn là MASAC/MATD3 với **PQL-inspired asynchronous training architecture**.

## 5. Training / evaluation / cloud lifecycle

- BASE/PQL trainers đều tạo explicit `episode_budget_complete` và `stop_reason` ở final result.
- Continuation kiểm prefix episode/global-step, environment state khi zero new episodes, new immutable checkpoint, account/source/config consistency.
- CPU finalization kiểm episode count/seed/device, checkpoint identity và W&B/HF publication.
- Evaluation dùng deterministic decentralized actor; comparison pair theo seed/layout hash; CPU metric axes/provenance fix trước vẫn giữ đúng.

### Confirmed defect mới: tracking completion fail-open

Trước audit này, `OnlineTracker.complete()` dùng:

```python
result.get("episode_budget_complete", True)
result.get("stop_reason", "episode_budget")
```

Nếu một legacy/future caller thiếu completion metadata, W&B có thể bị gắn nhầm là full episode-budget completion. Current BASE/PQL trainers truyền explicit fields nên các run 20k đã xác minh không bị finding này làm sai.

TDD:

- RED: `test_missing_completion_metadata_fails_closed_in_tracking_summary` tái hiện `episode_budget_complete=True` khi input thiếu field.
- Minimal fix: chỉ coi completion khi `completion is True`; missing -> `episode_budget_complete=False`, `stop_reason="unknown"`, `session_status="session_completion_unknown"`.
- Focused GREEN: **3 passed**.
- Lifecycle regression: **86 passed, 38 subtests passed** trên tracking/continuation/controller/CPU handoff/archive.

## 6. Findings đã điều tra rồi loại

- Ruff báo F821 trong `legacy_benchmark.py` cho nested `learn()`; outer scope đã khai báo `learner = replay = None`, direct MASAC/MATD3 legacy-vs-PQL benchmark test pass. Đây là static false positive, không sửa.
- Nghi ngờ MATD3 thiếu config validation: đọc tiếp hàm cho thấy validation gamma/tau/replay/batch/hidden/policy-delay/noise/epsilon đã có. Test thử pass ngay trước source change nên finding bị loại.
- `build_visualization_spec()` còn fallback historical `{episodes:1, seed:10044}` khi parent không có `cpu_eval`. Canonical current launcher truyền explicit `cpu_eval` và primary protocol là30/2000000044; fallback được giữ vì tests/spec legacy Oct14 dùng nó. Đây là **legacy provenance footgun/backward-compatibility risk**, không phải current canonical-path bug. Nên fail-closed hoặc gắn nhãn `legacy-preview` nếu sau này bỏ compatibility.

## 7. Reproducibility / methodological limits

- UavNetSim revision và Warp 1.18.0 được pin. Nhưng phần lớn Python dependencies chỉ pin theo range (`numpy>=2.2,<3`, `torch>=2.5,<3`, `gymnasium>=1.2,<2`, cloud/network ranges), không có lockfile đầy đủ. Chạy lại tương lai có thể gặp dependency drift dù source commit giống nhau. Đây là reproducibility risk; không tự đổi dependency trong correctness audit.
- Một training seed44 không đủ để kết luận thống kê thuật toán A tốt hơn B; final 30 evaluation scenarios chỉ đo policy đã train dưới seed đó.
- State-based centralized critic là hợp lệ với CTDE nhưng literature MARL gần đây chỉ ra critic dùng true state dưới partial observability có thể tạo bias/variance so với history-aware critic. Đây là methodological risk cần discussion/ablation, không phải execution leakage.
- Memoryless actor là restricted decentralized policy class; Dec-POMDP optimal policy nói chung có thể phụ thuộc local action/observation history.
- Sensor hiện tại được code/docs định nghĩa là **exact deterministic detection** trong camera footprint, có obstacle occlusion; không có Pd/Pf/false-alarm stochastic model. Repo hiện tại nhất quán với semantics đó. Nếu manuscript ngoài repo còn mô tả Pd/Pf thì manuscript và code đang khác model và phải sửa một phía trước khi công bố.
- Propulsion model là surrogate có chủ ý `20 + 120 + 0.5*v^2 + 2*a^2 W`, không phải closed-form rotary-wing model của Zeng/Xu/Zhang. Với battery277200J: đứng yên ~1980s, 10m/s ~1459s,15m/s ~1098s,20m/s ~815s trước acceleration/radio. Vì horizon=1800s, sustained movement có thể làm UAV hết pin trước deadline; đây là **scenario difficulty/calibration risk** có thể giải thích late-episode battery exhaustion, không phải arithmetic bug (README/design/tests đều xác nhận cộng idle+hover).
- External UavNetSim internals không nằm trong tracked owned source; chỉ pinned revision + wrapper/parity behavior thuộc bằng chứng audit này.
- Host hiện tại không thay thế full real-CUDA audit; CUDA-only tests có thể skip. Existing T4 locality/physics diagnostic trước đó là 78 passed,0 skipped, nhưng không được diễn giải thành full100-module GPU proof.

## 8. Verification

Pre-fix whole suite trong audit: **739 passed,77 skipped,8 warnings,1273 subtests, exit0**.

Post-fix final whole-suite result: **740 passed,77 skipped,8 warnings,1274 subtests, exit0, 285.53s**.

No cloud job was submitted, restarted, cancelled or deleted during this audit.
