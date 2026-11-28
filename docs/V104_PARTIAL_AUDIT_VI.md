# V104 — Kiểm toán bổ sung V103 (NOT SCENARIO_FROZEN)

Nguồn: ZIP UAV_V103_DS_ONLY_SCENARIO_20261127 do người dùng cung cấp. Không chỉnh thuật toán. DS-only active YAML giữ nguyên. Đây là bản sửa bổ sung, không phải bản chứng nhận CUDA.

## Đã tái lập và sửa
- README, default của 6 GPU gates và maintenance audit trỏ về Bayes trong khi Warp runtime từ chối Bayes: chuyển default sang DS V103. Các probe lịch sử và Bayes CPU vẫn tồn tại, không tuyên bố đã chuyển toàn bộ repository.
- CLI parity nhận --out alias, sticky gate nhận --config và --out alias, output bắt buộc.
- CPU TP active khác TP lịch sử: precision/recall/F1 dùng active; reward latch không được reset khi belief giảm, kể cả enable_tp_unlatch=true (cờ cũ hiện không tác động tới latch). Mission termination và discovery AUC vẫn dùng lịch sử; GPU active metrics chưa được triển khai đồng bộ.
- Regression RED trên nguyên bản: 5 failed; GREEN: 15 passed (5 mới + 10 cũ). Test lifecycle inject ledger có chủ đích, không chứng minh packet path.

## Các gate chưa hoàn thành
- Map reachability, sensor geometry và correlation, safety continuous swept path, CPU/Warp parity (spawn, observation values, packet and timing), CUDA compile và endurance 4096x900, RF/energy accumulation, peer routing, action utility. Không chỉnh tham số vật lý, reward weights hoặc policy để che lỗi.
- Full pytest bị chặn tại collection bởi thiếu warp-lang (10 errors). Sandbox torch 2.10 CPU, CUDA unavailable. Vast connector xác nhận RTX 3060 nhưng chưa đưa đúng ZIP lên instance; chưa chạy CUDA. Không có Git push, không thực hiện 4 subagent độc lập.
- Semantic pooling max/min/mean không bảo toàn BBA; là feature, không được gọi là BBA. Không tự đổi schema trước khi kiểm tra policy compatibility.
- Raw evidence count và dedup packet không chứng minh quan sát độc lập. K cumulative bão hòa khi có nhiều evidence trái chiều; nên log thêm disagreement rate và calibration, không dùng K làm ngưỡng cứng tùy tiện.
- Fixed TX 0.1 W và PEER OFF là design choices baseline, không chứng minh vô ích cho mọi RF/traffic. Chạy paired seeds với TX 0.05/0.1/0.2/0.5, noise/obstacle variants, PEER ON/OFF; đo delivery, AoI, goodput, collisions, retries, energy và mission success. So sánh heuristic có thể triển khai, không chỉ oracle.
- FP32 grouping giảm rounding nhưng không đảm bảo conservation qua hàng trăm bước; cần reference FP64 hoặc residual accumulator kiểm thử trước khi đổi kernel.

## Nguồn đối chiếu
- Zhang et al. IEEE T-IV 2024 DOI 10.1109/TIV.2024.3352581, PDF người dùng, mục III-A, Eq 9–17, Table II–IV: DS, mf=me=.8, max count across UAV; ngưỡng .95, không phải .90. V103 sum origin-indexed là khác biệt thiết kế.
- Xue et al. Electronics 2026 DOI 10.3390/electronics15091875, PDF người dùng, Eq 9–10, 11, 14–15: temporal discount khác global decay, ngưỡng .8/.2, AoI-related mechanism. V103 eta=0 không tái lập đầy đủ Xue.
- Denoeux 2008 cautious combination: https://www.sciencedirect.com/science/article/pii/S0004370207001063 — evidence phụ thuộc.
- UavNetSim https://github.com/Zihao-Felix-Zhou/UavNetSim và https://arxiv.org/abs/2507.09852 — SimPy upstream; không tuyên bố Warp event parity.
- NVIDIA Warp RNG https://nvidia.github.io/warp/v1.16/user_guide/runtime.html — seed + offset; cần kiểm thử runtime 1.18.
- Q1/A* ranking theo năm/ngành chưa xác minh; không gắn nhãn mọi nguồn là Q1/A*.

## Chạy tiếp trên Vast, từ root ZIP
```bash
export PYTHONPATH=.:research/zhang_xue/src
python scripts/maintenance/verify_source_manifests.py --check
python scripts/maintenance/check_project_policy.py --scope full
pytest -q tests/test_v104_scenario_audit.py tests/test_v103_ds_only_scenario.py tests/test_v101_energy_conservation.py
python scripts/gpu/v98_simpy_vs_warp_mac_parity.py --device cuda:0 --des-us 24000000 --output /tmp/v104_parity.json
python scripts/gpu/v99_endurance_sticky_cuda_gate.py --scenario research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml --device cuda:0 --envs 4096 --steps 900 --output /tmp/v104_endurance.json
```
Không khóa scenario nếu chưa xử lý các gate mở. Log trong outputs/v104_audit, source manifest không bao gồm log theo policy.
