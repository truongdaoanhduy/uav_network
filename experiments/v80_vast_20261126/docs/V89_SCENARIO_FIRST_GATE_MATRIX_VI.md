# V89 — KIỂM TOÁN KỊCH BẢN TRƯỚC BASELINE (SCENARIO-FIRST GATE)

**Trạng thái khoa học: CANDIDATE — NOT FROZEN — NO GO BASELINE/TUNING.**

## Phạm vi kịch bản đề xuất khóa

Bản đồ 1.000 × 1.000 m; 6 UAV, GCS; 15 mục tiêu tĩnh; belief map 20 × 20; UAV có độ cao cố định 50 m; 1 bước mô phỏng = 1 s; horizon 900. Mọi thay đổi thuộc kịch bản nghiên cứu, **không sửa baseline MATD3/MASAC hoặc tối ưu hyperparameters trong giai đoạn này**.

Nguồn đề xuất: `research/zhang_xue/configs/scenario_v89_static_1km.yaml`.
Runner thí nghiệm không-RL: `scripts/gpu/v89_environment_gate.py`.
YAML V88 cũ được giữ nguyên làm đối chứng; nhánh riêng `experiment/v89-scenario-first-20261127`.

## Gate kịch bản và chứng cứ

| Gate | Kết quả | Bằng chứng |
|---|---|---|
| GCS route mask / resolver: fail closed | PASS RED→GREEN CPU+CUDA | `tests/test_v89_gcs_route_mask.py`, `/workspace/v89_scenario_lock_RED_GCS.log` |
| Spawn đủ khoảng trống cho phanh hai UAV, không xây dựng/blackout | PASS RED→GREEN CPU+CUDA | `tests/test_v89_spawn_geometry.py`; từ ~71,59 m lên ~123,69 m |
| Reward gồm 8 thành phần và tổng khớp scalar | PASS RED→GREEN CPU+CUDA | `tests/test_v89_reward_ledger.py` |
| Mục tiêu tĩnh có chứng cứ bền vững, không decay theo tuổi packet | PASS bounded | `tests/test_v89_static_packet_semantics.py`; counts < uint16 capacity với 900 bước |
| Data mule/GCS nhận số đếm evidence thật qua RF | PASS bounded | `tests/test_v89_static_packet_semantics.py` |
| FoV và ray che khuất, Pd/Pf định lượng | PASS bounded | `tests/test_v89_fov_sensor_statistics.py`, 1024 world CUDA |
| 5 peer từ HELLO giao thức RF, không oracle | PASS bounded | `tests/test_v89_hello_no_oracle.py`; kiểm tra khi peer thật di chuyển nhưng chưa nhận gói |
| Reset chọn world không chạm world khác | PASS CPU+CUDA | `tests/test_v89_selective_reset_hello.py` |
| Môi trường độc lập RL 4 world × 900 bước | PASS 8/8 | `/workspace/v89_scenario_env4_s900.json`; KHÔNG learner |
| Parallel Warp 4096 world × 3 bước | PASS 8/8 | `/workspace/v89_scenario_env4096_s3.json`; KHÔNG learner |
| Upstream UavNetSim MAC/PHY/RF event parity | OPEN (không được tự nhận) | Dựng tình huống có cùng seed, cường độ tín hiệu và kết quả ACK/CSMA/TTL giữa SimPy và Warp |
| Suy hao 12 dB/công trình và noise được hiệu chuẩn theo nghiên cứu/đo thực địa | OPEN | Cấu hình hiện chỉ là surrogate minh họa |
| DS m_f=m_e=.80 có calibration đúng với Pd=.85/Pf=.03 | OPEN | Test độ tin cậy, false confirmation, độ nhạy m_f/m_e, xác suất sau hợp nhất |
| HELLO và DS delivery trong blackout đa hop, tốc độ cao, tất cả 6 UAV | OPEN STRESS | Packet trace và movement policy ngẫu nhiên, coverage không oracle |
| Pairwise physical safety và shield feasibility trong mọi trạng thái hợp lệ | OPEN | Stress trajectories / formal viability; shield fallback rollback có thể vi phạm a_max |
| GPU 4096 × 900 bước và multi seed, peak memory/time | OPEN | smoke 3 bước không đủ |

**Nguyên tắc:** Không thể coi môi trường khoa học hoàn chỉnh chỉ từ pytest PASS hoặc stationary-action rollout. GCS packet delivery >0 không chứng minh đầy đủ tính trung thực MAC/PHY. Những thay đổi quan trọng (spawn, chứng cứ, quan sát) làm bài toán khác V88; số liệu các run trước đây KHÔNG so sánh trực tiếp.

## Các thay đổi code V89 chưa phải baseline

- `_observe_core_v79`: GCS mask xác thực route local còn hạn.
- `resolve_compact3_v67`: không gửi GCS nếu route stale.
- `_reset_world_v79`: spawn giãn x=90 + 120u; tránh pair gần sát biên an toàn.
- `_reward_v79`: xuất 8 thành phần vào buffer riêng, **không đổi scalar reward**.
- `_sense_visible_v79` / DS fusion: chế độ opt-in `raw_u16_count` dùng uint16 integer count với epoch thời gian riêng, tránh Q8.8 bão hòa do `eta=0`.
- `NetworkBuffersV67`, `mac_v67.py`: capture HELLO pose sender vào thời điểm phát, cache receiver khi PHY decode thành công; 20B telemetry minh họa tăng 80us airtime, không khẳng định bit-exact với UavNetSim.
- `mission_v89_hello_peer.py`: 5 × 7 peer features từ cache RF; missing và stale → 0, hiện không có oracle.
- `run_v81_real_rollout.py`: obs_dim suy ra từ runtime, không còn gắn cứng kích thước V88; **không chạy baseline**.

## Những việc phải hoàn thành để ra SCENARIO FROZEN

1. Xác nhận tiêu chuẩn peer safety ở đầu vào/đầu ra (safety rejection có được xem là episode failure hay vẫn tiếp tục?) và kiểm thực nghiệm stress.
2. So sánh Warp surrogate với UavNetSim-v1 theo cùng topology, packet trace/CSMA, retry, TTL, AoI, interference, ACK, PDR/delay, blackout.
3. Chốt mô hình chứng cứ: phân biệt target confidence với freshness của dữ liệu; kiểm định hiệu chuẩn Pd/Pf và sự hợp nhất không đếm trùng theo origin/generation.
4. Kiểm đa seed, edge cases geometry, network partitions và selective reset 900 bước; kiểm E4096×900 dưới ràng buộc bộ nhớ.
5. Chỉ sau đó đóng băng YAML, schema observation/action, reward contract, test & SHA256, đặt **SCENARIO_FROZEN**; sau đó mới triển khai baseline.

## Đối chiếu học thuật

- UAV communication simulator upstream: https://github.com/Zihao-Felix-Zhou/UavNetSim
- NVIDIA Warp memory/capture/synchronization: https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/concurrency.html
- Target belief & multi-UAV work (Zhang): https://ieeexplore.ieee.org/abstract/document/10388472
- TD3/SAC chỉ là tài liệu baseline dùng SAU freeze: https://proceedings.mlr.press/v80/fujimoto18a.html; https://proceedings.mlr.press/v80/haarnoja18b.html

**Kết luận:** Giai đoạn này vẫn ưu tiên giải quyết các gate mở; KHÔNG được đánh dấu hoàn tất, KHÔNG chạy training/tuning MATD3 hoặc MASAC.
