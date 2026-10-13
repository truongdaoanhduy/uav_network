# Tăng tốc train UAV MARL — 13/10/2026

Đã triển khai tối ưu trong `/home/aduy/Documents/NCKH/uav_network`, giữ native UavNetSim và sáu actor/critic độc lập. Bản mặc định tăng throughput của pipeline ngắn trên Tesla T4 khoảng **18,8% với MASAC, 37,2% với MATD3** so với snapshot bàn giao. Chế độ critic vector hóa có phép đo riêng bên dưới.

## Code và phạm vi

- Baseline bàn giao: `fb5c3f53c44b1acdcd32b9d62e204e0f75ef9d8c`.
- Bản mặc định đã kiểm thử/benchmark GPU: `70003629039e1ea265520f27c7d033c972e9b7f9`.
- Bản thêm critic vector hóa: `90791735fa1cfc42e845a2fb0417fa32e8a6fed3`.
- Native UavNetSim giữ revision `04daafb815eb377409b40b285574eeb62b9a8d58`.
- Dữ liệu máy đọc: [training-acceleration-20261013-evidence.json](training-acceleration-20261013-evidence.json).
- Nguồn thư viện/paper: [gpu-throughput-options-20261013.md](gpu-throughput-options-20261013.md).

Không thay dynamics, reward, mask, TTL, retry, ACK, energy, activation, model size, batch/replay, seed hoặc update/collection ratio của profile train. Deadline/warmup nhỏ hơn dùng riêng cho benchmark được ghi rõ dưới đây.

## Nút thắt đã đo

Trong profile collector trước tối ưu, 80 bước dưới cProfile mất 9,52 giây; native network chiếm 8,30 giây (~87%), có khoảng 560 nghìn lượt kiểm tra queue. Phần lớn chi phí là queue rỗng vẫn thức dậy theo chu kỳ 1000 microsecond cho bảy node, gồm GCS.

Project đã có spawn workers, shared observation/state buffers, inference theo batch môi trường, một lượt action readback và overlap CPU collection/GPU learning. Những cơ chế này được giữ lại.

Đó là profile của collector riêng. Trên T4, update MASAC trước tối ưu mất ~109,55 ms, MATD3 ~53,30 ms, trong khi một vector collection ~26–27 ms. Vì collector và learner có overlap, không thể kết luận toàn bộ train luôn bị CPU giới hạn chỉ từ utilization GPU hoặc profile collector.

## Thay đổi

1. **Gộp polling rỗng khi toàn bộ radio idle.** Backend kiểm tra queue, inbox, MAC process, ACK/retry process và PHY transmission. Giữ lịch polling cũ khi có bất kỳ traffic nào, khi clock/poll không nguyên; giữ tick cuối trước ranh giới để không đổi pha polling/order ở bước kế tiếp.
2. **Gom backward và finite checks của actor.** Mỗi actor vẫn clip gradient và bước Adam riêng. Kiểm tra norm của cả nhóm một lần trước khi cập nhật actor, giảm đồng bộ CPU/CUDA. Nonfinite gradients chặn cả nhóm trước mọi bước Adam.
3. **Critic vector hóa tùy chọn.** `torch.vmap` + `torch.func.functional_call` chạy sáu mạng cùng kiến trúc theo một chiều batch model. Mỗi lần đọc weights hiện tại, giữ đường gradient về parameters gốc; không dùng cached `stack_module_state` tạo leaf khác optimizer. MASAC vẫn enumerate destination của chính actor, peer actions detach; MATD3 giữ target smoothing và policy delay.
4. **Fused Adam tùy chọn.** Mặc định không truyền `fused=False`: để PyTorch chọn backend Adam cũ, bao gồm CUDA foreach khi phù hợp. Chỉ truyền `fused=True` khi được yêu cầu trên CUDA.
5. **Benchmark tái sử dụng.** `scripts/benchmark_acceleration.py` đo collector, synchronized warmed learner và pipeline thực có native packet simulation; xuất timing/config/hashes.

`runtime.coalesce_idle_polls=true` mặc định. `vectorize_critics=false` và `fused_optimizers=false` mặc định giữ chế độ tương thích. Bật vectorized critics phải để `compile_models=false`, có validation chặn tổ hợp không hỗ trợ.

## Kết quả CPU và CUDA mặc định

Máy Ubuntu hiện tại không có CUDA khả dụng; không coi CPU tests skip là bằng chứng CUDA. GPU tests và benchmark chạy trên Kaggle, Torch 2.11.0+cu128, Python 3.13.15, bốn CPU, Tesla T4. Host có hai T4 nhưng từng mode dùng cùng GPU0, chạy lần lượt.

| Phép đo | Baseline | Tối ưu mặc định | Thay đổi |
| --- | ---: | ---: | ---: |
| Ubuntu: collector 160 bước, median 3 lượt | 6,178 s | 2,201 s | 2,81× throughput |
| T4 host: collector 160 bước, median 3 lượt | 3,592 s | 1,306 s | 2,75× throughput |
| T4: MASAC learner update | 109,55 ms | 105,59 ms | 3,8% throughput |
| T4: MATD3 learner update | 53,30 ms | 51,89 ms | 2,7% throughput |
| T4: pipeline MASAC, joint transitions/s | 30,08 | 35,73 | +18,8% |
| T4: pipeline MATD3, joint transitions/s | 39,90 | 54,76 | +37,2% |

Collector/learner timings là median ba lượt cùng root seed44; không phải ba research seeds. Mỗi learner trial có 2 warmup updates và 50 timed updates, CUDA synchronize trước/sau vùng đo. Pipeline mỗi mode/algorithm là **một** lượt ngắn, throughput loại startup; có ghi elapsed gồm startup trong JSON.

Pipeline dùng 2 env, MLP256×256, batch512, replay100000 trên CUDA, 1 update/vector collection; riêng benchmark giảm deadline xuống384 và warmup xuống512 để có 129 update trong768 joint transitions. Profile `motion-study.yaml` vẫn deadline800, warmup4000. Không suy diễn hội tụ hay mức tăng tốc train dài từ bài đo này.

Fused-only có MASAC104,70 ms/update và MATD3 51,64 ms/update; throughput pipeline36,18/56,28 transitions/s. Lợi ích nhỏ trong lần đo này, không chọn làm mặc định vì rounding/hashes khác backend cũ.

Job đã hoàn tất: [paired T4 default modes](https://www.kaggle.com/code/haibro1234/uavnet-speed-paired-20261013-7000362).

## Critic vector hóa

Giữ mode này vì có lợi ích trên cả hai learner. Phép đo riêng cùng T4 GPU0, reference7000362 so với source9079173, cả hai đã bật idle coalescing và để fused Adam tắt:

| Phép đo | Critic loop | Critic vmap | Tăng throughput |
| --- | ---: | ---: | ---: |
| MASAC update, median3 lượt | 100,47 ms | 77,92 ms | +28,9% updates/s |
| MATD3 update, median3 lượt | 49,53 ms | 34,63 ms | +43,0% updates/s |
| MASAC pipeline, joint transitions/s | 37,93 | 43,75 | +15,3% |
| MATD3 pipeline, joint transitions/s | 54,89 | 69,15 | +26,0% |

Collector vẫn cùng trace; collection13,7–13,8 ms/vector step ở cả hai mode. Chênh lệch collector160-step ở hai mode (~1,21/1,16s) là nhiễu timing vì vectorized critics không được dùng trong collector probe.

Không nhân các tỷ lệ từ hai job để quảng cáo một speedup tổng hợp so với baseline: điều kiện và tần số hoạt động của host giữa hai phiên có thể khác. Các tỷ lệ trong mỗi bảng là so cặp ở cùng job.

Full CUDA suite kiểm tra outputs/gradients/3 complete updates với tolerance và repeated-seed native training. Pipeline FP32 vmap có model/episode hashes khác loop đúng như contract đã ghi. Native network semantics vẫn được giữ; chưa có kết quả hội tụ dài cho execution mode này.

Job hoàn tất: [paired T4 vmap](https://www.kaggle.com/code/haibro1234/uavnet-speed-vmap-20261013-9079173).


## Kiểm chứng hành vi

- Scripted collector trước/sau khớp toàn bộ observation/state/reward/info/native event trace, hash `340b5ab2fabb7e81a04581094582258116e4dcbe8375c5867776402826b15c67`, trên tất cả ba lượt mỗi mode.
- Native traffic tests xét ACK/retry, relay, fragmentation, expiry/recovery; radio bận giữ polling gốc, kể cả bước silent khi frame dài vẫn còn trên kênh.
- Test clocks/polls nguyên và lẻ giữ trace; idle single-step event count giảm từ7096 xuống dưới256.
- Grouped actor update khớp loss, weights, clipping và Adam state của cách update riêng trên CPU/CUDA.
- Bản mặc định so với baseline trên T4 khớp cả episode và model hashes của **cả MASAC/MATD3** trong pipeline thực.
- Vectorized critic có test outputs, live-weight refresh, gradients theo actions và từng parameter; float64 tolerance1e-12. Cả complete learners được so qua3 update ở FP32 tolerance2e-5; đây là local numerical equivalence, không phải bảo đảm hash của training dài khớp legacy.
- CUDA tests kiểm tra vectorized native training lặp cùng seed cho cùng episode/model hashes trong cùng mode.
- CPU source9079173: 139 tests, 129 pass, 10 CUDA skips, 63,276s.
- CUDA source7000362: 135/135 pass, không skip, 81,674s.
- CUDA source9079173: **139/139 pass**, không skip, 81,359s.

Job thử trước ở source87f705d có assertion không khớp hash do vô tình truyền `fused=False` làm thay backend Adam mặc định. Source7000362 đã sửa và đo lại đầy đủ. Không dùng số liệu job cũ đó làm kết quả cuối.

## Cách chạy

Trên máy có CUDA, dùng profile nhanh đã thêm (chỉ thay execution mode/output/tag so với motion-study):

```bash
cd /home/aduy/Documents/NCKH/uav_network
.venv/bin/python -m uav_network train --config configs/motion-study-fast.yaml \
  --algorithm masac \
  --set runtime.output_dir=outputs/motion-study-fast-masac
```

Thay `masac` bằng `matd3` và đổi output directory cho thuật toán còn lại. Máy Ubuntu được kiểm tra trong phiên này chưa có CUDA khả dụng; có thể dùng các overrides tương tự với lệnh `kaggle launch` hiện có, giữ account và private credential dataset đã cấu hình.

Đo lại trên GPU đích, không khởi chạy train dài:

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_acceleration.py \
  --label vectorized --device cuda --vectorize-critics \
  --output outputs/speed/vectorized.json
```

Bỏ `--vectorize-critics` để đo mode mặc định. Driver từ chối ghi đè report đã có. Muốn đối chiếu polling cũ trên cùng source, dùng `--no-coalesce`. Muốn đối chiếu fused Adam dùng `--fused`.

Chế độ mặc định đã tự bật coalescing; không cần thay số env/batch/update. Chế độ vectorized có cùng công thức và parameters nhưng khác thứ tự phép nhân/cộng FP32: dùng output/run riêng, không kỳ vọng hash khớp legacy. Thí nghiệm dài để đánh giá convergence/delivery chưa được chạy trong lượt tối ưu này.


## Paper bạn gửi và lựa chọn thư viện

Trang10 của paper **Heterogeneous UAVs Trajectory Optimization for Post-Disaster Target Search Based on MARL With Graph Attention Network** ghi NVIDIA RTX6000 và Intel Xeon Gold5220R, PyTorch1.7/Python3.8. Paper không xác định RTX A6000, không cung cấp profile utilization, CPU-only comparison hay bằng chứng env/network chạy CUDA. Trang12 nói nhu cầu compute khi deployment nhỏ hơn và không cần GPU server cho giai đoạn đó. Training và deployment là hai khối lượng công việc khác nhau.

Với project hiện tại, SimPy xử lý event tuần tự trên CPU; GPU giúp inference/update tensor, và batching nhiều mạng giảm overhead kernel. Tăng tốc thực tế được đánh giá bằng transitions/s và update time, không đặt mục tiêu utilization100%.

| Thư viện/phương pháp | Khả năng | Phù hợp hiện tại |
| --- | --- | --- |
| PyTorch vmap/functional_call | Batch mạng độc lập, giữ tensor compute trên CUDA | Đã triển khai và kiểm thử; không thêm dependency |
| VMAS | Batched 2D physics bằng PyTorch | Hợp cho một simulator mới, thiếu native packet/ACK/MAC/TTL của project |
| JaxMARL | JAX env và learner trên accelerator | Cần rewrite environment, network và learner |
| WarpDrive | CUDA/Numba env cùng PyTorch learner | Cần port toàn bộ network; Salesforce repo đã archive01/05/2025 |
| EnvPool | C++ thread pool cho env | Cần viết env C++; bọc Python SimPy không tự có tốc độ đó |
| Sample Factory | Shared buffers, inference batching, asynchronous sampling | Project đã dùng các ý shared-buffer/overlap; thay trainer sẽ đổi MASAC/MATD3 |

Nếu cần toàn bộ env trên GPU, phần lớn công việc nằm ở chuyển event queue và packet state thành tensor/array, giữ ordering và kiểm chứng lại MAC/ACK/retry/TTL/SINR/energy. Cài thêm VMAS/WarpDrive rồi giữ SimPy bên dưới sẽ không làm native event engine chạy CUDA. Các hệ số tăng tốc trên Tag/MPE/Atari trong paper của họ không áp dụng trực tiếp cho bài toán UAV này.
