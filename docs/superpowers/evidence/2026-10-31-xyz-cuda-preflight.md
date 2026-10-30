# XYZ CUDA preflight — 31/10/2026

## Phạm vi và trạng thái

Tiếp tục yêu cầu đã được duyệt: XYZ/Bayesian camera, learned RF, 20 target/15 obstacle,
giữ visualize 2D, rồi kiểm CUDA trước study dài. Parent source:
`42c1dacf3f516c2e4c4f8933e2433f14d51575cb`.
Runtime fix commit: `5fee6c753b4e7cd86801a68ab5a0fb72521bda86`.
Không đổi FOV, Pd/Pf, prior0.5, confirmation, action, reward, RNG, packet scheduler,
reset algorithm, struct layout hoặc checkpoint schema trong thay đổi compiler này.

## Lần Kaggle đầu

[Job BASE MASAC/MATD3](https://www.kaggle.com/code/kesfljhk/ben-g3-both-base-h75-10301741-894057)
được gửi 00:41 ngày31/10 (+07), source42c1dac, benchmark-only, timeout1800s.
Hardware thực tế: hai Tesla T4, mỗi GPU14.562GiB, compute capability7.5;
Torch2.11.0+cu128, Python3.13.15.

Watcher quan sát `cancelled` lúc01:11:53 (+07) và tải đầy đủ đầu ra.
Không gửi lệnh hủy job. Thời điểm gần timeout cấu hình; output không nêu lý do hủy.
Mười test CUDA đầu đã hoàn tất, gồm Native-world MASAC/MATD3 updates, finite weights,
reproducibility, replay transfers và restart. Test Warp preflight cuối chưa hoàn tất.
Không có profile XYZ benchmark JSON, throughput hoặc VRAM fit.
Đây không phải20k episode training và chưa chứng minh hội tụ.

Benchmark giữ E4096/B8192/replay1M/Hybrid75/FP32. Override riêng cho preflight:
`runtime.compile_models=false`, `tracking.checkpoint_store=wandb`,
`tracking.log_checkpoint=false`. Không sửa YAML cha.
Account được ghim `account_01`, không xoay account; quota lúc01:16:54 (+07)
còn27.11h. W&B IDs trong receipt chưa chứng minh BASE benchmark đã log online.

## Chẩn đoán compiler trên máy không GPU

Dùng Warp1.18.0/NVRTC13.4 AOT cho SM75. Các số này chỉ là compile,
không phải CUDA execution hoặc performance của policy.

| Thử nghiệm | Kết quả quan sát |
| --- | --- |
| Reset riêng, CUBIN mặc định | Pass2.60s process;2.18s AOT |
| MAC gốc, CUBIN O3 mặc định | SIGKILL sau473.50s; chưa xác nhận nguyên nhân OOM |
| MAC gốc, CUBIN O2 | Dừng theo RSS cap3GiB sau164.24s |
| MAC gốc, CUBIN O1 | Dừng theo timeout180.21s; peak469.57MiB |
| MAC gốc, event noinline/O3 | Dừng theo RSS cap3GiB sau60.98s |
| MAC gốc, PTX/O3 | Pass55.06s; peak1180.24MiB; PTX38,688,920B |
| MAC dùng struct references, CUBIN/O3 | Pass20.77s; peak646.32MiB; binary877,096B |

PTX gốc có801,058 `st.local`, local depot190,960B trong kernel packet_step.
Điều này khoanh vùng code expansion/copy descriptor ở chuỗi helper và native codegen;
không phải bằng chứng learner hoặc nhiệm vụ tìm target sai.
Các thử nghiệm O1/O2/noinline/PTX chỉ nằm trong outputs, không đưa các knob này vào source.

## Thay đổi được kiểm

85 `@wp.func` helper trong17 file nhận `WorldState`/`WorldParams` bằng
`wp.ref[...]` (140 argument annotations). Kernel signatures giữ by-value như Warp yêu cầu.
Các helper chỉ sửa phần tử array và đọc parameter; không gán lại struct/field.
AST trước/sau giống hệt sau khi bỏ argument annotations.
Tránh sao chép155 array descriptors của WorldState qua từng helper.

Simulator là forward-only. Sáu kernel kiểm tra vật lý trong bốn test file được
ghi rõ `enable_backward=False`; không bỏ assertion.
Review độc lập không phát hiện Critical/Important. Một Minor về certificate-bank
arrays trong masked-reset test đã được sửa: so sánh fallback_* toàn bộ, per-world
arrays mới so hàng world1.

Bốn test `XYZCUDATests` được thêm vào `tests/test_cuda.py`, nên cloud gate hiện có
tự chạy trước profile benchmark:
- Native-vs-CUDA XYZ camera, motion, belief và observation;
- high/low altitude confirmation;
- snapshot exact next draw cùng masked reset;
- finite-height RF LOS/link budget đối chiếu Native.

Các test này skip rõ ràng khi không có CUDA; chưa được gọi là GPU passed trước
khi đọc được log chạy thật.

## Bằng chứng CPU và cách chạy lại

Prototype đầy đủ:1146 passed,98 skipped,8 warnings,1556 subtests,260.41s.
Có một lần chọn sai tên test file trước đó (không chạy test), rồi đã dùng đúng tên.
Lần chạy root đầu gặp258 collection errors vì hai checkout chẩn đoán nằm dưới outputs.
Đã chuyển hai checkout do phiên này tạo sang `/tmp/uav-xyz-preflight-checkouts-20261031/`
và chạy lại bare pytest; không sửa test discovery toàn repo để che lỗi.
Final bare pytest trên source mới: **1146 passed,102 skipped,8 warnings,1557 subtests,264.89s**, exit0.
Bốn skip mới là test yêu cầu CUDA thật; GPU retry chưa được gửi tại commit này.

AOT tối thiểu (máy cần cài optional GPU dependencies, không cần driver/GPU):

```python
import warp as wp
import uav_network.communication.gpu.mac as mac
wp.init()
wp.compile_aot_module(mac, arch=75, module_dir="/tmp/uav-mac-aot", use_ptx=False)
```

Raw logs/receipts nằm ở `outputs/xyz-gpu-preflight-42c1dac-20261031/`.
Không commit key/token hay binary compiler cache.

## Bước tiếp theo

Chạy lại BASE benchmark trên source đã kiểm, kiểm cả15 CUDA tests (gồm bốn XYZ),
các gate scenario/mission, actual E/B/replay, updates và peak VRAM.
Nếu BASE đạt, chạy PQL tuần tự trên cùng account. Chỉ quyết định20k training
từ số đo thực và quota còn lại; không dùng trạng thái Kaggle complete thay cho kết quả test.
