# V7.0 — Gate CUDA / Kaggle account 2, không nhúng credential

**Chưa dùng tài khoản Kaggle account 2 trong lượt kiểm định này.** ZIP tải lên tồn tại trong ChatGPT container, nhưng container này chỉ có PyTorch CPU và không có `warp-lang`, `nvidia-smi` hoặc driver CUDA. Sandbox Ubuntu được kết nối qua MCP cũng không có torch/warp/nvidia-smi tại thời điểm kiểm tra. Không đọc, sao chép, in hoặc đưa `key.txt` vào ZIP, vì key là bí mật tài khoản. Người chạy Kaggle tự chọn tài khoản/accelerator T4×2 trên Kaggle.

## Gate đúng thứ tự

1. **CPU source gate**: `bash setup.sh --qa --seed 44` → tất cả regression và 1km MATD3/MASAC 4-transition phải PASS, QA JSON phải ghi GPU BLOCKED rõ ràng.
2. **Kaggle setup**: upload ZIP vào notebook của account 2 (qua Kaggle Dataset/Add Input); bật accelerator Tesla T4×2, kiểm tra `nvidia-smi`, `import torch; print(torch.cuda.is_available())`, cài `warp-lang==1.18.0` sau khi chuẩn bị môi trường. Không upload file key/private API credentials.
3. **Packet-only CUDA smoke (không phải training)**: `python scripts/gpu/warp_packet_cuda_gate.py --envs 4 --uavs 6 --grid 20 --device cuda:0` rồi lặp `--envs 32/256/1024/2048` (giữ `--uavs 6 --grid 20` để đánh giá đúng cấu hình chính). Nếu có lỗi compile, traceback được giữ nguyên; *không* tăng batch khi có FAIL.
4. **Kiểm tra device 2**: `--device cuda:1 --envs 32` riêng biệt. T4×2 không hợp nhất thành một bộ nhớ 32GiB; mỗi CUDA device cần budget riêng.
5. **Memory benchmark**: dùng `torch.cuda.mem_get_info()` trước/sau và NVML nvidia-smi; không chỉ `torch.cuda.max_memory_allocated()` vì Warp cấp phát độc lập. Tính headroom, kernel scratch, torch activations và replay trước khi đưa ra kết luận fits.
6. **CPU ↔ Warp packet-event trace**: cùng seed/topology/PHY/time-step, so HELLO, route, CCA, TX start/end, SINR, MAC ACK, E2E ACK, dropped/dedup, queue, latency, AoI; hiện **chưa có gate parity hoàn chỉnh**. `NumPySoARefV67` *không đủ* làm chuẩn event.
7. **Full env GPU port**: hành động actor -> network -> mobility -> belief -> reward -> done/reset -> replay -> MATD3/MASAC backward -> eval. Sau khi 32 env đúng mới đo 256/2048/4096; ưu tiên 2048 env trên *mỗi* T4 để đạt tổng 4096 env, không giả định một GPU chứa tất cả.
8. **Nghiên cứu học thuật**: train multi-seed 44/45/46, eval seed khác train; so matched `datamule` vs `no_peer` tại từng map size, có plot mean±CI. Không được báo RL gains từ smoke 4 transitions.

**Script CUDA hiện chỉ kiểm chứng radio subsystem:** `reset → PHY → HELLO/DV → sense → inject → packet step → snapshot` và đo memory delta. Nó không gọi mô hình MATD3/MASAC, không chấm reward map và không chạy parallel RL training. Trạng thái hợp lệ trước full parity là `BLOCKED`.

**Khi dùng 3km:** `--grid 60` có 3600 cell và ~6.08 GiB network SoA cho 2048 env/GPU; 4096 env/GPU ~12.16 GiB chưa cộng replay/optimizer. Xem `outputs/qa_evidence/memory_estimates.json` được tạo từ schema (không phải đo CUDA).
