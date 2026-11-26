# V73: cấu hình 4096 env, batch8192 và log chẩn đoán

Ngày làm việc: 26/11/2026 (Asia/Saigon). Đây là phần tiếp nối bản kiểm toán V72;
toàn bộ phân tích map, UAV, reward, DS/AoI, packet custody và nguồn nghiên cứu vẫn
ở `V72_AUDIT_AND_FIXES_VI.md`, `V72_RESEARCH_AND_GPU_PLAN_VI.md` và bảng 39 vấn đề
`V72_BUG_STATUS.json`. Không xóa những vấn đề chưa được kiểm chứng trong V72.

## Yêu cầu đang áp dụng

- Target dùng thẳng **4096 env, batch8192**, không bắt buộc chạy tăng dần env.
- Kiểm thử chức năng được dùng số env vừa phải, ghi cả requested/effective.
- Cấu hình mặc định learner vẫn H256×256, replay50000, sáu UAV, compact3,
  điều khiển XY hai chiều, RF0.1W cố định. Không sửa loss hoặc optimizer.
- Dùng **account2 sau khi đánh số lại**, theo chỉ dẫn mới nhất của người dùng.
  Account3 không còn là tài khoản chạy được yêu cầu cho bước này.

## Tài khoản

File đang hoạt động đã được chuẩn hóa tên liên tục account1..account5:

| Tên trước | Tên sau |
|---|---|
| account1 | account1 |
| account7 | account2 |
| account8 | account3 |
| account9 | account4 |
| account10 | account5 |

Giữ nguyên thứ tự, token, cờ enabled, cấu hình chọn tài khoản và các phần W&B/HF.
Bản sao riêng được tạo trước khi sửa. Token account2 mới hoạt động; API quota trả
GPU used00:28:25.52 / tổng30giờ, reserved0 ở lúc preflight. Không đưa token hoặc
file khóa vào mã nguồn, notebook, log và ZIP. Kết luận token account2/3 inactive
trước đó thuộc bộ file/số thứ tự cũ; không áp dụng cho account2 hiện tại.

## Chạy cái gì và kết luận đến đâu

`scripts/gpu/run_capacity_probe.py` có hai chế độ:

| Chế độ | Env/batch | Công việc | Không chứng minh |
|---|---|---|---|
| functional | mặc định4/16; ghi rõ effective | packet Warp thật, fixture CPU thật, replay, actor inference và hai update | hiệu năng/VRAM target |
| capacity | lấy đúng YAML4096/8192 | cấp phát/reset và packet workload4096 mạng; replay50000; inference4096; sample8192; hai update thật | full4096 GPU RL hoặc convergence |

Placement capacity: Warp radio trên cuda0; learner/replay trên cuda1. Hai bộ nhớ
GPU được ghi riêng; không cộng VRAM hai T4 rồi coi như một vùng liên tục. Bốn
transition CPU thật được lặp lại chỉ để lấp đầy replay và thử kích thước tensor.
Không dùng dữ liệu lặp này làm bằng chứng chất lượng chính sách.

Log `resources.jsonl` được flush/fsync trước bước nguy hiểm, có phase_begin,
phase_end hoặc phase_error, thời gian, RSS/peak RSS, RAM khả dụng, VRAM free/total,
Torch allocated/reserved và peaks. `progress.json` lưu bước gần nhất, `result.json`
lưu kết quả. Nếu hệ điều hành kill tiến trình thì result có thể còn RUNNING;
phải đọc phase cuối, exitcode và log cấp cha. Torch peak không đo toàn bộ Warp;
device_used là snapshot, không được gọi là peak toàn thiết bị.

## Kiểm chứng và kết quả

- Test RED trước khi thêm runner đã được lưu trong `outputs/qa_evidence_v73/`.
- Test chức năng mới:10pass trên CPU, gồm hai thuật toán và bước actor trì hoãn.
- Các log chạy CUDA và capacity sẽ được ghi vào cùng bộ bằng chứng khi job kết thúc.

## Nguồn chính thức cho cách triển khai

- [NVIDIA Warp1.18 PyTorch interoperability](https://nvidia.github.io/warp/v1.18/user_guide/interoperability/pytorch.html):
  Warp có allocator riêng và stream riêng; vì vậy không lấy Torch peak thay cho
  bộ nhớ toàn thiết bị. Không suy luận từ hỗ trợ tương tác rằng full env đã tích hợp.
- [Kaggle CLI metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md):
  notebook private, bật GPU và cấu hình machine_shape bằng API chính thức.
- Các nguồn Zhang/Xue, tác giả thuật toán và UavNetSim đã ghim commit được đối chiếu
  ở báo cáo V72; bước chẩn đoán này không đổi giả thuyết nghiên cứu hoặc reward.

## Các phần còn phải hoàn thiện để training đầy đủ

Warp hiện chỉ xử lý radio: còn cần motion/safety/energy, sensing/DS/fusion, reward,
observation, reset/termination/truncation trên GPU; nối rollout thật vào replay và
learner; kiểm chứng CPU–Warp ở hidden terminal/timing và các trường hợp ngẫu nhiên;
đánh giá nhiều seed và baseline. Capacity pass không đóng các mục đó. Thứ tự này
tránh tối ưu tốc độ trên một môi trường đang khác logic tham chiếu.
