# Năm notebook MATD3 BASE cho Kaggle UI

Cả5 file dùng35 mục tiêu,25 vật cản,6 UAV,bản đồ1km và tối đa1500 bước.
Seed mặc định44; dùng cùng seed để so sánh cấu hình.

| File/case | Actor+critic | Hidden cả hai | Updates/collection | Actor LayerNorm | Actor calls/collection |
|---|---|---|---:|---|---:|
|01|Shared|512→512|8|Có|4|
|02|Shared|512→512|16|Có|8|
|03|Independent|512→512|16|Có|8|
|04|Independent|512→512|16|Không|8|
|05|Shared|256→256|8|Có|4|

LayerNorm là cờ actor_layer_norm hiện có: chuẩn hóa trong encoder actor.
Critic không có LayerNorm trong cả5 trường hợp. Không đổi initialization
hay phương trình thuật toán. Các notebook là MATD3 adapter hành động lai UAV;
không phải tuyên bố tái tạo nguyên vẹn toàn bộ thí nghiệm của paper gốc.

## So sánh có đối chứng

- 01 với02: chỉ đổi update8→16.
- 02 với03: đổi cả actor và critic shared→independent.
- 03 với04: chỉ bật/tắt LayerNorm ở actor.
- 01 với05: chỉ đổi hidden512→256, cùng kịch bản35/25.

Shared có1 actor và1 twin-critic module; actor dùng234 chiều gồm228
observation+6 agent-ID. Independent có6 actor và6 twin-critic modules,
mỗi actor nhận228 chiều. Critic nhận896 chiều state+joint action ở cả5 bản.
Sharing ở đây là các UAV dùng chung tham số actor và chung tham số critic;
actor và critic vẫn là hai mạng khác nhau, không dùng chung trọng số với nhau.

Mỗi vòng collector tạo4096 joint transitions; sau warmup learner học8 hoặc16
lượt,batch8192. MATD3 policy_delay=2 nên mỗi policy học4 hoặc8 lượt actor.
Không hiểu U8/U16 là8/16 update trên từng transition riêng lẻ.
Nhánh continuous XYZ/RF và categorical next-hop giữ nguyên.

Các điều kiện chung:
- BASE,NOCHUNK,critic_chunk_size=0,vectorize_critics=false.
- FP32; AMP,torch.compile và fused optimizers tắt.
- Replay1M;75%CPU/25%GPU;warmup131072;20000 episode tổng trên4096env.
- MATD3 initialization=default,output_normalization=false,grad clip per_parameter0.5.
- Learning rate,gamma,tau,exploration/target noise và replay peer actions giữ nguyên.
- Reward,CBF,sensing,network,pin không đổi; tất cả UAV hết pin thì kết thúc episode.
- Deterministic=true; cùng5 validation layouts mỗi3.6864M transition.
- Training-only; lưu metrics W&B và validation-best policy lên HF; không visualize.
- Fresh run; không warm-start và không auto-continue.
- Tối đa10 giờ train; số transition đạt được có thể khác giữa cấu hình.

So các checkpoint ở cùng transition budget. Thời gian train và dung lượng
GPU phải được đo trên job thật; CPU kiểm tra không chứng minh đủ VRAM hoặc
cấu hình nào học tốt hơn. Hai T4 không cộng VRAM thành một GPU.
Actor dùng quan sát cục bộ; CBF là tầng riêng sử dụng hình học toàn cục,
theo giả định đã ghi trong DEC_POMDP_OBSERVATION_AUDIT_2026-11-07.md.

## Cách dùng từng file

1. Tạo một notebook Kaggle Private riêng,import file đúng case.
2. Bật Internet và accelerator T4x2.
3. Cấp quyền Secrets WANDB_API_KEY và HF_TOKEN.
4. Sửa NOTEBOOK_REF ở ô code đầu theo URL thật. Giữ TRAIN_SEED=44.
5. Chạy preflight rồi Save & Run All trong phiên mới.

Mỗi lần chạy chọn run ID mới; tên run chứa SH/IND,H256/H512,U8/U16,
ALN/NOLN và seed. ALN nghĩa actor LayerNorm. Tất cả thuộc cùng group W&B
uav-matd3-ablation-35t25o-architecture-updates-20261107.
Đổi TRAIN_SEED sẽ đổi config,tags và tên run theo seed đó.
Không sửa riêng nhãn U/H mà bỏ qua config; mỗi notebook đã khóa đúng case.

Source được nhúng sẵn,không cần clone GitHub hoặc Kaggle API key.
Không chạy lại riêng ô launch trong phiên vừa train; thư mục cũ có thể còn.
Notebook chưa được submit tự động. Không xuất thêm MASAC trong yêu cầu này.

## Tạo lại file từ source sạch đã commit

```bash
python scripts/build_matd3_ablation_ui_notebooks.py --seed 44
```

Lệnh tạo đúng5 notebook trong Downloads và outputs/kaggle, không ghi đè
các notebook independent20t15o/256 hoặc35t25o/512 đã giao.
Cả5 notebook dùng một source archive chung. Profile nền là
configs/comparison-independent-35t25o-h512-xyz-sensing.yaml;
SPEC ghi rõ toàn bộ override của từng case. W&B lưu resolved config thực tế.

Kết quả kiểm chứng/source commit/file hashes xem CURRENT_TASK.md và
docs/superpowers/evidence/2026-11-07-matd3-five-ui-*.
