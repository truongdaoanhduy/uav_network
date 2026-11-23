# Kiểm tra lại learned power và chọn checkpoint — 30/10/2026

Snapshot W&B: 2026-10-30T08:27:52.415043+07:00. Đọc132 run trong cửa sổ mở rộng24–29/10 để bắt run gốc; tải unsampled history của37 training run, ghép thành32 lineage. Phần chính chỉ lấy lineage có outcome trong27–29/10 giờ Việt Nam. Đợt này chỉ đọc W&B, không khởi chạy hoặc hủy production job.

## Kết luận theo dữ liệu

Nhận xét policy đã học được tìm kiếm và truyền/relay là đúng. Success đo giao đủ35/35 target, không phải packet success hoặc việc vòng episode có kết thúc. Episode timeout/battery vẫn là episode đã kết thúc; delivery cao và success0 có thể cùng tồn tại.
Thậm chí training có một số episode giao đủ target. Tuy nhiên tần suất thấp, có suy giảm theo thời gian, và không thể coi quality plateau là mission solved. Không quy toàn bộ hạn chế cho kịch bản hoặc gọi mọi learned-power policy là kém.

## Cùng episode, không lấy trung bình toàn run lệch budget

Bảng dưới dùng đúng episode8193–12288, mỗi hàng4096 outcome đầy đủ. Tất cả FP32,E4096,B8192,seed44,35T/25O/map1km/horizon1800/report1KiB. Có ghi source revisions và actual global-step budget trong JSON. Đây là performance training của policy thay đổi theo thời gian, không phải evaluation của một checkpoint đóng băng.

| Run | Thuật toán / sharing / replay | Search % | Delivery % | Success % | Return |
|---|---|---:|---:|---:|---:|
| [7317eeba](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/7317eeba) | MASAC BASE / both / HYBRID75 | 94.77 | 78.87 | 0.464 | 1229.94 |
| [af6502a8](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/af6502a8) | MASAC BASE / both / FULLGPU | 84.98 | 73.01 | 0.098 | 1190.74 |
| [81cd777d](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/81cd777d) | MASAC BASE / critic / HYBRID75 | 74.57 | 66.30 | 0.073 | 1010.90 |
| [92e50d5b](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/92e50d5b) | MASAC BASE / critic / FULLGPU | 73.54 | 64.48 | 0.049 | 967.85 |
| [3fcf25c7](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/3fcf25c7) | MATD3 PQL / independent / HYBRID75 | 81.76 | 64.37 | 0.000 | 1019.91 |
| [b8d0b40a](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/b8d0b40a) | MATD3 PQL / independent / HYBRID75 | 83.58 | 62.19 | 0.000 | 930.92 |
| [d62f1326](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/d62f1326) | MASAC BASE / independent / HYBRID75 | 81.37 | 50.66 | 0.000 | 232.48 |
| [a85d067a](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/a85d067a) | MATD3 BASE / independent / HYBRID75 | 83.33 | 49.58 | 0.000 | 483.01 |
| [bc3695bd](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/bc3695bd) | MASAC BASE / actor / HYBRID75 | 70.95 | 43.56 | 0.000 | 145.97 |
| [f1cd8533](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/f1cd8533) | MASAC BASE / actor / FULLGPU | 73.94 | 33.19 | 0.000 | -21.45 |
| [8a3028b9](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/8a3028b9) | MASAC BASE / independent / FULLGPU | 49.55 | 32.92 | 0.000 | -54.55 |
| [a6efa19d](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/a6efa19d) | MASAC PQL / independent / FULLGPU | 53.19 | 29.44 | 0.000 | -251.29 |

Ở mốc8192, MASAC shared-actor Hybrid đạt delivery84.22%, success1.855%; shared-both Hybrid83.09%,1.392%. Đến12288, shared-actor Hybrid giảm43.56% delivery, shared-both vẫn78.87%. Vì vậy thứ hạng đổi theo mốc, và shared-both không luôn thắng mọi cấu hình ở mọi thời điểm.
MATD3 PQL FullGPU76b73021 là tham chiếu bổ sung tốt: window15905–20000 search95.38%, delivery83.99%, success1.147%. Nhưng history run đó kết thúc22:38 ngày26/10 giờ Việt Nam; không gắn nhãn nó là run27–29/10. Nó được giữ riêng trong dữ liệu mở rộng. Các run MATD3 mới hơn và FP16 cũng được đọc, không bị loại khỏi inventory vì score thấp; FP16/E2048 không trộn vào bảng FP32/E4096.
Trong các sharing training run ở khoảng được xét, các kết quả shared actor/critic là MASAC; không suy diễn rằng MATD3 shared-both đã được production-train chỉ vì profile đã tồn tại.

Global-step endpoint có thể khác dù cùng episode. W&B không có đầy đủ mọi episode đầu ở một số run; chỉ cửa sổ đủ4096 record được xếp vào bảng. Continuation quay lại checkpoint cũ có đoạn lịch sử trùng nhưng trajectory khác: bỏ nhánh cũ bị thay thế từ episode đầu của child quan sát được, ghi rõ rewind; không đếm hai lần hoặc coi child là training seed độc lập. Không dùng tail vài chục episode để chọn winner.

## Bằng chứng relay

Một cohort4096 outcome gần12k của shared-both Hybrid ghi238795 custody transfers qua peer và73979 report tới GCS qua relay, bên cạnh39073 report direct. FullGPU có227180 peer transfers,74378 report relayed và30288 direct ở một cohort4096 gần12k. Đây là telemetry tổng nhiều episode, không phải số target của một scene, và cohort theo transition có thể lệch nhẹ membership với cửa sổ episode cố định của bảng.
Code kiểm tra receiver active, report custody/buffer, range500m và nominal required power<=Pmax trước khi admission. Trong các link hợp lệ, action chọn power0.001–0.4W theo mapping log, receiver hoặc im lặng. LOS/NLOS/pathloss, SINR và interference/CSMA/retry/ACK vẫn có thể gây thất bại. Ngoài range/Pmax thì GCS bị mask; policy có thể relay, di chuyển hoặc im lặng. Không có forced relay tự động, công suất vô hạn hoặc bảo đảm gửi thành công.

## Đánh giá lại checkpoint thật

Đã hoàn tất 7/7 candidate trên cùng5 validation seeds1000000044..1000000048, deterministic CPU, dùng config lưu trong từng checkpoint. Candidate được chọn theo các mốc training và artifact sẵn có; đây là phân tích hậu nghiệm, không phải tìm kiếm exhaustive. Đây là post-hoc comparison nhỏ, không phải nhiều training seed hay tìm kiếm exhaustive mọi artifact.

| Candidate | Episode | Search % | Delivery % | Success | Return |
|---|---:|---:|---:|---:|---:|
| hybrid-ep12288 | 12288 | 91.43 | 80.57 | 0.0% | 1425.90 |
| fullgpu-ep12288 | 12288 | 77.14 | 60.57 | 0.0% | 881.80 |
| hybrid-final12353 | 12353 | 88.00 | 72.00 | 0.0% | 986.91 |
| fullgpu-final16384 | 16384 | 67.43 | 56.57 | 0.0% | 1001.21 |
| hybrid-actor-ep8192 | 8192 | 68.00 | 32.00 | 0.0% | -864.59 |
| hybrid-sharedboth-ep8192 | 8192 | 98.29 | 81.14 | 0.0% | 1228.62 |
| matd3-pql-hybrid-ep12421 | 12421 | 71.43 | 49.71 | 0.0% | 634.32 |

Riêng shared-both Hybrid: checkpoint12288 delivery80.57% tốt hơn cuối12353 delivery72.00%. FullGPU:12288 đạt60.57%, cuối16384 đạt56.57%. Đây là bằng chứng cụ thể cần chọn checkpoint bằng validation. So cuối12353 với cuối16384 không phải so cùng training budget; chỉ dùng cặp đó để đánh giá suy giảm trong từng lineage.
Candidate MATD3 PQL12421 (nếu có trong bảng) gần12288 hơn133 episode, nhưng không phải đúng cùng mốc; giữ episode/global_step chính xác trong JSON. Mốc8192 của shared-actor/shared-both được so riêng. Tất cả dựa trên5 layout, không suy luận configuration tốt nhất phổ quát hoặc tốc độ hội tụ.
Evaluation lịch sử30 episode của MATD3 PQL Hybrid b8d0b40a→63ad8bd6 có search82.10%,delivery54.29%,success0; MATD3 BASE a85d067a→dbd9e9fa42.95%,35.81%,0. Không trộn các số test bank2000000044 này vào phép chọn validation mới hoặc bảng training.

## Thay đổi code

- Ba profile20T/15O/1200 giữ sharing/replay/E/B/FP32 và trả power_control về learned. Link-budget+3dB là ablation riêng tênlink_budget; historical YAML/manifests không bị viết lại.
- Best chọn success→delivery→search→return trên periodic validation. All-zero success vẫn phân biệt delivery/search. Bằng điểm giữ candidate sớm hơn; không lấy peak training reward hay test bank để chọn.
- Copy đúng immutable actor snapshot đã được evaluate sangbest.pt; best.json có source step/episode/version,hash,score và seed bank. Không dùng weights mới ở thời điểm child trả kết quả.
- Train result checkpoint, final model artifact và CPU evaluation/visualize chọnbest; aliasbest chỉ khi validated. Full resume giữ last learner/optimizer/replay/RNG/environment và kèm winner để continuation ở thư mục mới không mất best cũ.
- Không có validation hợp lệ thì ghiunvalidated_final_fallback, không giả vờ last làbest. best.pt làactor-only; manual BASE warm-start từ đó làlineage mới với critics/optimizer/replay/counters mới, được ghi rõ; exact continuation vẫn dùngresume.pt.
- Study yêu cầu train đủ budget nhưng đối chiếu evaluation với step của selected policy, có thể sớm hơn cuối training. Khôi phục best cũng kiểm validation/test disjoint, score bounds và score/metrics consistency; sidecar thiếu/hỏng được repair từ committed state.

## Giới hạn và bằng chứng kiểm tra

Không launch/cancel production job, không đổi35/25 checkpoint sang20/15, không đổi sensing/XYZ. Máy hiện tại không cóCUDA; test CPU không chứng minh runtime/performance GPU. Không có live model publication mới lênW&B để giả lập production; boundary upload best vs full resume được kiểm bằng tracker thật và SDK double.
Independent review:3 Important đã có reproductions RED→GREEN; sidecar metadata repair được nâng mứcImportant do ảnh hưởng provenance. Không còn item bị hoãn theo đánh giá của tác giả; toàn bộ suite phải xanh trước bàn giao. Kết quả cuối đọc tại evidence2026-10-30-learned-power-best.json và CURRENT_TASK.md.

Nguồn:W&B Public API run configs/summaries/unsampled histories; logged artifact metadata; checkpoint downloads xác minhSHA256; fresh CPU evaluation metrics; source code tại nhánhfeat/cpu-replay-20261022. Raw input giữ tạioutputs/learned-power-best-20261030/ trên máy dự án; không chứa API credential values.
