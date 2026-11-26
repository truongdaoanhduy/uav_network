# Kiểm định độc lập V81 và bản sửa V82 — 26/11/2026

## Kết luận có thể kiểm chứng

V81 sửa đúng lỗi bất đối xứng live/dead UAV và bổ sung được các lựa chọn ổn định số. Tuy nhiên, kết luận “đã tìm đủ nguyên nhân gốc và đã ổn định MATD3” chưa được dữ liệu chứng minh. Chạy lại gói đính kèm nguyên bản trên CUDA cho MATD3 E64, H64, B128, seed44, 300 bước có 66 safety terminations; MASAC có 5. Cấu hình này khác V80 về warmup, replay, kiến trúc, loss và khởi tạo nên đây là kiểm tra phản ví dụ cho lời hứa cải thiện, không phải thí nghiệm cô lập nguyên nhân.

Các trường boundary_collision / uav_pair_collision là cờ phát hiện quỹ đạo ứng viên SAU shield bị vi phạm. Fail-closed giữ nguyên pos/vel của world lỗi. Vì vậy không nên gọi mọi cờ này là UAV đã thật sự đâm xuyên tường trong trạng thái được commit.

Đã thực hiện sửa lỗi có test RED/GREEN; giữ cấu hình ổn định V81 làm thí nghiệm opt-in; không bật nó thành một baseline đã được chứng minh tốt hơn. Không đưa CBF trở lại code.

Phiên bản, SHA, cấu hình, lệnh, thiết bị, số update và bảng chạy đầy đủ nằm trong 02_RUN_RESULTS.csv, RESULTS.json và SOURCE_PROVENANCE.json. Thư mục source_repository là toàn bộ repository tại commit phát hành; project hiện hành vẫn nằm tại experiments/v80_vast_20261126. Tên đường dẫn này được giữ để bảo toàn lịch sử Git.

## 1. Các thành phần THỰC SỰ đang chạy

| Thành phần | Hiện trạng đọc từ source | Xung đột / giới hạn cần công bố |
|---|---|---|
| Không gian | 1000×1000 m; 6 UAV và một GCS; 15 target tĩnh; grid20×20, ô50 m; tòa nhà hình trụ và blackout | Fixed-altitude 2.5D, XY tại50 m. GPU hiện chỉ nhận 6 UAV, grid20, map1000. Các comment YAML cho phép đổi tự do là hướng dẫn chung của CPU, không phải khả năng của kernel GPU. |
| Động học | dt1 s, vmax15 m/s, amax5 m/s², separation50 m | Clamp gia tốc trước rồi vận tốc; không nới ràng buộc để làm đẹp kết quả. Radio dùng vị trí cố định trong từng tick môi trường, chưa phải kênh liên tục theo quỹ đạo. |
| Safety | 66 ứng viên; ba sweep; kiểm swept physical; fail-closed pos/vel | Đây là tìm kiếm rời rạc có thể không tìm được ứng viên khả thi. Việc dừng episode an toàn không đồng nghĩa luôn tiếp tục được nhiệm vụ. UAV hết pin là vật cản đứng yên, một giả thiết mô phỏng cần ghi rõ. |
| Sensor | Pd=.85, Pf=.03; footprint50 m; occlusion3D; m_f=m_e=.8, threshold.9, uncertainty≤.2 | Đây là baseline sensor/DS cố định, không phải phương án Bayes-calibrated của dự án cũ. Không có ánh xạ duy nhất từ Pd/Pf thành ba mass nếu chưa nêu calibration/prior/discount. |
| Belief | Origin ledger, UQ8.8, generation timestamp, eta=.02; sum các origin đã nhận | Khác phép max-count ở Eq.16 của Zhang. Origin riêng chống gộp trùng cùng evidence, nhưng còn cần kiểm tra toàn bộ packet lifecycle. Aging effective count là một mở rộng, khác discount trực tiếp mass DS. |
| Actor | Raw514=14 local+5×100 pooled map; encoded518 thêm distance và ba mask | Không thấy rõ năm peer pose/velocity và NFZ geometry. Chỉ thêm HELLO đã nhận, age, validity hoặc sensor local; không lấy trực tiếp ground-truth peer state để gọi là Dec-POMDP. |
| Critic | CTDE từ joint encoded observations, state_dim=6×518=3108 trong runner | Đây là thông tin gộp của các Actor; không tự động là toàn bộ trạng thái Markov của radio/mission. Truth target chỉ dùng mô phỏng/reward/đánh giá, không đi vào Actor. |
| Mạng | WarpCompressedRadioNetworkV67, event loop CSMA/CA, HELLO, MAC ACK, E2E ACK, TTL, custody, retry, data mule | Là một discrete-event MAC trong mỗi macro tick, không chính xác nếu chỉ gọi là “MAC slotted surrogate”. Chưa chứng minh tương đương upstream SimPy theo event. |
| RF | TX.1 W; noise4.389577e-13 W; CCA-82 dBm; 6dB; 12dB/tòa nhà | Nhiễu thấp khớp công thức kTB+NF7dB, BW22MHz của upstream hiện hành; 4e-11 W là cấu hình Table I bài 2025. 12dB là heuristic; ITU P.2109 không xác nhận hằng số xuyên mọi hình trụ. |
| Reward | Local uncertainty-minimum credit; GCS confirmation credit; AoI delay bonus; FP, safety, intervention, airtime costs | w_loc=.5 không tự động oracle leak. Nó thay mục tiêu so với GCS-only. lambda_b=0 và w_p2p=0 là lựa chọn objective, không phải crash bug. |
| Learner | Independent MATD3/MASAC mặc định; Torch replay; world Warp | Chạy được CUDA end-to-end không đủ chứng minh học tốt, parity toàn mission, benchmark T4 hay 2-GPU. |
| V81 recipe | Actor LN; critic LN; SiLU; Huber10; MATD3 pre-tanh reg1e-3; MATD3 orthogonal init | Không đối xứng hoàn toàn: MASAC vẫn default initialization, không có cùng pre-tanh penalty. MASAC entropy là khác biệt thuật toán. |

Chi phí reward thường được lưu bằng số dương rồi bị trừ trong kernel: r_fp=12.5, fp_hold=.1, safety_failure_penalty=100. Không đổi chúng thành số âm chỉ vì diễn giải reward là -12.5/-100; phải đọc dấu ở nơi tính reward.

## 2. Kiểm định toàn bộ 24 mục V80

Bảng 03_ISSUE_MATRIX_V80_V81_V82.csv giữ nguyên các cột proof/source/action của V80 và bổ sung verdict, bằng chứng mới, cách xử lý. Không gộp các mục mở thành bug đã sửa.

| ID | Kết luận sau đối chiếu code/log | Trạng thái và hành động |
|---|---|---|
| 01 | int64 Torch→Warp int32 là lỗi ABI thật | Giữ bản sửa; full CUDA regression. |
| 02 | Backward codegen/unroll từng gây compile lâu | Giữ enable_backward=False/max_unroll2 trong workload forward-only; log cold/warm không được trộn. Không suy ra cùng tốc độ trên mọi máy. |
| 03 | Sai sensor launch ABI là lỗi thật | Giữ đủ14 đối số, có failed; regression CUDA. |
| 04 | Sai key termination_reason là lỗi thật | Giữ đúng key, kiểm bốn bit kết thúc. |
| 05 | Clamp sai thứ tự gây lệch CPU–Warp | Giữ gia tốc trước, vận tốc sau; gate CPU–CUDA4×16. |
| 06 | dt khác1 làm lệch các clock được hardcode | Guard dt1 là đúng; muốn dt khác phải sửa đồng bộ physics/radio/sensor. |
| 07 | Overflow kiểm sau reset có thể bị xóa | Giữ mỗi-step guard trước reset. Test injection và regression; không bỏ guard để tăng throughput. |
| 08 | Ghi pos/vel sau failure là lỗi thật đã sửa | Giữ fail-closed. Phạm vi bảo toàn pos/vel, không phải rollback toàn bộ radio đã chạy trước đó. |
| 09 | Thiếu bảo đảm formal là đúng; thêm lỗi dead-pair thật | V81 sửa dead-pair đúng; test V80 RED và V81/V82 GREEN hai thứ tự trên CPU/CUDA. Bảo đảm viability vẫn OPEN. |
| 10 | Tính 71.59−50=21.59 m là đúng nhưng chưa chứng minh spawn sai | Spawn có velocity0. Với đối đầu hai UAV15m/s cần xem relative stopping45m và tick rời rạc; không lấy riêng22.5m làm proof bắt buộc nới spawn. |
| 11 | Thiếu explicit peer/NFZ observation là có thật | DESIGN GAP, chưa phải lỗi tensor. Thêm HELLO age/validity cần observation schema và checkpoint version mới; chưa tự thêm truth. |
| 12 | Fixed mass/aging là lựa chọn mô hình; eta0 bị guard | Không bypass UQ8.8. Cần thay representation và wire/ACK nếu muốn900 phép đo không decay; calibration mass cần ablation riêng. |
| 13 | Nghi ngờ công thức chưa đủ để kết luận sai | Sweep400 ô CPU/CUDA PASS với cùng encoded ledger. Khác reducer max/sum và aging vẫn phải công bố; chưa gọi là toàn bộ paper parity. |
| 14 | Local credit có thể đổi objective | Giữ baseline; ablate w_loc0 trước. Reward hiện tại không tự thỏa potential shaping theorem. |
| 15 | Không phạt airtime khi lambda_b0 | CONFIG/ABLATION. Thử hệ số khác với cùng seeds và offered traffic; không mặc định0 là bug. |
| 16 | Chênh19.596dB so với Table I là đúng | PAPER CONFIG DIFFERENCE; upstream hiện tại có đúng công thức noise thấp. Không thay mặc định thành4e-11 như một bugfix. |
| 17 | 12dB/cylinder chưa calibration là đúng | MODEL LIMIT; cần sweep và đo/hiệu chuẩn. NFZ pháp lý không tự hấp thụ RF. |
| 18 | Chưa có upstream packet-event parity là đúng | OPEN. Golden model hiện có là reference của custom network; cần cùng packet IDs, traffic, seed/event ordering, ACK, retry, queue, TTL với upstream pinned commit. |
| 19 | Có khác scale raw gradient, nhưng chưa là bug optimizer | Giữ scale sync opt-in. Test Adam cho gradient×3 mà update bước đầu giống hệt; không được nói tốc độ học×3 từ hệ số loss. |
| 20 | Chưa convergence/held-out mission evaluation | OPEN. 900 vector steps là integration stress, không phải train20k episodes. |
| 21 | Chưa đo 2GPU | OPEN. GPU0 đang dùng~11.7GiB; không động vào process khác. cuda:0 khi CUDA_VISIBLE_DEVICES=1 thực ra là physicalGPU1. |
| 22 | Trước đây chỉ zero-action chứng minh horizon | Đã chạy policy900 bước với autoreset và update thật. Safety terminations vẫn có; chưa chứng minh mỗi world sống900 bước hoặc nhiệm vụ thành công. |
| 23 | Ablation bị từ chối là đúng, nhưng validator đang bảo vệ semantics | Highnoise V81 hỗ trợ all_depleted; eta0 và w_p2p chưa support vẫn reject có chủ đích. Không đổi battery mode để giả cùng một ablation. |
| 24 | D2H metrics và JSON mỗi step có overhead thật | V82 thêm log-interval và bỏ lưu toàn bộ statuses trong RAM; test interval1/2 có cùng update/final_record. D2H scalar metrics vẫn còn; chưa tuyên bố đã tối ưu throughput. |

## 3. Những suy luận V81 cần sửa

1. ReLU zero-activation fraction không phải tỷ lệ neuron dormant đo trên một tập states, càng không tự chứng minh neuron “chết” vĩnh viễn. Sokar dùng hoạt tính trung bình chuẩn hóa từng neuron. SiLU làm mất phần lớn zero outputs theo định nghĩa nhưng không bảo đảm plasticity hay tránh collision. LayerNorm/SiLU là lựa chọn ablation có cơ sở nghiên cứu.
2. Bão hòa tanh và gradient nhỏ được log hỗ trợ. Quan hệ nhân quả “vì thế chết38 lần” chưa được tách khỏi replay, exploration, shield và observation. SAC entropy thường khuyến khích phân phối có entropy; không phải hàng rào cứng bảo đảm mean/pre-tanh không bão hòa.
3. Replay512/warmup thiếu là đúng ở bài E64/B128 V80. Đây là runner smoke/stress, không được suy ra mọi run production cũng dùng512. V81 cập nhật từ bước64, tức4096 joint transitions; 237 updates trong300 bước, thay cho299 ở V80.
4. Loss scale: trong independent6 critics, MATD3 MSE1+MSE2 lấy mean trên agents, MASAC .5×sum×6. Có hệ số3 ở raw gradients trong cùng tensor regression. Adam, epsilon, clipping và TD targets khác nhau khiến “học chậm3×” không được suy ra.
5. Huber10 không chặn giá trị TD/Q vô hạn: loss tăng tuyến tính ở ngoài delta; gradient theo residual được chặn, còn gradient theo weights phụ thuộc Jacobian. Với residual nhỏ, Huber=.5 e² trong khi MSE=e²; bật Huber đồng thời đổi cả scale.
6. Pre-tanh² continuous-only là một adapter thử nghiệm, không giống nguyên xi regularization mọi policy output/logit của MADDPG. Hệ số1e-3 không phải một nghiệm đã tối ưu cho mission này.
7. Cấu hình V81 đổi đồng thời ít nhất LN, activation, init, loss, scale, regularization, replay và warmup. Không thể gán kết quả cho riêng một yếu tố.
8. Gói V81 nói “4 nguyên nhân” nhưng danh sách chi tiết chỉ có ba nhóm; nếu tính scale là nhóm thứ tư thì nó vẫn là giả thuyết cần thí nghiệm.

## 4. Sửa gì trong V82 và bằng chứng

| Tệp | Sửa | Vì sao / kiểm thử |
|---|---|---|
| mission_v80_shield.py | Giữ bản sửa live/dead của V81 | V80 bỏ cặp khi dead có index nhỏ; test bad state khoảng cách25m. V80 RED, V82 fail-closed cả hai thứ tự; suite CUDA kiểm tra thật. |
| schema.py | Kiểm bool của critic_layer_norm/critic_batch_norm/sync_critic_loss_scale; cấm LN+BN critic cùng bật | Checkpoint dict “false” trước đây bị coi truthy; mlp ưu tiên BN làm LN bị lờ đi. Sáu case config RED→GREEN. YAML loader vốn đã kiểm kiểu, lỗi này ở direct config/checkpoint path. |
| rollout_v82_contracts.py | Kiểm số dương, replay≥env/batch/warmup, budget đủ update trừ explicit smoke | Tái hiện V81 replay512/warmup10000 vẫn báo STABLE PASS với0 updates. V82 reject trước cấp phát world. |
| run_v81_real_rollout.py | Tên file được giữ; kết quả versionV82; cấu hình đầy đủ, seed, Git dirty/SHA, input hashes; hidden chỉ override khi truyền CLI | Không ghi đè ngầm hidden_sizes YAML bằng256. Config effective là nguồn đối chiếu run. |
| run_v81_real_rollout.py | Tách integration_pass, safety_gate_pass, learning_updates_executed và scientific_status | Run hoàn tất không được đồng nghĩa ổn định/an toàn/hội tụ. Không dùng status “STABLE ... PASS” nữa. |
| run_v81_real_rollout.py | Chặn output/progress tồn tại; đóng handle; --log-interval; chỉ giữ final_record | Tránh trộn log nhiều lượt; giảm ghi JSON và RAM host. Guard overflow mỗi bước giữ nguyên. |
| rollout_v82_contracts.py | Snapshot current GCS latch, FP, coverage, battery | Dữ liệu đánh giá tách Actor. Snapshot là world hiện tại sau autoreset, không phải kết quả mean evaluation complete episodes. |
| test_v82_behavioral_regressions.py | Behavior tests config/runner, dead-pair, gradient penalty, LN broadcast, real updates shared/independent, checkpoint roundtrip, interval parity | Kiểm hành vi thật thay vì chỉ tìm chuỗi AST. Test penalty có gradient ở raw10, không chạm destination weights. |
| Các lựa chọn V81 | Giữ LayerNorm/SiLU/Huber/sync/pre-tanh opt-in; default V80 được migrate | V80 checkpoint MATD3/MASAC load bằng V82 và action deterministic giống bitwise. Không khẳng định mọi checkpoint mọi phiên bản đều tương thích. |

Không thay safe_d50, vmax15, amax5, dt1; không tắt failure termination/overflow; không đưa target truth vào Actor; không tự bật learned TX, peer reward, eta0 hay đổi map.

## 5. Kết quả chạy

| Run | E / steps | Updates | Safety | Boundary | Pair | Horizon | Full15 success |
|---|---:|---:|---:|---:|---:|---:|---:|
| 38_FINAL_v80_matd3_e64_h900_s300 | 64/300 | 299 | 38 | 35 | 6 | 0 | 0 |
| 39_FINAL_v80_masac_e64_h900_s300 | 64/300 | 299 | 3 | 1 | 2 | 0 | 0 |
| 03_v81_matd3_e64_s300 | 64/300 | 237 | 66 | 49 | 21 | 0 | 0 |
| 04_v81_masac_e64_s300 | 64/300 | 237 | 5 | 5 | 0 | 0 | 0 |
| 12_v82_matd3_e4096_s3_h900 | 4096/3 | 2 | 0 | 0 | 0 | 0 | 0 |
| 14_v82_masac_e4096_s3_h900 | 4096/3 | 2 | 0 | 0 | 0 | 0 | 0 |
| 15_matd3_warmup_only_s300 | 64/300 | 237 | 242 | 187 | 114 | 0 | 0 |
| 16_masac_warmup_only_s300 | 64/300 | 237 | 7 | 7 | 1 | 0 | 0 |
| 17_v82_matd3_stability_s900 | 64/900 | 837 | 210 | 151 | 69 | 45 | 0 |
| 18_v82_masac_stability_s900 | 64/900 | 837 | 5 | 5 | 0 | 59 | 0 |
| 23_FINAL_v82_masac_e4096_s3_h900 | 4096/3 | 2 | 0 | 0 | 0 | 0 | 0 |
| 23_FINAL_v82_matd3_e4096_s3_h900 | 4096/3 | 2 | 0 | 0 | 0 | 0 | 0 |
| 26_FINAL_matd3_warmup_only_s300 | 64/300 | 237 | 242 | 187 | 114 | 0 | 0 |

Đơn vị safety là world-episode termination, không phải số UAV. Cờ boundary/pair có thể cùng xuất hiện trong cùng episode. Các tổng này không được cộng thành “số lần chết”. Các run300/900 khác exposure nên không so raw counts trực tiếp giữa hai budget.

- V81 original: nguồn đúng ZIP đính kèm, H64 để so với log stress V80 H64. Lệnh H256 mặc định được nêu trong prompt không được giả là đã chạy.
- Warmup-only: giữ physics V81 dead-pair fix, kiến trúc/default loss V80, chỉ dùng replay65536 và learning_starts4096. Đây là đối chứng bundle replay+warmup, chưa tách riêng hai yếu tố.
- Runs900: thực hiện900 vector steps, có thể autoreset nhiều episode. Hoàn tất mission đòi all15 targets confirmed ở GCS; xem bit4, không lấy horizon truncation làm success.
- E4096/B8192: horizon900 nhưng chỉ3 steps, hai update; đây là smoke bộ nhớ và nối world–learner. Không phải E4096×900 training.
- Không đủ nhiều seed để báo confidence interval hoặc kết luận thuật toán nào tốt hơn.
- Torch peak VRAM không tính toàn bộ allocations riêng của Warp. Replay65536 chiếm3,265,789,952 bytes; phải cộng world/radio và activation/optimizer trước khi scale.

Lượt full-suite ban đầu có3 failure: hai manifest stale trong lúc chỉnh code và một CPU test vô tình khởi tạo CUDA context trên GPU0 gần đầy dưới Torch2.14. Các log FAIL được giữ. Sau cập nhật manifest và cô lập GPU1, full suite PASS; không sửa mất test hoặc kill process để làm xanh.

## 6. Kaggle lịch sử và phạm vi bằng chứng

history_archives giữ nguyên các ZIP lịch sử đã tìm được, kèm SHA256; kaggle_history có notebook và raw logs trích từ master cũ. 04_KAGGLE_HISTORY.csv ghi từng job và raw status.

- V77 account2: packet/radio, NFZ, capacity tests; overall DIAGNOSTICS_ONLY_NOT_FULL_RL. Không dùng nó làm convergence.
- V78 account2: packet gates, hidden-terminal64seeds, mission smoke, MATD3/MASAC independent E4096/B8192. Status công bố là experimental, không phải upstream parity.
- V79 CPU và Vast V80 được để ở thư mục riêng. Không đổi nhãn các log trung gian41/3 thành kết quả final38/3.
- Không gửi thêm job Kaggle trong lượt này: Vast đang hoạt động, đúng ưu tiên bàn giao. Không đọc/in key.txt hay đính kèm credential. Không khẳng định có đầy đủ mọi lượt Kaggle từ mọi phiên trước; chỉ các archive/log thực sự hiện hữu được lập chỉ mục.

## 7. Nên làm tiếp theo — theo gate

1. Map/physical safety: xây tập state khả thi/adversarial sát biên, head-on, live/dead, nhiều UAV; đo candidate rejection, jerk/acceleration, lỗi commit, termination theo cùng seed. Tách formal viability khỏi việc dừng fail-closed. Nếu sửa shield, chứng minh RED/GREEN trước và giữ ràng buộc vật lý. Không cần tự đưa CBF trở lại.
2. Observation: thiết kế HELLO có vị trí/vận tốc, timestamp/TTL, validity; hoặc sensor local NFZ distance. Version hóa obs/checkpoint, kiểm mất gói/stale; làm ablation với observation cũ.
3. Belief: giữ reference closed-form đã PASS; nếu cần static eta0, thiết kế uint32/floating log-evidence cùng payload/version/ACK rồi parity CPU/CUDA/network. So sánh count-aging với mass discount như hai mô hình riêng. Bayes mass calibration phải có derivation và sensor calibration.
4. Network: pin upstream commit; làm traffic fixture DATA/ACK/retry/hidden terminal/TTL/out-of-order/energy, đối chiếu event traces trước rồi distribution PDR/delay/AoI. Hai định nghĩa offered traffic khác nhau không tạo một PDR so sánh được.
5. Reward: GCS-only vs local credit; airtime0 vs một hệ số đã chuẩn hóa; giữ cùng seed, transition/update budget. Nếu chọn potential shaping, dùng gamma*Phi(s_next)-Phi(s) và điều kiện terminal phù hợp; không gọi reward uncertainty giảm hiện tại là có bảo đảm policy invariance.
6. Learning: chạy ma trận từng yếu tố ở equal transition budget và nêu update budget; held-out seeds, save/resume cả optimizer/RNG/replay chứ không chỉ Actor; đánh giá discovery/delivery/coverage/AUC/capped delay. Chưa mở full E4096×900/20k khi safety/scientific gates còn đỏ.
7. Performance: profile event-kernel, Torch update, D2H và JSON riêng; dùng stream interop đúng trước graph capture. log-interval chỉ giải quyết một phần, chưa có benchmark tăng tốc được kiểm soát.
8. Multi-GPU/T4: GPU0 chưa trống; không có phép đo2GPU hoặc KaggleT4 V82. Chạy hai seeds độc lập trước khi thử phân vai world–learner qua PCIe, rồi đo chi phí truyền và memory thực.

## 8. Nguồn và giới hạn

05_SOURCES_AND_TOOL_STATUS.md ghi URL, pin và phạm vi đọc. Zhang full PDF được đọc từ file người dùng đã có; kiểm Eq.6–17 và mô hình thông tin. Không tái phân phối toàn văn paper trả phí trong ZIP.

ICML và AAAI được kiểm ở ICORE2026 A*. T-IV có SJR2025 Q1 trong Automotive Engineering. Không gọi ICCC, tài liệu hãng, tiêu chuẩn ITU là A*/Q1. Venue quality không chứng minh một patch đúng.

Consensus hết hạn mức; Liner từ chối vì billing. Chúng không cung cấp bằng chứng nghiên cứu cho kết luận. Đã dùng các nguồn gốc còn truy cập được, không tự mua/nạp phí. Browser không cần mở phiên đăng nhập cho các thông tin công khai này.
