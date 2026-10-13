# Kiểm tra các run đã hoàn tất và sửa lỗi nhiệm vụ

Ngày 13/10/2026. Repository đang dùng: `/home/aduy/Documents/NCKH/uav_network`.
Project cũ chỉ là tham khảo. Dữ liệu lấy trực tiếp từ W&B API, episode JSONL,
console W&B, output Kaggle, checkpoint và các rollout thực tế.
Source của hai run motion-study: `2e8ea830b3dfabc42e4a2604a6ad2ae12019ef03`.

## Kết luận

MASAC stochastic trong train làm nhiệm vụ tốt hơn trước khi thêm avoidance,
nhưng policy cuối chưa ổn định khi thực thi; MATD3 còn suy giảm rõ.
Cả hai chưa có bằng chứng tối ưu năng lượng toàn nhiệm vụ.
Không thể dùng reward đẹp hoặc PDR packet cao để kết luận hoàn thành mission.

Đã đọc đủ 800 episode của tám run 100-episode đã hoàn tất. Các bảng bên dưới
là trung bình episode trong **một training root seed 44**, không phải nhiều
training seeds. Hai run mới có seed evaluate duy nhất 10044, 2 env CPU,
6 UAV, 10 target, 8 obstacle, map 1000 m, tối đa 800 s/episode,
MLP [256,256], batch512, replay100000, warmup4000, 1 update/vector collection.
Native UavNetSim chạy CPU, learner/replay CUDA.

## Hai run mới: toàn bộ lịch sử và 20 episode cuối

- [MASAC 119fc307](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/119fc307)
- [MATD3 35837919](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/35837919)
- [Kaggle motion-study](https://www.kaggle.com/code/saoling1234/uavnet-train-20261013-113147-daa8a2)

Cả hai W&B finished, episode_budget_complete=true, stop_reason=episode_budget;
Kaggle COMPLETE, không phải time-cutoff.

| Chỉ số | MASAC: 100 ep | MASAC: 20 cuối | MATD3: 100 ep | MATD3: 20 cuối |
|---|---:|---:|---:|---:|
| Return | 465,62 | 534,53 | 161,35 | 151,80 |
| Search | 96,4% | 98,5% | 47,8% | 35,5% |
| Delivery về GCS | 82,9% | 93,0% | 30,6% | 28,5% |
| Success toàn mission | 30% | 55% | 2% | 0% |
| Hard block | 0% | 0% | 0% | 0% |
| Safety intervention | 58,72% | 55,47% | 91,57% | 97,24% |
| Movement saturation, abs(action)>=0,99 | 7,92% | 10,50% | 50,20% | 52,53% |
| Report hết TTL, trung bình/ep | 2,24 | 1,10 | 2,22 | 0,90 |
| Report drop, trung bình/ep | 0,63 | 0,35 | 0,06 | 0,05 |

MASAC 20 episode đầu: return457,70, delivery80,5%, success20%.
MATD3 20 episode đầu: return283,76, delivery50,5%, success10%.
Các episode train có exploration; episode maps theo một chuỗi seed xác định.
100 episode là lượt chẩn đoán, chưa chứng minh hội tụ.

MASAC có30 success/70 time_limit; MATD3 có2 success/98 time_limit.
Không episode nào trong hai run có UAV hết pin hoặc mất một UAV do pin.
Pin cuối trung bình49,20%/56,03%; pin UAV thấp nhất toàn lịch sử15,53%/38,98%.
Đây là failure do nhiệm vụ chưa xong trong deadline, không phải hết pin.

## Đánh giá checkpoint đóng băng: đúng map10044

| Policy/mode | Return | Search | Delivery | Success |
|---|---:|---:|---:|---:|
| MASAC ep50, mean/argmax | 52,24 | 100% | 20% | Không |
| MASAC ep100, mean/argmax | -57,57 | 60% | 0% | Không |
| MASAC ep100, lấy mẫu policy đã học | 271,98 | 70% | 50% | Không |
| MATD3 ep50, deterministic | 34,75 | 20% | 10% | Không |
| MATD3 ep100, deterministic | -15,08 | 0% | 0% | Không |

MASAC lấy mẫu Gaussian/categorical **đã học**, không thêm external exploration
noise. Lượt CPU sampling khởi tạo action RNG bằng44; map vẫn10044.
Kết quả sampling trên CPU không được coi là bitwise GPU inference parity.
Checkpoint không retrain giữa các hàng. Trục seed/mode được nêu rõ để không
gộp stochastic train với deterministic execution.

MASAC mean: có1681 quyết định mà một receiver hợp lệ tồn tại, nhưng silent1672
lần; GCS hợp lệ567 lần mà không chọn GCS lần nào. Có9 chọn peer, không chọn GCS,
9 expiry và1 drop; cả đội hết buffer cuối episode.
MASAC sampling:121 cơ hội receiver hợp lệ, silent93; chọn GCS10 lần;
5/10 target về GCS. Mạng có làm việc, phối hợp còn yếu.

MATD3 deterministic:9600/9600 thành phần movement có abs(action)>=0,99;
99,96875% tanh derivatives tính float32 bằng0. Có4682/4800 safety
interventions,4632 liên quan biên,4491 commanded stalls. Không thấy target,
không có report để truyền. Không có hard collision block không đồng nghĩa
policy điều hướng tốt: safety layer đang sửa/giảm lệnh của policy.

Learner log cũng có dấu hiệu cần theo dõi: MASAC alpha_discrete cuối là 148,413
đã chạm clamp exp(5); critic_grad_norm trước clip là 7691,05, còn MATD3 actor_grad_norm
là 0,03282 và critic_grad_norm là 173,55. Khoảng99% snapshot critic-gradient đã clip.
Đó là bằng chứng training còn khó; không chứng minh riêng sign của alpha loss
sai hoặc ReLU phải được thay. Checkpoints hữu hạn, chưa thấy numerical crash.

## Có đúng tìm-target → direct/relay → GCS không?

Có ở wiring và nhiều episode: camera phát hiện target trong footprint100 m
ở altitude100 m/FOV90°, có occlusion; ReportStore tạo report1024 B;
actor của từng UAV chọn XY acceleration, next-hop categorical và TX power0,1–0,4 W.
Native CSMA/CA, payload, PHY SINR, retry và ACK xử lý một hop; UAV receiver
giữ custody để policy chọn hop tiếp theo. Không có greedy routing thay policy.

Report hop trung bình theo episode là3,03(MASAC)/1,94(MATD3);
các native tests kiểm tra direct,2 hop và3 hop tới GCS.
Đây là phát hiện trực tiếp trong simulator, không có belief/Bayesian confirmation
trong repository mới. Sáu actor có tham số riêng, critic dùng joint state/action.
MASAC/MATD3 là các bản thích nghi cho action lai của mission này.

Mask chỉ xét range, active và receiver admission; không cam kết SINR.
Link qua obstacle vẫn NLOS và có thể thử/thất bại; sender giữ report để thử lại.
Tăng công suất không xóa suy hao. Nó có thể nâng tín hiệu nhận lên đủ SINR,
trong giới hạn công suất và điều kiện nhiễu. Max power vẫn có thể không đủ.

## Năng lượng: có penalty, chưa chứng minh tối ưu

| Trung bình/episode | MASAC | MATD3 |
|---|---:|---:|
| Propulsion cả6 UAV | 844848,92 J | 731252,10 J |
| UAV RF, gồm retries/ACK | 0,071298 J | 0,019503 J |
| Packet PDR | 83,77% | 87,32% |
| PHY success | 68,08% | 77,86% |

Flight energy dùng vận tốc/gia tốc thực sau safety; RF tính P×frame airtime,
gồm frame lỗi và retries, trừ battery thật. Replay/actor có power action và
reward có cả propulsion penalty0,01/kJ và RF penalty0,01/J.

RF penalty trung bình của MASAC chỉ0,000713 reward/episode, so với flight
penalty8,448 và delivery reward50/target. Report chỉ1 KiB, không phải1 MB.
Policy có động cơ năng lượng nhưng tín hiệu RF hiện cực nhỏ.
Mô hình RF chưa bao gồm công suất mạch điện, RX electronics hoặc beacon:
control-plane admission/beacon là lý tưởng và không có airtime/energy riêng.
Không gọi RF là toàn bộ năng lượng communication hardware.

MATD3 dùng ít năng lượng hơn chủ yếu cùng với tìm/đưa về ít target và đứng biên.
Đó chưa phải tiết kiệm năng lượng trên cùng kết quả nhiệm vụ.
Muốn kết luận tối ưu cần so năng lượng/delay/reliability ở mức delivery/success
tương đương và đối chứng có/không power adaptation, cùng map/seed/protocol.

## Các lỗi đã tái hiện và sửa

### 1. Bootstrap vượt deadline

max_steps800 là deadline của mission, có thời gian trong actor observation/state.
Code cũ trả truncated=True, terminated=False; replay/Bellman vì vậy cộng
gamma×Q của thời điểm sau khi nhiệm vụ đã hết giờ.
Ở reproduction một bước, reward=-0,0184 và giả sử nextQ100 thì target sai98,9816.
Ở checkpoint cuối, terminal next value agent0 còn khoảng693,78(MASAC mean)
và239,07(MATD3), dù simulator không cho bước mission tiếp theo.

Sửa tại môi trường: deadline → terminal; max_train_seconds của runner vẫn là
giới hạn ngoài và không tự chế thêm terminal episode khi dừng giữa chừng.
CUDA regression yêu cầu terminal Bellman target bằng reward cho cả hai learner.
Đây là lỗi xác định; chưa chứng minh nó là nguyên nhân duy nhất của saturation.

Tham khảo sơ cấp: [Gymnasium: intrinsic finite horizon và bootstrapping](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

### 2. Có knowledge nhưng mất report vĩnh viễn sau TTL/drop

Code cũ giữ known=True khi report hết TTL hoặc không vào được buffer;
camera nhìn lại target cũng không tạo report. Sửa cho phép tạo generation mới
chỉ sau quan sát camera mới của UAV đã chịu loss, khi target chưa delivered
và buffer đủ chỗ. Không xóa knowledge/tính lại discovery reward; không refresh
TTL trong relay; không tự sinh lại report sau custody transfer thành công.

Trường report status hiện có dùng >0 cho remaining TTL,0 cho không có custody,
-1 cho local loss cần fresh observation. Actor/critic thấy recovery metadata;
không thêm dimensions hay tọa độ target khuất. Native regression đã hết TTL,
quan sát lại, rồi truyền generation mới về GCS thành công.

### 3. ACK quá yếu làm sender phát lại

Code cũ ACK luôn0,1 W kể cả khi data cần0,4 W mới vượt threshold.
Ca native NLOS cố định, cùng layout/seed44/data0,4 W:

| ACK | PHY successes/failures | UAV RF | Report đến peer |
|---|---:|---:|---|
| Cũ0,1 W | 5/3 | 0,008692 J | Có |
| Theo selected data power0,4 W | 2/0 | 0,0017792 J | Có |

Sửa ACK dùng công suất của frame data nhận được; battery/RF accounting vẫn
tại native PHY. Cặp này giảm79,53% RF trong **ca cố định này**, không phải
cam kết giảm cùng phần trăm cho mọi episode hoặc đã học tối ưu công suất.
ACK có thể vẫn lỗi khi interference thay đổi; không bảo đảm mọi link.

### Console và backward compatibility

Console=wrap vẫn online. Thêm expired/dropped/regenerated vào từng episode;
cảnh báo rolling shield intervention, commanded stall và movement saturation
để hard_blocks=0 không che hành vi bám biên.

Legacy checkpoint thiếu rediscovery_recovery/ack_uses_data_power load cả hai=False.
Giữ dimensions và network semantics cũ khi đánh giá lịch sử.
Fresh config bật hai bản sửa. Không âm thầm cập nhật weights của các run cũ;
training với dynamics/termination mới phải đánh giá lại. Adam/replay full-state
resume chưa có; warm-start model không phải exact continuation.

## Reproducibility

Hai MASAC receiver-fixed/repeat có cùng model/episode/settings hashes sau100 ep;
hai MATD3 fixed/repeat cũng vậy. Cả hai đã lặp dài trên vuliu123456 và haibro1234.
Native smoke lặp lại trong CUDA suite vẫn kiểm tra hashes của cả hai thuật toán.
Giữ cùng source/settings/platform và đầy đủ episode budget; seed44 riêng lẻ
không bảo đảm giống kết quả sau khi đổi code/hardware/software hoặc warm-start.

## Verification và GPU follow-up

CPU suite sau code fixes:126 tests,119 pass,7 CUDA skip trên host không có CUDA.
Tất cả8 audit regressions pass; diagnostics6 pass; independent reviewer không
còn material findings. Các lỗi được thấy fail trước khi sửa (deadline,report,
ACK và aliased recovery state); source-size contract500 lines/file vẫn pass.

GPU validation của source `4341196` đã COMPLETE trên account `saoling1234`:

- [Kaggle](https://www.kaggle.com/code/saoling1234/uavnet-validate-20261013-152836-e37191).
- Hai Tesla T4,14,56 GiB/card, Python3.13.15, NumPy2.5.4,
  PyTorch2.11.0+cu128, CUDA12.8.
- **7/7 CUDA tests pass**,33,326 s. Có finite/replay transfer, native spawned
  learners, warm restart, repeated seed44 model/episode hashes, và Bellman
  target ở mission deadline bằng reward cho cả MASAC/MATD3.
- **8/8 native mission regressions pass** trên host Kaggle,2,727 s;
  gồm recovery/ACK/legacy config và recovery-status observability.
- [MASAC smoke](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/db4394d5)
  và [MATD3 smoke](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/f8c44aac):
  mỗi run 4 episode,160 transitions,77 optimizer updates, CUDA learner/replay,
  W&B finished và episode budget complete. Cả checkpoint hữu hạn; CPU evaluate
  đều hoàn tất1 episode. Smoke này không chứng minh convergence.
- W&B API xác minh source SHA,35 history rows mỗi run, global_step tăng đơn điệu,
  console nonempty 2809/2779 B. Group `uav-network-2d-validation` giữ smoke riêng
  với các run training trong Personal Workspace mặc định.
- [Bằng chứng máy đọc được](training-audit-cuda-evidence.json).

Source code kiểm thử là43411966942d1067d6098cf1a4218959d66e81f1;
commit bổ sung bằng chứng chỉ thay documentation, không đổi code đã kiểm thử.

Các log gốc đã đọc: MASAC train78629 B, MATD3 train80044 B,
W&B captured console73044/74659 B. Không có traceback trong log train/console,
checkpoint hữu hạn và job-result hoàn tất. Có report_loss/return_drop warnings.
Điều này không đồng nghĩa policy tốt hoặc không còn thiếu sót mô hình.

Không tăng episode, giảm penalty, đổi activation/norm hay chạy nhiều training
seeds để làm các số hiện tại đẹp hơn. Ba bản sửa chữa semantics/wiring thật;
cần fresh retraining mới kết luận được learning/success có cải thiện.
