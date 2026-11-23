# Kiểm toán paper Q1/A* và source code MASAC/MATD3 — 08/11/2026

## Phạm vi và cách đánh giá

Mục tiêu: tìm paper MASAC / MATD3 (không nhất thiết là UAV), truy ngược paper thuật toán nền, xác minh source GitHub thực sự có code training và thử chạy phần reference trên CPU độc lập trước khi xem xét chuyển sang UAV SAR.

**Xếp hạng**: “Q1” chỉ dùng khi có nguồn hạng *tạp chí* (SJR/Scopus hoặc JCR, ghi rõ khi có thể); “A* / A” thuộc **hội nghị** qua ICORE. AAMAS đáng chú ý **CORE2023 A*** nhưng **ICORE2026 đã hạ A**: https://portal.core.edu.au/conf-ranks/922/ . Workshop NeurIPS không tự động là paper main-track A*.

**Trạng thái thực thi**: Đã clone/source-audit 7 GitHub repositories (cùng các source single-agent kiểm tra qua web); chạy smoke thành công **BenchMARL MASAC + VMAS**. Không có Kaggle/GPU, không thay đổi code thuật toán MASAC/MATD3 UAV, không đưa source không có license vào repo sản phẩm.

## 1. Bảng ưu tiên: papers, xếp hạng và code thực tế

| Paper/thuật toán | Venue/rank | Source và khả năng chạy | Mức phù hợp với bài toán UAV |
| --- | --- | --- | --- |
| Bettini et al., [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html), 2024 | **JMLR — Q1 (Scopus/SJR)** | [facebookresearch/BenchMARL](https://github.com/facebookresearch/BenchMARL), clone **65d649d80e0bdcbdbe2c5d6a3f02dbfed8f0bec1**; `benchmarl/algorithms/masac.py`, **chạy CPU training thành công** | **Best executable MASAC reference** PyTorch/TorchRL; supports **continuous OR discrete**, không xử lý đồng thời hybrid `(XYZ,RF power,destination mask)` ngay. |
| Liu et al., [AC-MASAC](https://arxiv.org/abs/2602.11735), 2026 | **AAMAS 2026 — ICORE2026 A** (CORE2023 A*) | [Wanhao-Liu/AC-MASAC](https://github.com/Wanhao-Liu/AC-MASAC), clone **771465c**, >100 files, `main_SAC_curriculum.py`, `masac_adapter/actor_networks.py`, `critic_networks.py`, `rl_env` | **UAV gần nhất có source train thực**. Nhưng MASAC + attention + curriculum + role-specific actor/replay, **không phải unmodified MASAC baseline**, không có hybrid RF receiver mask. Chưa chạy end-to-end trên máy hiện tại. |
| Fujimoto et al., [TD3](https://proceedings.mlr.press/v80/fujimoto18a.html), 2018 | **ICML — A*** | [sfujim/TD3](https://github.com/sfujim/TD3), **mã tác giả**, PyTorch single agent; tác giả ghi README đã chỉnh nhẹ code sau paper | **Primary reference** về TD3: twin min-Q, delayed actor update, target smoothing; cần MADDPG/Ackermann để ra MATD3. |
| Haarnoja et al., [SAC](https://proceedings.mlr.press/v80/haarnoja18b.html), 2018 | **ICML — A*** | [Spinning Up SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html), mã tham chiếu độc lập uy tín; không phải MASAC | **Primary citation** cho Gaussian policy, soft Bellman backup, auto temperature (theo SAC variant), tanh Jacobian. |
| Lowe et al., [MADDPG](https://arxiv.org/abs/1706.02275), 2017 | **NeurIPS — A*** | [openai/maddpg](https://github.com/openai/maddpg), official, archived Apr 2026, TF1/legacy gym, code công khai | **Primary citation** cho centralized training/decentralized execution và joint-action Q. |
| Ackermann et al., [MATD3](https://arxiv.org/abs/1910.01465), 2019 | **NeurIPS Deep RL Workshop**, **không phải main-track A*** | [JohannesAck/MATD3implementation](https://github.com/JohannesAck/MATD3implementation), clone **fd6c7d0df4fc4effd4e0fa11abdbc7d12d8f01a1**, TF1.12, Gym0.10.5. Có `matd3/matd3/trainer/matd3.py` và train. | **Official MATD3 reference** (mã tác giả), nhưng dependencies rất cũ; không có UAV hybrid. |
| Ackermann, [tf2multiagentrl](https://github.com/JohannesAck/tf2multiagentrl) | Repository bổ sung, **không phải paper A*** | Clone **d5442f6**, MASAC + MATD3 + MADDPG TF2.x, training/test. README báo MATD3 tốt nhất với MPE | Code MASAC có **TODO Gaussian policy và entropy temperature learning**; `entropy_coeff` cố định, không phù hợp làm official full auto-temperature hybrid. Rất hữu ích bác bỏ giả định MASAC luôn hơn MATD3. |
| Yu et al., [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://papers.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html), 2022 | **NeurIPS — A*** | [marlbenchmark/off-policy](https://github.com/marlbenchmark/off-policy), clone **41fd5eb46d12df2847e1c2e29842997ff2c24998**, có MATD3 PyTorch training và twin-Q; **mã MATD3 hiện có bug Q-head không đăng ký** (bên dưới) | **Không sử dụng nguyên xi** nếu chưa sửa, audit lại và thống nhất action type. Có thể đọc tham số/kiến trúc. |
| Qiao et al., [Modelling Competitive Behaviors in Autonomous Driving Under Generative World Model](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5085_ECCV_2024_paper.php), 2024 | **ECCV — A*** | [qiaoguanren/MARL-CCE](https://github.com/qiaoguanren/MARL-CCE), clone **ad8e4c1**, **chỉ 2 file README + LICENSE**, không có code MASAC training; [repo liên kết khác](https://github.com/qiaoguanren/Multi-agent-Competitive-Autonomous-Driving) cũng chỉ README/LICENSE theo nội dung truy cập | Paper có CCE-MASAC mathematical idea nhưng **không có full reference training code ở repo kiểm tra**. |
| Jiang et al., [Discovering Latent Variables for the Tasks With Confounders in Multi-Agent Reinforcement Learning](https://www.ieee-jas.com/en/article/doi/10.1109/JAS.2024.124281), 2024 | **IEEE/CAA Journal of Automatica Sinica — SCI Q1** theo publisher | MASAC-LV, **chưa xác minh được source chính thức** | Có latent variable, continuous multi-agent coordination; có thể dẫn trong related work, không dùng như source copy-ready. |
| [Attentional MATD3 / AMATD3](https://www.sciencedirect.com/science/article/pii/S0952197625003525), 2025 | **Engineering Applications of Artificial Intelligence — Q1** JCR/SJR 2025 | **Chưa xác minh source tác giả** | Attention/communication đáng tham khảo, là biến thể MATD3 chứ không phải MATD3 nguyên bản. |
| [Hybrid-MASAC for Hexapod](https://proceedings.mlr.press/v270/fu25c.html), Fu et al., 2025 | **CoRL / PMLR 270**; chưa xác nhận A* theo ICORE2026 | **Chưa xác minh được source chính thức** | Rất gần công thức **hybrid continuous + discrete** của UAV, nhưng cấu trúc mạng paper có attention / ứng dụng chân robot khác. |
| [HR-MASAC UAV aerial monitoring](https://ieeexplore.ieee.org/abstract/document/11676089), 2026 | **IEEE Transactions on Vehicular Technology (Early Access)**; hạng Q1 cần phân biệt danh mục/năm | **Chưa xác minh source chính thức** | Gần nhất về **3D flight + sensing/communication + power**; chia hybrid action theo kiến trúc khác. |
| [ffelten/MASAC](https://github.com/ffelten/MASAC), 2023 | Code public GitHub; **không claim Q1/A*** | Clone **7936aa9a**, `masac/masac.py` PyTorch + JAX; README cảnh báo WIP/chưa test rộng | MASAC shared actor-ID, shared critic; **continuous PettingZoo only**, dùng CleanRL SAC, không mask destination. |

Source/rank backing:
- JMLR https://www.jmlr.org/papers/v25/23-1612.html ;
  https://journalindexes.com/journal-of-machine-learning-research-15324435/ (Scopus/SJR Q1);
- ICML/NeurIPS/ECCV and AAMAS ICORE https://portal.core.edu.au/conf-ranks/?by=all&page=1&search=&sort=arank&source=CORE2023 ;
  **AAMAS 2026 downgraded** https://portal.core.edu.au/conf-ranks/922/ ;
- JAS explicit SCI Q1 at https://www.ieee-jas.com/en/article/doi/10.1109/JAS.2024.124281 ;
- EAAI 2025 Q1 https://www.iit.comillas.edu/publicacion/info_revista/en/240/Engineering_Applications_of_Artificial_Intelligence ;
- Official AAMAS 2026 proceedings record https://dblp.org/rec/conf/ifaamas/LiuDZYL26 .

## 2. Citation genealogy — nên dẫn paper gốc của ai?

**MASAC** (không có một bản “MASAC gốc” duy nhất cho mọi action space):
1. **Haarnoja et al., ICML2018, SAC** — entropy maximum, soft Q/Actor.
2. **Lowe et al., NeurIPS2017, MADDPG** — MARL CTDE, centralized joint-action critic.
3. **Delalleau et al., Hybrid SAC**, https://arxiv.org/abs/1912.11077 — kết hợp continuous/discrete action.
4. **Fu et al., CoRL2025 Hybrid-MASAC**, https://proceedings.mlr.press/v270/fu25c.html — paper hybrid MARL ứng dụng robot, không sao chép attention architecture.
5. **Bettini et al., JMLR2024 BenchMARL** — phần triển khai chuẩn hóa TorchRL; chỉ cite như **implementation reference**, không nhận vơ tác giả SAC/MASAC đầu tiên.

**MATD3**:
1. **Fujimoto et al., ICML2018, TD3** — Q double-min, delayed actor, smoothing.
2. **Lowe et al., NeurIPS2017, MADDPG** — centralized multi-agent.
3. **Ackermann et al., NeurIPS Workshop2019, MATD3** — chuyển TD3 sang MARL, GitHub chính thức.
4. Nếu dùng reference PyTorch [marlbenchmark/off-policy](https://github.com/marlbenchmark/off-policy) thì cite repo source chính xác **và nêu đã sửa lỗi / kiểm thử**, không ngầm coi đó là nguồn gốc của MATD3.

Ví dụ các paper Q1 trong bảng cũng đi theo các chuỗi này: MASAC-LV của Jiang liên hệ SAC/MADDPG; attentional MATD3 liên hệ TD3/MATD3, ngoài những ý tưởng đóng góp mới của riêng các tác giả.

## 3. Kết quả **thực nghiệm source** — chạy xong, không phải kế hoạch

### 3.1. BenchMARL chính thức: MASAC CPU on VMAS

- Clone `facebookresearch/BenchMARL` ở commit **65d649d80e0bdcbdbe2c5d6a3f02dbfed8f0bec1**.
- Python riêng dưới `outputs/reference_source_audit_20261108/venv` (không thay system Python), dùng cùng PyTorch 2.7.1+cu126 CPU, `torchrl==0.10.1`, `tensordict==0.10.0`, `vmas==1.5.1`; `pyvers` bổ sung dependency.
- Do VMAS 1.5.1 dùng `from gym import spaces`, wrapper trong riêng Python tiến trình tạo gym alias từ gymnasium; **không chỉnh upstream code**.
- `algorithm=masac task=vmas/balance seed=44`, 4 agents, max episode length 20, **192 frames**, 6 iterations, 2 optimizer steps/iteration, 16 batch, buffer256, CPU only, CSV logging, W&B disabled, evaluation disabled, checkpoint disabled. Đã có episode kết thúc và return logged.
- `train_agents_alpha` **0.99997497 → 0.99947548**.
- `train_agents_loss_qvalue` **3.100564 → 2.057223**.
- `train_agents_loss_actor` **−1.335954 → −1.528385**.
- Đây **không phải full convergence, không phải UAV SAR và không đo delivery rate**. Việc train được không chứng minh tương thích action hybrid.

**Tái hiện** nếu đã có dependencies/clone:
`outputs/reference_source_audit_20261108/venv/bin/python scripts/research/run_benchmarl_masac_cpu_smoke.py`.
Machine-readable evidence:
`docs/methods/evidence/reference_benchmarl_masac_smoke_2026-11-08.json`.
Script chủ ý chạy trong `outputs` gitignored; không ghi vào repo source upstream.

### 3.2. Upstream MATD3 PyTorch: Q heads không cập nhật do Python list

File thực tế ở commit **41fd5eb46d12df2847e1c2e29842997ff2c24998**
`offpolicy/algorithms/maddpg/algorithm/actor_critic.py`
chứa `self.q_outs = [nn.Linear(...), nn.Linear(...)]` (qua `init_`).
Trong PyTorch, Python list không tự đăng ký parameters. Parent
`MADDPGPolicy` tạo Adam bằng `self.critic.parameters()`,
cho nên cả 2 output heads **nằm ngoài optimizer**.

Đã chạy trực tiếp `MATD3_Critic` upstream với input cố định,
3 bước Adam, seed 44:
- `q_head_registered = [false,false]`.
- `q_head_grad_exists = [true,true]`.
- `q_head_weights_changed_after_3_adam_steps = [false,false]`.

Sửa **trong bộ nhớ** (không thay file upstream) thành `torch.nn.ModuleList`
thì kết quả chuyển thành
`q_head_registered=[true,true]` và
`q_head_weights_changed_after_3_adam_steps=[true,true]`.

Reproducer:
`scripts/research/verify_upstream_matd3_critic_registration.py`.
JSON:
`docs/methods/evidence/reference_matd3_q_registration_2026-11-08.json`.
**Đây là lỗi ở clone upstream riêng**, không phải chứng cứ rằng
`uav_network/algorithms/matd3.py` của người dùng có lỗi đó.
Mã UAV dùng `nn.ModuleList` + `TwinCritic` Q towers đăng ký hợp lệ.

### 3.3. Nguồn TF2 MARL của tác giả MATD3

`JohannesAck/tf2multiagentrl`, commit **d5442f6**,
`tf2marl/agents/masac.py` có docstring `todo: entropy temperature learning`,
`todo: gaussian policy`; SAC dùng `entropy_coeff` cố định ở phiên bản
này. README ghi MATD3 tốt nhất trong thí nghiệm MPE của họ,
không thấy cải thiện đáng kể từ SAC.
**Không có quy tắc MASAC luôn luôn phải vượt MATD3**.
Đây không phải reference đầy đủ cho **hybrid SAC có alpha tự học**
trong UAV hiện tại.

## 4. Khả năng port sang bài toán UAV SAR 3D

UAV hiện tại: 6 UAV, 35 target, 25 obstacles, 1500 steps,
continuous XYZ+RF power, discrete receiver 0…6 masked,
local partially observable actor, privileged centralized critic,
CBF shield, Bayesian belief, Packet PHY. Vì thế:

| Thành phần | BenchMARL MASAC / nguồn SAC | Code UAV |
| --- | --- | --- |
| Reparameterized continuous distribution | `SACLoss`, `TanhNormal` | `Normal.rsample` + `tanh`, log-Jacobian |
| Discrete SAC | `DiscreteSACLoss` chọn thay vì continuous | Categorical exact expectation đồng thời continuous |
| Action mask `destination` | Có `MaskedCategorical` dùng trong một số task | Support mask, silence khi không receiver hợp lệ |
| Continuous power conditional on route | Không trực tiếp trong general SAC run | `canonical_continuous`, power=0 khi silent |
| CTDE centralized joint action Q | Có MASAC centralized critic | Có joint state and joint action; shared/independent cases |
| Safety shield/CBF + report queue | Không trong VMAS Balance | Có, làm learning dynamics khác nhiều |

**Khuyến nghị**: Giữ baseline UAV hiện tại ở nhánh riêng; lấy BenchMARL
để làm reference tests cho phần SAC continuous, masked discrete
và autodiff; dùng Hybrid-SAC (Delalleau/Fu) để kiểm tra toán kết hợp.
Nếu cần true independent external baseline, phải viết **adapter rõ ràng**
cho hybrid UAV và kiểm thử 1-1 các hành động, masks, Q target, replay,
safety shield; ghi là **adapted from BenchMARL**,
không viết “chạy nguyên BenchMARL trên UAV” khi chưa thực hiện.

## 5. Trạng thái / giới hạn

- **Đã**: điều tra papers Q1/A*, kiểm tra clone source thật, xác minh code availability/commits, chạy MASAC BenchMARL trên VMAS, tìm & xác thực bug source MATD3 Python list, tạo script và JSON tái hiện.
- **Chưa**: train MASAC/MATD3 Q1/A* full 35T/25O, khởi chạy Kaggle GPU, đánh giá sample efficiency giữa các implementation trên cùng env, port full hybrid, kiểm chứng tác giả code tất cả papers chưa công bố.
- GPU sandbox hiện không có CUDA driver; smoke test CPU 192 frames là **chỉ một sanity check**.
- Các source trong `outputs/reference_source_audit_20261108/` là *clone gitignored*, không commit lại mã tác giả lên repo UAV.
