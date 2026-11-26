# Nguồn và phạm vi đọc — kiểm ngày 26/11/2026

| Nguồn gốc | Phạm vi dùng / giới hạn | URL |
|---|---|---|
| Zhang et al., IEEE T-IV2024 | Đọc full PDF của người dùng; Eq6–17: vacuous mass, simple-support update, max count, closed form; không chứng minh GPU network parity | https://doi.org/10.1109/TIV.2024.3352581 |
| Sokar et al., ICML2023 | Đọc định nghĩa per-neuron normalized mean activity và ReDo; không đồng nhất zero-output fraction với dormant-neuron fraction | https://proceedings.mlr.press/v202/sokar23a.html |
| Ball et al., ICML2023 | Paper RLPD, vai trò LayerNorm/regularization trong benchmark riêng; không đưa ra nghiệm SiLU+Huber10 cho mission này | https://proceedings.mlr.press/v202/ball23a.html |
| Nauman et al., ICML2024 | Abstract và thiết kế nghiên cứu regularization; không lấy kết quả paper làm proof nguyên nhân ở log UAV | https://proceedings.mlr.press/v235/nauman24a.html |
| Fujimoto et al., ICML2018 | TD3 twin Q, delayed policy, target smoothing; opt-in mới không được coi author baseline nguyên xi | https://proceedings.mlr.press/v80/fujimoto18a.html |
| Wang/Ames/Egerstedt, T-RO2017 | Abstract/QP safety certificates có giả thiết/feasibility; không bảo đảm mọi bounded acceleration state; không triển khai lại CBF | https://doi.org/10.1109/TRO.2017.2659727 |
| Ng/Harada/Russell, ICML1999 | Potential shaping gamma*Phi(next)-Phi(current), không đồng nhất local uncertainty reward | https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf |
| Yates et al., JSAC2021 | AoI status timeliness, phân biệt evidence truth permanence với packet freshness | https://doi.org/10.1109/JSAC.2021.3065072 |
| Zhou et al., UavNetSim-v1 ICCC2025 | Đọc full extracted paper qua alphaXiv và arXiv; Table I noise4e-11W; không gọi hội nghị này A* | https://arxiv.org/abs/2507.09852 |
| UavNetSim upstream hiện hành | Pin c2703f24eb58cdb557a5f67ded5c7b0368a964ed; utils/config.py dùng thermal noise density-174, BW22MHz, NF7dB; không đồng nhất với Table I2025 | https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/c2703f24eb58cdb557a5f67ded5c7b0368a964ed/utils/config.py |
| ITU-R P.2109-2 | Trang tiêu chuẩn building entry loss; không hiệu chuẩn 12dB xuyên cylinder | https://www.itu.int/rec/R-REC-P.2109-2-202308-I/en |
| NVIDIA Warp | Context7 trả tài liệu chính hãng về stream interop, synchronize, zero-copy; giữ every-step overflow guard | https://github.com/NVIDIA/warp/blob/main/docs/user_guide/execution_and_performance/concurrency.rst |
| NVIDIA skill catalog | Đã kiểm live catalog; warp-compile-time-optimizer phân biệt cold/warm, cache isolation và không suy runtime gain từ compile gain; không cài thêm capability | https://github.com/NVIDIA/skills/tree/main/skills/warp-compile-time-optimizer |
| PyTorch HuberLoss | Huber residual gradient, small-error scale; không khẳng định Huber chặn Q target | https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html |
| ICORE2026 | ICML A*, AAAI A*, kiểm theo portal chính thức | https://portal.core.edu.au/conf-ranks/?by=acronym&search=ICML |
| ICORE2026 AAAI | A* theo portal, không tự áp cho ICCC | https://portal.core.edu.au/conf-ranks/?by=acronym&search=AAAI |
| SJR2025 | T-IV Q1 theo Automotive Engineering, không phải “top mọi ngành” | https://www.scimagojr.com/journalrank.php?category=2203 |

## Tool thực sự dùng

Vast: source/test/CUDA/log, không thuê instance mới. Sandbox: kết nối laptop, SSH/SCP, Git publication và nhận file. GitHub: đọc upstream, pin source và xác nhận nhánh. NVIDIA: catalog; Context7: official docs. Exa, Parallel, alphaXiv, Octen, Firecrawl, Undermind và web search: truy hồi nguồn; paper findings chỉ dựa nguồn gốc nêu trên. Superpowers: quy trình tìm nguyên nhân và test RED/GREEN, không phát sinh sub-agent.

Consensus trả quota hết tháng; không có kết quả dùng làm bằng chứng. [Trang tài khoản Consensus](https://consensus.app/continue?from=mcp&client=chatgpt&intent=upgrade). Nếu đã xử lý tài khoản và muốn dùng lại, nhắn “try again”. Liner trả payment_method_required, không có kết quả dùng làm bằng chứng. [Thông báo billing Liner](https://platform.liner.com/pay?_ptxn=txn_01m6gbdsawhqwdkyhescmcpz5g). Không retry trả phí hoặc thay đổi billing. Firecrawl scrape ITU lấy được trang và link PDF nhưng có nhiều menu; feedback sau cửa sổ thời gian bị từ chối, không được tính là thành công. Không cần thêm plugin để tiếp tục công việc đã làm.

Phần full text nghiên cứu công khai/trả phí không được chép toàn bộ vào archive. Raw code/log do dự án sinh ra, các nguồn tham chiếu, số liệu trích xuất và đường dẫn kiểm chứng được giữ riêng.
