# Nguồn gốc, lựa chọn và giới hạn sử dụng

| Nguồn | Đã kiểm tra / lý do dùng | Link |
|---|---|---|
| Krasowski et al., TMLR2023, arXivv3 | Đọc full extracted text, nhất là mục2.5 và discussion. Có naive/adaption penalty/safe action/both; không chứng minh raw tuple là bug ở mọi off-policy system. | https://arxiv.org/html/2205.06750v3 |
| Gros, Zanon, Bemporad, IFAC2020 | Đọc HTML/PDF, projected policy và các điều kiện sửa gradient; không chuyển chứng minh MPC differentiability sang shield66candidate. | https://arxiv.org/html/2004.00915v1 |
| Agrawal/Sreenath, RSS2017 | Trang proceedings gốc về discrete control barrier functions; dùng để phân biệt discrete-time assumptions. Không triển khai CBF trong project. Không tự chứng nhận rank RSS trong lượt này. | https://www.roboticsproceedings.org/rss13/p73.html |
| Batra et al., CoRL2021 proceedings2022 | Publisher abstract về decentralized quadrotor swarms; là căn cứ thiết kế observation, không bảo đảm HELLO/schema hiện tại. | https://proceedings.mlr.press/v164/batra22a.html |
| NVIDIA Warp official docs | Context7 trả docs mặc định Warp blocking stream và Torch null stream có implicit sync; custom nonblocking stream cần contract khác. | https://github.com/NVIDIA/warp/blob/main/docs/user_guide/execution_and_performance/concurrency.rst |
| UavNetSim current upstream | GitHub commit vẫn c2703f24eb58cdb557a5f67ded5c7b0368a964ed; alias UavNetSim-v1 redirect về cùng repository ID955363828. | https://github.com/Zihao-Felix-Zhou/UavNetSim |
| UavNetSim2025 paper | TableI noise4e-11W khác current upstream thermal-noise formula. Full-text và code comparison được giữ trong audit V82. | https://arxiv.org/abs/2507.09852 |
| 3GPP TR36.777 | Official portal xác nhận technical report “Enhanced LTE support for aerial vehicles”, Release15 và archive versions. Chưa dùng nó để calibrate cylinder12dB. | https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3231 |
| LTE drones overview | Bài arXiv1805.00826 là overview, không chính bản TR36.777. | https://arxiv.org/abs/1805.00826 |
| Zhang et al., T-IV2024 | Full PDF của người dùng đã đọc ở V82; DS simple support/count fusion, max-vs-sum và aging là phạm vi mô hình riêng. | https://doi.org/10.1109/TIV.2024.3352581 |
| Sokar et al., ICML2023 | Dormant-neuron definition khác zero-sample output fraction; giữ kết luận V82, không bảo đảm SiLU giải hết safety. | https://proceedings.mlr.press/v202/sokar23a.html |
| Ng et al., ICML1999 | Potential shaping có gamma và terminal assumptions; không tự gán invariance cho local uncertainty reward. | https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf |
| ITU-R P.2109-2 | Building entry loss khác deterministic loss mỗi cylinder; không được suy calibration. | https://www.itu.int/rec/R-REC-P.2109-2-202308-I/en |
| ICORE / SJR | Bằng chứng ICML/AAAI A* ICORE2026 và T-IV Q1 SJR2025 từ lượt V82 được giữ. Không gọi tiêu chuẩn/ICCC là Q1/A*. | https://portal.core.edu.au/conf-ranks/?by=acronym&search=ICML |

## Tool sử dụng thực tế

Vast: source, CPU/CUDA, các cohort diagnostic. Sandbox: SSH/SCP đồng bộ/lưu bản cuối và Git publication. GitHub: source/pin/redirect/nhánh. Context7/NVIDIA docs: interop; NVIDIA catalog đã kiểm trong lượt trước cùng cuộc hội thoại, không cài capability. Exa/Parallel/alphaXiv/websearch/Firecrawl/Undermind: truy hồi nguồn và metadata. Firecrawl3GPP đã đọc trang chuẩn và gửi feedback thành công. Superpowers: điều tra trước khi sửa, RED/GREEN và kiểm tra trước khi kết luận. Không spawn sub-agent trong lượt này.

Consensus đã trả hết quota tháng và Liner yêu cầu payment trong lượt V82; không retry sau khi tool yêu cầu dừng. Không có kết quả từ chúng dùng làm bằng chứng. [Consensus](https://consensus.app/continue?from=mcp&client=chatgpt&intent=upgrade); [Liner billing notice](https://platform.liner.com/pay?_ptxn=txn_01m6gbdsawhqwdkyhescmcpz5g). Nếu tài khoản Consensus đã xử lý và cần dùng lại, nhắn “try again”. Không thay đổi billing.

OpenReview direct bị trang verification; dùng author arXiv full text, không coi trang challenge là paper. Undermind nhận diện paper Krasowski theo preprint2022; bản TMLR/v3 được công bố2023. Browser không cần phiên đăng nhập cho các nguồn công khai này.

Raw text paper bên ngoài không được chép toàn bộ vào gói. Source code/log của project, liên kết nguồn, phép tính độc lập và phạm vi citation được giữ để kiểm chứng.
