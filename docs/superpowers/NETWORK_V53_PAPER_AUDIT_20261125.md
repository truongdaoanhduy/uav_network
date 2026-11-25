# Đối chiếu nguồn sơ cấp cho UAV v5.2

Ngày kiểm tra: 2026-11-25. Phạm vi: hai PDF người dùng cung cấp, mô hình pin/thời gian/AoI, định vị đóng góp và nguồn UavNetSim. Đây là kiểm tra nội dung khoa học; không thay đổi mã nguồn hay xác nhận production Warp/CUDA.

## 1. Kết luận có thể dùng ngay

- Zhang dùng **MAX đúng như bản PDF**, không phải phép cộng tổng quát trên mọi UAV. Thay bằng cơ chế MAX theo nguồn rồi SUM giữa các nguồn là một thiết kế mở rộng cần được mô tả và đánh giá riêng.
- Xue đã có DS, vùng blackout theo vị trí, lưu đệm có timestamp, giảm trọng số theo tuổi và đồng bộ khi thoát vùng chặn. Tuy nhiên bài không đặc tả một hệ thống truyền gói PHY/MAC/routing về GCS. Các quan sát actor còn chứa bản đồ niềm tin **toàn cục**.
- Công thức công suất chỉ phụ thuộc tốc độ của Zeng là mô hình xấp xỉ bỏ qua phần năng lượng do gia tốc. Không thể gọi ngân sách pin được suy ra từ công thức đó là đã xác thực cho chuyển hướng/gia tốc thường xuyên.
- Pardo phân biệt **deadline thuộc nhiệm vụ** với **ngưỡng dừng kỹ thuật**. Chỉ đổi deadline thành pin không tự động chứng minh toàn bộ local observation có tính Markov.
- AoI ở GCS phải dựa vào thời điểm sinh phép đo mới nhất **đã đến GCS**, theo đúng thực thể/ô/nguồn cần đo. Không đặt tuổi bằng 0 khi một UAV khác đang nhìn thấy ô đó.
- RAI-CoS có thật và gần v52 hơn mô tả hiện tại: đã có provenance, chống tái sử dụng bằng chứng, CI, xử lý báo cáo trễ và lập lịch theo byte/virtual queue. Khoảng trống còn có thể kiểm chứng là vòng lặp truyền gói và chuyển động dưới mô hình kênh đã công bố, trong nhiệm vụ tìm kiếm mục tiêu ẩn cụ thể.
- Nguồn UavNetSim đang bị trộn phiên bản. Tag `v1.0.1` không có `phy/a2a.py`; file này có trên `master` mới hơn. Phải cố định commit và ghi các phần tự mở rộng.
- Chưa xác thực xếp hạng JCR/CORE theo đúng năm và ngành; không dùng nhãn Q1/Q2/A* như một kết quả đã kiểm chứng.

## 2. Zhang: công thức và reward gốc

Nguồn chính là PDF đầy đủ 16 trang đã cung cấp. Đã đọc toàn văn và kiểm tra hình trang PDF 5, 7, 10. Metadata xác nhận với [IEEE](https://ieeexplore.ieee.org/abstract/document/10388472/authors), DOI [10.1109/TIV.2024.3352581](https://doi.org/10.1109/TIV.2024.3352581).

Thư mục đang ghi sai tác giả đầu: phải là **Boquan Zhang (B. Zhang)**, không phải `J. Zhang`. Các tác giả: Boquan Zhang, Xiang Lin, Yifan Zhu, Jing Tian, Zhi Zhu; *IEEE Transactions on Intelligent Vehicles*, 9(2), 3827–3842, 2024.

### Đối chiếu phương trình

| Vị trí gốc | Nội dung | Hệ quả cho v52 |
|---|---|---|
| PDF 3, trang 3829 | Giả định mạng liên lạc đầy đủ để tập trung nghiên cứu bản đồ niềm tin | Không phải baseline vô tuyến đa chặng đã có hàng đợi và tranh chấp |
| PDF 4–5, Eqs. (3)–(13) | Khung DS `{E,F}`, mass cho có/không có mục tiêu và bất định; simple support với `m_f`, `m_e` | `P_d`, `P_f` mô tả kênh quan sát; `m_f`, `m_e` là trọng số bằng chứng khác nghĩa, không tự động bằng nhau |
| PDF 5, Eqs. (14)–(15) | Bộ đếm quan sát dương/âm từng UAV | Cần xác định rõ counter là quan sát riêng hay đã đồng bộ |
| PDF 5, Eq. (16) | `N_c(+) = max_i N^i_c(+)`; `N_c(-) = max_i N^i_c(-)` | Giữ làm baseline tái hiện bài; không gọi SUM là Eq. (16) |
| PDF 5, Eq. (17) | Mass dựng từ hai counter sau MAX | Không có provenance theo nguồn trong phương trình này |
| PDF 7, Eqs. (27)–(33) | Giảm tổng bất định, vượt ngưỡng phát hiện, phạt va chạm/vùng cấm và ra biên | Không có reward delivery tại GCS/AoI/airtime/pin |

Đặt `f=(1-m_f)^(N+)`, `e=(1-m_e)^(N-)`, `D=f+e-ef`. Eq. (17) là:

\[
m(E)=\frac{f-ef}{D},\quad m(F)=\frac{e-ef}{D},\quad m(U)=\frac{ef}{D}.
\]

Reward Eq. (27) là `ω1 Σ_c [m_{t-1,c}(U)-m_{t,c}(U)]`; Eq. (28) thưởng khi `m_t(F)≥τ` và trước đó `<τ`. Đây là crossing của niềm tin, không trực tiếp là ground-truth TP hay GCS đã nhận. PDF 10, Table II: `m_f=m_e=0.8`, `τ=0.95`; weights `ω1=30, ω2=10, ω3=30, ω4=20`. v52 đổi `τ=0.9` và các reward là cấu hình mới, không phải bản sao bài.

**Suy luận kiểm toán:** Với hai UAV có các quan sát riêng, MAX mất đóng góp độc lập. Ví dụ A có một dương, B có một dương: pure MAX cho 1, bộ đếm theo hai nguồn riêng cho 2. Mô hình `count[receiver,origin,cell,sign]` có thể merge bằng MAX **cùng origin**, sau đó SUM **khác origin** để đếm đúng các event phân biệt. Đó là đề xuất thiết kế dựa trên provenance, không phải định lý của Zhang. Nó còn đòi hỏi mỗi nguồn chỉ tăng counter bằng quan sát mới của chính mình, reset/epoch rõ ràng, và không gộp dữ liệu có trọng số theo tuổi thành hai counter không giữ lịch sử.

Các lỗi in nhỏ trong bài (ví dụ lời dẫn “five components” nhưng Eq. (33) có bốn hạng, Eq. (31) in `u_i-u_i`) không nên được tái hiện thành lỗi triển khai.

## 3. Xue: giới hạn mô hình liên lạc cần diễn đạt chính xác

Nguồn: PDF 25 trang đã đọc toàn văn; đối chiếu hình trang 7, 8, 12; [toàn văn nhà xuất bản](https://www.mdpi.com/2079-9292/15/9/1875), DOI [10.3390/electronics15091875](https://doi.org/10.3390/electronics15091875). Tác giả Lingtao Xue, Xuewen Dong, Xinyu Hu, Lingxiao Yang, Gang Xiao; đăng 29-04-2026.

| Thành phần | Vị trí | Nội dung đã xác minh |
|---|---|---|
| Blackout | PDF 6, §3.1.1 | `Φ:Ω→{0,1}` tĩnh theo episode, vùng nối dưới khoảng cách Manhattan; bản đồ khả dụng cho các UAV |
| Cảm biến | PDF 6, Eqs. (1)–(3) | FOV ℓ1; hit/false alarm phụ thuộc khoảng cách |
| DS | PDF 7, Eqs. (4)–(7) | Mass ba trạng thái; conflict và chuẩn hóa Dempster |
| Truyền khi vùng mở | PDF 7 | Bằng chứng được nhập ngay vào bản đồ toàn cục khi `Φ=0` |
| Buffer và chỉ số tuổi | PDF 7, Eq. (8) | Lưu bằng chứng/timestamp khi `Φ=1`; `θ` tăng trong blackout và về 0 ở vùng mở |
| Đồng bộ | PDF 7–8, Eq. (9) | Thoát vùng chặn thì tải tất cả buffer; nhân hai mass đơn bằng `exp[-η(t-ts)]` |
| Quên toàn cục | PDF 8, Eq. (10) | Mỗi bước nhân mass có/không mục tiêu với retention `γ`, phần còn lại về bất định |
| Xác nhận | PDF 10, Eqs. (11)–(12) | Chỉ xác nhận khi UAV ở vùng mở; dùng ngưỡng niềm tin/bất định; mục tiêu đã xác nhận bị loại và ô reset |
| Actor observation | PDF 12, Eq. (20) | Toàn bộ `π1,π0,πϕ`, toàn bộ `Φ`, vị trí mình, kênh `θ` |
| Thử nghiệm | PDF 16, Table 3 | `20×20`, 3 UAV, 15 mục tiêu, horizon 300, sensor range 4, retention 0.995 |

Không tìm thấy đặc tả SINR, CSMA/CA, ACK/retry, hàng đợi hữu hạn theo byte/gói, route discovery, forward từng hop, kênh bị che 3D hoặc GCS riêng trong toàn văn. Do đó có thể mô tả là **mô hình khả dụng liên lạc ở mức ứng dụng và hợp nhất**, không phải một mô phỏng đầy đủ các tầng vô tuyến.

**Điểm phải thận trọng:** Xue gọi observation là riêng cục bộ/CTDE, nhưng Eq. (20) vẫn đưa bản đồ toàn cục vào actor và không đặc tả protocol giữ bản sao stale riêng khi bị ngắt. Đây là giới hạn của mô hình thông tin được công bố; chưa có mã tác giả để kết luận chính xác hành vi runtime. Không được ngầm coi Xue là ví dụ đã chứng minh tính cục bộ thông tin của v52.

`θ` của Xue là thời gian ở vùng chặn, còn `t-ts` là tuổi một record tại lúc hợp nhất. Cả hai khác AoI liên tục tại GCS với cạnh tranh, retry và nhiều hop. Benchmark “Xue-style Ideal Sync” cần mô tả trung thực rằng chỉ mượn quy tắc khả dụng/đồng bộ, vì task, số UAV, reward và sensing của v52 đã khác bài.

Số cải thiện 26.13%, 24.32%, 3.65%, 31.96% trong abstract là kết quả của tác giả so baseline trong môi trường đó; không thể chuyển thành dự đoán cho v52. Ablation `w/o AoI` trên PDF 23 cho thấy phần belief chiếm phần lớn mức tăng; không nên gán toàn bộ cải thiện cho AoI.

## 4. Zeng và pin vật lý: đúng công thức, giới hạn áp dụng

Metadata bài journal xác nhận tại [IEEE](https://ieeexplore.ieee.org/abstract/document/8663615/) và [trang tác giả NUS](https://www.ece.nus.edu.sg/stfpage/elezhang/publication_UAV.html): Yong Zeng, Jie Xu, Rui Zhang, *IEEE TWC* 18(4), 2329–2345, 2019, DOI [10.1109/TWC.2019.2902559](https://doi.org/10.1109/TWC.2019.2902559).

Bản sơ cấp đọc được đầy đủ là [arXiv:1804.02238v1](https://arxiv.org/html/1804.02238v1), §II-B Eq. **(6)**, Appendix A Eqs. (61)–(66), trang PDF 27–29. Công thức trong design khớp biểu thức tốc độ ở bản này. Chưa lấy được toàn văn bản journal qua cổng IEEE, nên chưa xác nhận trực tiếp việc đánh số **Eq. (12)**; cần ghi rõ phiên bản khi viện dẫn.

§II-B nói rõ bỏ qua năng lượng tăng thêm do gia tốc, hợp lý khi thời gian thao tác chỉ chiếm phần nhỏ. Appendix giả định bay bằng, góc nghiêng nhỏ, thrust xấp xỉ trọng lượng. Vì vậy `P(||v||)` có thể dùng làm **baseline công suất theo tốc độ**, nhưng những quỹ đạo thường xuyên gia tốc 5 m/s² hoặc quay vòng cần kiểm tra thêm.

[Gao et al., arXiv:2005.01305v1](https://arxiv.org/html/2005.01305v1), §II-A trang 3 và §IV trang 6–8, xác định mô hình tốc độ chỉ áp dụng trực tuyến, độ cao không đổi, tốc độ đều; có mô hình tổng quát **heuristic** thêm gia tốc vuông góc và thay đổi động năng, cùng thí nghiệm vòng tròn. Điều này hỗ trợ việc báo cáo sai số/sensitivity do turning; không biến mô hình heuristic thành độ chính xác vật lý đã bảo đảm cho mọi UAV.

Các thời lượng `E/P` trong design là tính toán điều kiện theo bộ tham số được chọn. Chúng không xác thực pin, tải trọng, hiệu suất motor/ESC, gió, giới hạn SOC hay quỹ đạo thực. Bản arXiv gốc Table I dùng `W=100 N`, `U_tip=200 m/s`, `v0=7.2 m/s`, khác bộ `79.86/88.63/120/4.03` của design; không gọi bộ design là bảng gốc đã xác minh từ phiên bản này. `15 W` avionics và `3 W` PA/baseband cần khai báo là giả định cấu hình hoặc dựa trên phần cứng đo được.

Suy luận thiết kế: H=1500 có thể là safety cap hợp lý trong mô hình chọn, nhưng không phải kết luận phổ quát của Zeng. Phải đo tỷ lệ episode còn sống khi chạm cap; bất kỳ cải thiện “do mạng thực tế” cũng cần báo cả phân bố năng lượng/độ dài nhiệm vụ.

## 5. Pardo: deadline, truncation và local clock

Nguồn [PMLR ICML 2018](https://proceedings.mlr.press/v80/pardo18a.html), [PDF tác giả trên proceedings](https://proceedings.mlr.press/v80/pardo18a/pardo18a.pdf), §2 trang 3, Eq. (4), §3 trang 5–6, Eqs. (5)–(6).

| Mục tiêu thực sự | Actor/state | Target TD tại điểm dừng |
|---|---|---|
| Nhiệm vụ kết thúc theo deadline thật | Có thời gian còn lại hoặc cách suy ra nó | Terminal: `y=r` |
| Nhiệm vụ có thể tiếp tục, cap chỉ để thu thập/train | Cap không được biến thành terminal của nhiệm vụ | `y=r+γV(s_final)` |
| Hết năng lượng là trạng thái kết thúc của task được định nghĩa | SOC là state quan trọng; thời gian có thể vẫn hữu ích nếu dynamics phụ thuộc thời gian | Terminal khi task thực sự kết thúc |

Thêm đồng hồ flight-controller khả dụng khi triển khai không tự thân là leak. Pardo không bắt buộc một vector cụ thể `(H-t)/H`; bài chấp nhận thời gian hoặc khả năng suy ra. Thêm đồng hồ cũng không làm mọi Dec-POMDP trở thành MDP: mục tiêu ẩn, trạng thái queue/radio không biết và bản đồ peer vẫn có thể gây partial observability.

Việc bỏ time feature trong regime pin là lựa chọn dựa trên task không có deadline, không phải định lý “pin luôn phải thay đồng hồ”. Bootstrap phải lấy **final observation trước reset**, với mask terminal tách khỏi mask reset; terminal thật vẫn ưu tiên khi trùng cap.

## 6. Yates: AoI theo receiver, source và ô

Nguồn journal sơ cấp [PDF do đồng tác giả Ulukus lưu tại UMD](https://user.eng.umd.edu/~ulukus/papers/journal/aoi-survey.pdf), trang 1183–1185 (§I–II), trang 1196 Eq. (40); [MIT repository](https://dspace.mit.edu/entities/publication/ca039de7-e561-4a0f-bd3a-b92d0c436020), DOI [10.1109/JSAC.2021.3065072](https://doi.org/10.1109/JSAC.2021.3065072).

AoI chuẩn là `Δ(t)=t-u(t)`, với `u(t)` là timestamp sinh bản tin **mới nhất trong những bản tin receiver đã nhận**. Sau delivery, AoI bằng độ trễ của update đó, không tự động bằng 0.

**Áp dụng suy ra cho v52:** với mỗi ô `c`, định nghĩa `u_GCS,c(t)=max{gen_ts(m): m đã tới GCS trước t và chứa quan sát mới của c}`; `Δ_GCS,c=t-u_GCS,c`. Nếu cần freshness từng nguồn, giữ `u_GCS,origin,c` riêng. Timestamp của packet được tạo để relay không được thay timestamp phép đo; relay lại một count cũ không làm ô trẻ hơn. Packet nhiều ô phải có tuổi theo từng entry, không dùng timestamp mới nhất của một ô cho tất cả ô khác.

Ô chưa từng được nhận cần sentinel/cap rõ ràng và báo coverage cùng AoI; không loại chúng khỏi trung bình mà không công bố mẫu số. Độ phân giải giây có thể làm delay dưới một giây bị lượng tử thành 0; báo riêng delay/AoI theo event timestamp nếu dùng mô phỏng microsecond. Với mục tiêu tĩnh, cần nêu lý do nhiệm vụ coi thông tin cũ giảm giá trị; DS forgetting không phải hệ quả bắt buộc của định nghĩa AoI.

## 7. Wu và RAI-CoS: mức độ đã xác minh

### Wu TMC 2026

Tựa/DOI tìm được: *AoI-Aware Joint Sampling-Buffering-Routing Optimization for Autonomous UAV Swarms via a MARL Approach*, DOI [10.1109/TMC.2026.3699777](https://doi.org/10.1109/TMC.2026.3699777). Hồ sơ tác giả [ORCID](https://orcid.org/0000-0001-9272-0475) được search index liệt kê Haoxu Wu, Shaohua Wu, Aimin Li, Siqi Meng, Qinyu Zhang và mốc 2026-10; direct page cần JavaScript. [KTH DiVA được design viện dẫn](https://kth.diva-portal.org/smash/record.jsf?pid=diva2:2112470) trả access-denied; IEEE full text/abstract endpoint cũng bị chặn.

Abstract thứ cấp thống nhất về Leader–Follower, follower vừa sensing vừa relay, AASBR và COMH-MAPPO; nhưng đây **chưa phải kiểm chứng nội dung full text sơ cấp**. Không dùng sự thiếu toàn văn để khẳng định bài không có PHY/MAC, target uncertainty, NLoS, hay các thành phần khác. Chưa xác thực `25(10),18200–18217` bằng bản journal sơ cấp. Giữ vào danh mục gần nhất với trạng thái “metadata được ghi nhận; mô hình chi tiết chưa đối chiếu”, không gán một hàng “không có” dứt khoát.

### RAI-CoS, Computers 2026

Nguồn đầy đủ sơ cấp [MDPI](https://www.mdpi.com/2073-431X/15/10/711), DOI [10.3390/computers15100711](https://doi.org/10.3390/computers15100711), đăng 13-10-2026. Tác giả Md Tahmid Rashid, Md Saikat Ahmed, Md Jawad Siddique.

§3–4 đã có belief cục bộ, reliability theo ngữ cảnh, timestamp/clock uncertainty, IMM, delayed-observation replay theo thời điểm sinh, provenance và CI chống tương quan không rõ, receiver-specific information gain/byte, virtual queue và policy phân tán. Vì vậy chỉ nói “tracking+CI” bỏ sót nhiều điểm gần v52.

§5.1 mô tả SUMO qua TraCI và **replayed ns-3 packet traces** dùng chung giữa các phương pháp. §5.9/§6.8 đo đường xử lý decision stack bằng Jetson trace replay; §7–8 dành tích hợp airborne sensing/radio/flight stack và calibrated aircraft energy cho nghiên cứu tiếp. Có vòng kín estimation/control trong harness; chưa đủ cơ sở gọi đó là vòng kín vật lý MAC/SINR chịu ảnh hưởng tức thời của traffic và chuyển động của policy.

Khác biệt có thể kiểm chứng cho v52 là static hidden-target DS search, delivery về một GCS ứng dụng, và co-simulation mạng từng gói phụ thuộc quỹ đạo/traffic hiện hành. Đây là hướng đóng góp; chưa phải bằng chứng “đầu tiên” hay “tốt hơn” nếu chưa có benchmark.

## 8. UavNetSim và benchmark mạng

Nguồn sơ cấp: [repo chính thức](https://github.com/Zihao-Felix-Zhou/UavNetSim); [bài mô tả arXiv:2507.09852v1](https://arxiv.org/abs/2507.09852). Đã đọc toàn bộ bài 6 trang qua alphaXiv.

Bài §III trang 3–5 có routing, contention, ACK/retry, queueing và mô hình năng lượng; chủ đích là simulator Python/SimPy nhẹ. Bài bỏ modulation/coding, chỉ mô hình năng lượng RF bức xạ và propulsion tốc độ không đổi. Thử nghiệm routing giữa greedy/DSDV/OPAR không phải validation tương đương ns-3 hoặc thực nghiệm radio. Không có kết quả xác thực phần cứng/ns-3 trong bài này.

### Pin phiên bản đã kiểm tra

| Phiên bản | Commit/file | Kết quả |
|---|---|---|
| `v1.0.1` | `a5622c5d0a9f7b62b4e1d5a7dcc8978f34f0c85f` | Tag thực, đã đọc bằng GitHub connector |
| Cùng tag | [utils/config.py](https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/a5622c5d0a9f7b62b4e1d5a7dcc8978f34f0c85f/utils/config.py) | 0.1 W, noise 4e-11 W, 2 Mbps 802.11b config, SIFS/slot/DIFS, ACK/retry |
| Cùng tag | [phy/large_scale_fading.py](https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/a5622c5d0a9f7b62b4e1d5a7dcc8978f34f0c85f/phy/large_scale_fading.py) | SINR và path gain LoS; đây là file propagation hiện hữu |
| Cùng tag | `phy/a2a.py` | 404; cây Git không có file này |
| Cùng tag | [mac/csma_ca.py](https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/a5622c5d0a9f7b62b4e1d5a7dcc8978f34f0c85f/mac/csma_ca.py) | CSMA/CA giản lược không RTS/CTS; retry khi ACK timeout |
| `master` quan sát ngày kiểm tra | `c2703f24eb58cdb557a5f67ded5c7b0368a964ed` | Có module mới scene/Sionna/OSM và [phy/a2a.py](https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/c2703f24eb58cdb557a5f67ded5c7b0368a964ed/phy/a2a.py) |

`a2a.py` master có urban exponent 3, excess 12 dB, reference loss ở 1 m. Các giá trị đó là mô hình được code công bố, chưa có citation/đo urban air-to-air cụ thể ngay trong file. v52 nên ghi “borrowed log-distance parameters, sensitivity-tested”, không gọi đó là thông số thực nghiệm đô thị mặc định đã xác thực. Lưu ý tag config có `ROTOR_BLADE_TIP_SPEED=500` trong khi `ΩR=400×0.5=200`; không sao chép mù các hằng số năng lượng.

[ns-3 Wi-Fi design chính thức](https://www2.nsnam.org/docs/models/html/wifi-design.html) phân biệt MAC 802.11 và PHY abstraction/error model; `InterferenceHelper` xét SNIR theo các khoảng thời gian khi tín hiệu nhiễu thay đổi. Đây là đối chứng tốt để kiểm tra một port GPU, không đòi hỏi port phải mô phỏng RF waveform. [FlowMonitor docs](https://www2.nsnam.org/docs/models/html/flow-monitor.html) cũng cảnh báo packet còn trong queue lúc kết thúc và ranh giới đo IP/L2.

**Đề xuất benchmark suy ra từ các ranh giới trên:**

1. Công bố commit, băng tần/bitrate/header/CCA/SINR/ACK/retry/queue/routing, định nghĩa blackout và packet/evidence ID. Tách queue overflow, retry-limit, TTL, no-route, blackout.
2. Kiểm tra từng link/2-hop, hidden terminal, nhiễu bắt đầu giữa packet, ACK bị mất, half-duplex, duplicate, loop và chuyển động làm đổi liên kết. So distribution PDR/delay/AoI khi stochastic MAC khác RNG; chỉ đòi event equivalence khi engine chia sẻ đúng RNG/event semantics.
3. Chạy paired seeds với mobility/traffic giống nhau cho Ideal Sync và packet network; giữ thuật toán/reward hoặc công bố thay đổi. Báo TP/FP GCS, information coverage, AoI kể cả ô chưa nhận, delivery ratio, delay, airtime/control overhead, năng lượng và success trước cap.
4. So với một số cấu hình ns-3/trace chuẩn đã khóa phiên bản. Không lấy một episode scripted Full-Comm thắng Zero-Comm làm bằng chứng MARL đã học relay/store-carry-forward.
5. Đặt sự xuất hiện vai trò relay/store-carry-forward là giả thuyết cần đo. Hạ tải, đổi độ che và bitrate để xác nhận thắng lợi không chỉ do một link budget/packet budget đặc thù.

## 9. Nhãn nguồn và phần chưa xác minh

| Nguồn | Phân loại chắc chắn | Giới hạn |
|---|---|---|
| Zhang 2024; Xue 2026; RAI-CoS 2026 | Bài journal, toàn văn đã kiểm tra | Không suy ra chất lượng triển khai từ tên venue |
| Zeng 2019; Yates 2021 | Journal metadata xác thực; có bản tác giả mở | Số Eq. Zeng phải ghi đúng phiên bản |
| Pardo 2018 | ICML, proceedings PMLR80:4045–4054 | Chưa kiểm chứng CORE edition/rank |
| Gao 2020; UavNetSim 2025 | arXiv preprint phiên bản đã đọc | Không tự gắn nhãn journal peer-reviewed |
| Wu TMC 2026 | Metadata được ghi nhận từ index hồ sơ | Chưa đọc full text sơ cấp; không đủ để khẳng định absence |
| UavNetSim/ns-3 docs | Mã và tài liệu chính thức | Không thay thế validation riêng của v52 |

Không xác nhận Q1/Q2 từ một công cụ semantic-search hay dùng SJR thay JCR. Nếu giữ nhãn, cần nguồn tên hệ thống xếp hạng, năm dữ liệu và ngành; journal có thể có nhiều quartile theo ngành. Không có bằng chứng trong lần kiểm tra này rằng chỉ đổi pin/remaining_time hay thêm simulator sẽ đáp ứng tiêu chuẩn một venue cụ thể.

Kênh tra cứu: alphaXiv và nguồn chính thức hoạt động; Liner báo payment-required, Octen báo payment-required, Consensus hết quota tháng. Không retry các dịch vụ bị giới hạn, không truy cập tài khoản/credential, không gửi tin cho tác giả. Các giới hạn này không ảnh hưởng kết luận trực tiếp từ hai PDF đã có; phần Wu được giữ trạng thái chưa giải quyết.
