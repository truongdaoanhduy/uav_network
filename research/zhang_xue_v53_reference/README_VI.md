# Bản hiệu chỉnh phép tính tham chiếu v5.2

Đây là bản sửa tách riêng để kiểm tra công thức và chẩn đoán liên thông 6 UAV + 1 GCS. Bản này dùng đồ thị toàn cục và đọc trạng thái bên nhận để quyết định entry nào cần chuyển. Nó chưa phải môi trường UavNetSim/Warp, chưa có mô phỏng gói tin CSMA/CA, giao thoa, ACK mất, retry, queue, TTL, điều khiển path-vector, quan sát actor hay huấn luyện MARL. Không dùng kết quả của nó làm kết quả mạng production.

Các sửa đổi có kiểm thử hồi quy:

- DS dùng công thức đóng dạng log có chuẩn hóa ổn định; kiểm tra BBA và bộ đếm; tổng xung đột được báo lỗi rõ ràng.
- Công suất đẩy được tính từ độ dịch chuyển thực của từng UAV trong bước. Năng lượng phát data gán cho UAV gửi, ACK gán cho UAV nhận khi UAV đó phát ACK; ACK của GCS không trừ vào pin UAV.
- Timestamp mới vẫn được chuyển nếu bộ đếm bằng nhau, để không bỏ mất độ mới khi sử dụng bộ đếm bão hòa trong tương lai. Trong kịch bản hiện tại 1.500 bước, bộ đếm chưa đạt 65.535; phản ví dụ này là kiểm thử biên được tiêm vào trạng thái chạy.
- Đổi `mean_gcs_aoi_s` thành `mean_first_tp_confirmation_aoi_s`; thêm `mean_gcs_aoi_over_delivered_cells_s`, tính trung bình theo thời gian và ô sau lần nhận đầu tiên. Các ô chưa từng nhận bị loại khỏi mẫu số; `-1` biểu thị chưa có mẫu.
- Kết quả gắn `simulation_semantics=oracle_connectivity_diagnostic_no_packet_mac`.

Chạy từ thư mục `corrected`:

```bash
python3 -m unittest discover -s tests -v
```

Kết quả kiểm tra: **21/21 test PASS**, gồm 12 test gốc và 9 test hồi quy. Chín test hồi quy được chạy trước trên mã gốc và đều thất bại theo hành vi cần sửa; một số test có nhiều subtest nên log gốc báo 18 failure.

Với seed `20261125`, 300 bước, Full-Comm vẫn có tổng thưởng `769,97` và `11/15` mục tiêu được xác nhận trong chẩn đoán đồ thị. Tuổi thông tin ở lần xác nhận đầu tiên là `0,00 s`, trong khi trung bình receiver theo thời gian và ô đã được nhận là **`51,23 s`**. Pin cạn ở bước **1171**, thay cho 1173 khi giả định tốc độ không đổi 15 m/s. Đây vẫn chỉ là kết quả của quỹ đạo kịch bản và mô hình bay ổn định.

Các giới hạn chưa được sửa: quỹ đạo có đổi hướng tức thì và chưa ràng buộc gia tốc; năng lượng tăng tốc/giảm tốc và RX chưa được mô hình hóa riêng; sự thành công của mỗi lần truyền là lý tưởng theo đồ thị; việc chọn gói phụ thuộc thứ tự ID; hàm ước lượng VRAM chưa phải bản kê đầy đủ tensor GPU. Chưa xác nhận T4 fit, chuẩn IEEE 802.11, quá tải queue, clock/reset CUDA hay bất cứ chất lượng chính sách MARL nào.
