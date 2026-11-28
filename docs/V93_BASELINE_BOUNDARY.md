# Ranh giới kịch bản và baseline — chưa kiểm toán thuật toán

Môi trường nhận hai action XY liên tục và một lựa chọn truyền tin compact-3. Baseline continuous thuần không đồng thời học lựa chọn rời rạc đó nếu không thay đổi thuật toán hoặc cố định phần truyền tin.

Source hiện đã có một đường đi hợp lý cho baseline continuous: `uav_network/algorithms/baselines/canonical_continuous.py` đặt `act_dim=2`, critic nhận joint XY, và hàm `event_driven_comm_action_np/torch` chọn GCS/PEER/SILENT theo observation và mask cục bộ. Do đó cần mô tả thí nghiệm là **học chuyển động, dùng cùng quy tắc truyền tin cố định**. Không được mô tả thành baseline thuần đã học toàn bộ joint motion+communication.

Các hàm này là code sẵn có; V93 không sửa chúng. `vanilla.py` có Gumbel-Softmax và nhánh discrete; tên “vanilla” không tự chứng minh đó là thuật toán continuous nguyên bản. Đây là phân loại giao diện, chưa kết luận đúng/sai về Bellman target, gradient, entropy hoặc update schedule.

Trước pha baseline, khóa một protocol công bằng: mọi thuật toán continuous cùng observation/mask, cùng quy tắc truyền tin, cùng action XY, cùng safety shield, termination và reward. Nếu mục tiêu nghiên cứu sau này là học lựa chọn truyền tin, cần ghi đó là bài toán/action space khác và đánh giá extension riêng. Không xóa PEER chỉ vì một fixture cũ sai; fixture mới đã chứng minh vai trò khác biệt của dissemination khi contact tách rời.

V93 chưa xác nhận đầy đủ tính “gốc” của bất kỳ MASAC/MATD3 có sẵn. Thuật toán vẫn để nguyên cho pha sau theo yêu cầu. Đẳng thức SHA của30 file chỉ chứng minh không thay đổi trong phiên này.
