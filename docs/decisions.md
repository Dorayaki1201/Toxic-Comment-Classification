# Sổ quyết định kỹ thuật

Mỗi quyết định ghi: **bối cảnh** (vấn đề là gì), **quyết định** (chọn gì), **lý do**, **đánh đổi** (mất gì, phương án đã bỏ). Không xóa quyết định cũ; nếu đổi ý thì thêm quyết định mới và ghi "thay thế D-xxx".

Định dạng này dựa trên *Architecture Decision Records* (Michael Nygard, 2011): <https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions>.

---

## D-001. Giữ nguyên split train/dev/test gốc của ViHSD

- **Ngày:** 04/10/2026
- **Bối cảnh:** khi khám phá dữ liệu thấy 797/6680 dòng test (11.9%) trùng với train, 725 trong số đó có cùng nhãn. Mô hình chỉ cần ghi nhớ là đoán đúng, nên điểm test có thể cao hơn năng lực thật.
- **Quyết định:** giữ nguyên split gốc.
- **Lý do:** để so sánh được với kết quả của bài báo gốc và các nghiên cứu khác dùng ViHSD.
- **Đánh đổi:** điểm test có thể bị thổi phồng. Bù lại bằng cách báo cáo thêm điểm trên phần test không trùng với train (TODO, xem `data/README.md` mục 8). Phương án bỏ: tự chia lại không trùng, vì mất khả năng so sánh.

## D-002. Macro-F1 là chỉ số chính, không dùng accuracy

- **Ngày:** từ đầu dự án (CLAUDE.md mục 5); số liệu xác nhận ngày 04/10/2026
- **Bối cảnh:** CLEAN chiếm khoảng 83% dữ liệu. Mô hình luôn đoán CLEAN đã đạt accuracy 83.1% trên test.
- **Quyết định:** chỉ số chính là macro-F1, luôn báo cáo kèm F1 từng lớp.
- **Lý do:** macro-F1 cho mỗi lớp trọng số bằng nhau, nên mô hình bỏ qua OFFENSIVE/HATE sẽ bị phạt nặng.
- **Đánh đổi:** macro-F1 dao động mạnh hơn khi lớp nhỏ có ít mẫu, nên cần chạy nhiều seed.

## D-003. Đọc CSV với `keep_default_na=False`

- **Ngày:** 06/10/2026
- **Bối cảnh:** mặc định `pd.read_csv` biến cả chữ `"NA"`, `"null"`, `"None"`... thành NaN. Với văn bản tự do, một bình luận đúng là chữ `"null"` sẽ bị mất.
- **Quyết định:** đọc với `keep_default_na=False`, ô trống thành `""`, rồi tự quyết định thế nào là dòng lỗi.
- **Lý do:** đọc nguyên trạng, làm sạch tường minh trong code. Đo trên ViHSD thì hai cách cho cùng kết quả (2 dòng NaN ở train đều là ô trống thật), nhưng cùng hàm đọc sẽ dùng cho ViCTSD và câu người dùng gõ vào demo.
- **Đánh đổi:** không có NaN nên phải kiểm tra chuỗi rỗng thay vì dùng `isna()`.

## D-004. Dòng lỗi: bỏ ở train, giữ ở dev/test

- **Ngày:** 06/10/2026
- **Bối cảnh:** đo bằng code: train có 2 dòng rỗng + 10 dòng `#ERROR!`, dev có 3 dòng `#ERROR!`, test có 4 dòng `#ERROR!`. Không có biến thể khác.
- **Quyết định:** dòng lỗi = text sau khi bỏ khoảng trắng hai đầu là rỗng hoặc đúng bằng `#ERROR!`. Train bỏ các dòng này (còn 24036). Dev và test giữ đủ dòng (2672, 6680), thay text lỗi bằng `""`.
- **Lý do:** train sạch hơn để học. Dev/test giữ đúng số dòng gốc để so sánh được với bài báo (D-001) và không tự ý chỉnh tập đánh giá.
- **Đánh đổi:** 7 dòng rỗng ở dev/test vẫn mang nhãn gốc; dự đoán trên chuỗi rỗng không có ý nghĩa, nên nếu nhãn gốc là độc hại thì gần như chắc chắn bị tính sai. Ảnh hưởng rất nhỏ (3/2672 dòng dev, 4/6680 dòng test). Nhãn gốc của 7 dòng này chưa được thống kê. Phương án bỏ: bỏ ở cả 3 tập, vì lệch số với bài báo.

## D-005. `data.py` không sửa nội dung bình luận

- **Ngày:** 06/10/2026
- **Bối cảnh:** có thể tiện tay viết thường, bỏ khoảng trắng thừa ngay lúc nạp dữ liệu.
- **Quyết định:** `data.py` chỉ đảm bảo dữ liệu hợp lệ. Mọi biến đổi nội dung thuộc về `normalize.py`.
- **Lý do:** dự án cần so sánh cấu hình A (không chuẩn hóa) với B (có chuẩn hóa). Nếu `data.py` đã sửa chữ thì cấu hình A không còn là "không chuẩn hóa" thật.
- **Đánh đổi:** mọi bước sau phải gọi chuẩn hóa một cách tường minh.

## D-006. Test không đọc dữ liệu thật

- **Ngày:** 16/10/2026
- **Bối cảnh:** dữ liệu không được commit (giấy phép, nội dung nhạy cảm), nên không có trên máy người clone repo.
- **Quyết định:** test dùng bảng hoặc CSV nhỏ tự tạo trong thư mục tạm của pytest, chỉ chứa câu trung tính. Giá trị cần kiểm tra (như `"#ERROR!"`) được gõ thẳng trong test, không import hằng số từ code.
- **Lý do:** test chạy được ở mọi nơi. Gõ thẳng giá trị giúp bắt lỗi khi chính hằng số trong code bị sai.
- **Đánh đổi:** test không bắt được thay đổi của dữ liệu thật. Phần đó được kiểm tra bằng log số dòng khi nạp (24036 / 2672 / 6680).
