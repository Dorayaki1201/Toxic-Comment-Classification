# Phát hiện ngôn từ độc hại trong bình luận tiếng Việt

> ⚠️ **Cảnh báo nội dung:** dự án làm việc với dữ liệu chứa ngôn từ thô tục và thù ghét.

Hệ thống phân loại bình luận tiếng Việt thành 3 nhãn `CLEAN` / `OFFENSIVE` / `HATE`, có khả năng chuẩn hóa viết tắt, teencode, ký tự chèn, không dấu, ký tự kéo dài. Dự án đo bằng số liệu mức độ bền vững của mô hình trước các biến thể lách kiểm duyệt đó.

## Trạng thái

🚧 Đang phát triển. Đã xong: baseline, ViSoBERT, chuẩn hóa bằng luật và từ điển teencode (chưa nối vào huấn luyện). Tiếp theo: sinh nhiễu và bộ test nhiễu (tuần 4).

## Chuẩn hóa teencode (cấu hình B)

Hai lớp, đều chỉ xây và đo trên tập train:

- **Lớp luật:** số thay chữ, ký tự chèn giữa chữ, ký tự kéo dài. Đổi 2.33% câu train.
- **Từ điển teencode/viết tắt** ([resources/teencode.tsv](resources/teencode.tsv)): 83 mục, chọn từ 167 ứng viên bằng cách đọc 10 ngữ cảnh mỗi ứng viên và chỉ giữ token có ít nhất 9/10 ngữ cảnh cùng nghĩa. Đổi 23.6% câu train; precision **0.98** (Wilson 95%: 0.93–0.99) trên 100 lần thay lấy ngẫu nhiên, đo trên mẫu khác với mẫu dùng để chọn.

Nguyên tắc: thà bỏ sót còn hơn sửa sai, vì thay sai làm đổi nghĩa câu. Quy trình và hạn chế: [docs/decisions.md](docs/decisions.md) (D-021, D-022).

## Kết quả

| Mô hình | Cấu hình | Macro-F1 (test gốc) | Macro-F1 (test gốc, phần chưa gặp trong train) | Macro-F1 (test nhiễu) |
|---|---|---|---|---|
| TF-IDF (word + char) + LR, cân bằng nhãn | A | 0.643 (1 lần chạy) | 0.641 | TODO |
| ViSoBERT, cân bằng nhãn | A | **0.694 ± 0.004** (3 seed) | 0.695 ± 0.002 | TODO |

Chi tiết trong `reports/` (baseline: [reports/baseline_tfidf.md](reports/baseline_tfidf.md), ViSoBERT: [reports/transformer_visobert.md](reports/transformer_visobert.md)). Cấu hình được chọn trên dev theo quy tắc đặt trước; test chỉ chạy một lần cho cấu hình đã chọn. Tham chiếu: mBERT trong bài báo gốc đạt macro-F1 0.627 trên test (khác cách huấn luyện, nên chỉ để tham khảo). Khoảng 12% câu test trùng với train, nhưng điểm trên phần chưa gặp gần như bằng điểm toàn bộ test ở cả hai mô hình.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
docs/        kiến trúc hệ thống và sổ quyết định kỹ thuật
data/        hướng dẫn tải dữ liệu (không commit dữ liệu)
notebooks/   khám phá dữ liệu, thử nghiệm
src/         mã nguồn chính
configs/     cấu hình thí nghiệm (yaml)
resources/   tài nguyên tự tuyển chọn (từ điển teencode)
app/         demo Gradio
tests/       kiểm thử đơn vị
reports/     bảng kết quả, biểu đồ, phân tích lỗi
```

Sơ đồ luồng dữ liệu và trạng thái từng module: [docs/architecture.md](docs/architecture.md). Lý do của các lựa chọn kỹ thuật: [docs/decisions.md](docs/decisions.md).

## Đạo đức và giới hạn

TODO: hoàn thiện ở tuần 7 (xem mục 13 trong `CLAUDE.md`).
