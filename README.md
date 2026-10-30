# Phát hiện ngôn từ độc hại trong bình luận tiếng Việt

> ⚠️ **Cảnh báo nội dung:** dự án làm việc với dữ liệu chứa ngôn từ thô tục và thù ghét.

Hệ thống phân loại bình luận tiếng Việt thành 3 nhãn `CLEAN` / `OFFENSIVE` / `HATE`, có khả năng chuẩn hóa viết tắt, teencode, ký tự chèn, không dấu, ký tự kéo dài. Dự án đo bằng số liệu mức độ bền vững của mô hình trước các biến thể lách kiểm duyệt đó.

## Trạng thái

🚧 Đang phát triển (tuần 2: mô hình transformer).

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
app/         demo Gradio
tests/       kiểm thử đơn vị
reports/     bảng kết quả, biểu đồ, phân tích lỗi
```

Sơ đồ luồng dữ liệu và trạng thái từng module: [docs/architecture.md](docs/architecture.md). Lý do của các lựa chọn kỹ thuật: [docs/decisions.md](docs/decisions.md).

## Đạo đức và giới hạn

TODO: hoàn thiện ở tuần 7 (xem mục 13 trong `CLAUDE.md`).
