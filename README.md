# Phát hiện ngôn từ độc hại trong bình luận tiếng Việt

> ⚠️ **Cảnh báo nội dung:** dự án làm việc với dữ liệu chứa ngôn từ thô tục và thù ghét.

Hệ thống phân loại bình luận tiếng Việt thành 3 nhãn `CLEAN` / `OFFENSIVE` / `HATE`, có khả năng chuẩn hóa viết tắt, teencode, ký tự chèn, không dấu, ký tự kéo dài. Dự án đo bằng số liệu mức độ bền vững của mô hình trước các biến thể lách kiểm duyệt đó.

## Trạng thái

🚧 Đang phát triển (tuần 1: dữ liệu và baseline).

## Kết quả

| Mô hình | Cấu hình | Macro-F1 (test gốc) | Macro-F1 (test nhiễu) |
|---|---|---|---|
| TF-IDF + LR | A | TODO | TODO |
| ViSoBERT | A | TODO | TODO |

Chi tiết trong `reports/`.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
data/        hướng dẫn tải dữ liệu (không commit dữ liệu)
notebooks/   khám phá dữ liệu, thử nghiệm
src/         mã nguồn chính
configs/     cấu hình thí nghiệm (yaml)
app/         demo Gradio
tests/       kiểm thử đơn vị
reports/     bảng kết quả, biểu đồ, phân tích lỗi
```

## Đạo đức và giới hạn

TODO: hoàn thiện ở tuần 7 (xem mục 13 trong `CLAUDE.md`).
