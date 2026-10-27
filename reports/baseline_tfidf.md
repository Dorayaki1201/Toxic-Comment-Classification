# Baseline TF-IDF + Logistic Regression (cấu hình A)

> Cấu hình A: không chuẩn hóa teencode; tiền xử lý chỉ gồm chuẩn hóa Unicode NFC và viết thường (D-012).
> Mọi số dưới đây đo thật bằng `src/train.py` và `src/evaluate.py`, scikit-learn 1.9.1, `seed = 42`.
> Logistic Regression (solver mặc định) cho kết quả giống nhau mỗi lần chạy, nên chỉ có một lần chạy cho mỗi cấu hình.

## 1. Thiết lập

- Dữ liệu: ViHSD qua `load_vihsd`; train 24036 dòng (đã bỏ 12 dòng lỗi), dev 2672, test 6680 (D-004).
- Mô hình: `Pipeline` TF-IDF → `LogisticRegression` (`C = 1.0`, `max_iter = 1000`), từ điển và IDF chỉ học trên train (D-010).
- Đặc trưng: word n-gram 1–2 và/hoặc char n-gram 2–5 (`char_wb`), `min_df = 2`, `sublinear_tf = true`. Tham số chưa dò (D-011).
- Quy tắc chọn, đặt trước khi chạy: chọn cấu hình có macro-F1 dev cao nhất; chạy test đúng một lần cho cấu hình đó.

## 2. So sánh 6 cấu hình trên dev

| Cấu hình (`configs/`) | macro-F1 train | **macro-F1 dev** | F1 CLEAN | F1 OFFENSIVE | F1 HATE | Recall OFFENSIVE | Recall HATE |
|---|---|---|---|---|---|---|---|
| `tfidf_lr_word` | 0.697 | 0.560 | 0.929 | 0.219 | 0.532 | 0.132 | 0.452 |
| `tfidf_lr_word_balanced` | 0.873 | 0.608 | 0.904 | 0.383 | 0.538 | 0.415 | 0.707 |
| `tfidf_lr_char` | 0.679 | 0.582 | 0.929 | 0.259 | 0.559 | 0.160 | 0.493 |
| `tfidf_lr_char_balanced` | 0.782 | 0.619 | 0.896 | 0.431 | 0.530 | 0.542 | 0.689 |
| `tfidf_lr_wordchar` | 0.801 | 0.611 | 0.933 | 0.329 | 0.570 | 0.222 | 0.515 |
| **`tfidf_lr_wordchar_balanced`** | 0.886 | **0.633** | 0.915 | 0.429 | 0.555 | 0.467 | 0.693 |
| *Luôn đoán CLEAN* | | *0.300* | | | | | |

Chi tiết: `reports/<cấu hình>/dev_metrics.json`.

## 3. Kết quả test (chạy một lần, ngày 27/10/2026)

Cấu hình: `tfidf_lr_wordchar_balanced`. File: `reports/tfidf_lr_wordchar_balanced/test_metrics.json`.

| | Precision | Recall | F1 | Số mẫu |
|---|---|---|---|---|
| CLEAN | 0.953 | 0.886 | 0.918 | 5548 |
| OFFENSIVE | 0.390 | 0.491 | 0.435 | 444 |
| HATE | 0.495 | 0.690 | 0.577 | 688 |
| **macro-F1** | | | **0.643** | |
| accuracy | | | 0.840 | |

Confusion matrix (hàng = nhãn thật, cột = nhãn đoán; thứ tự CLEAN, OFFENSIVE, HATE):

| | CLEAN | OFFENSIVE | HATE |
|---|---|---|---|
| **CLEAN** | 4916 | 254 | 378 |
| **OFFENSIVE** | 119 | 218 | 107 |
| **HATE** | 126 | 87 | 475 |

Mốc so sánh trên test:

| Mô hình | macro-F1 test | Nguồn |
|---|---|---|
| Luôn đoán CLEAN | 0.303 | tính từ số nhãn test |
| mBERT (bài báo gốc) | 0.627 | Luu et al. (2021), số của bài báo |
| **TF-IDF + LR (dự án này)** | **0.643** | đo thật |

## 4. Nhận xét

1. **Cân bằng nhãn giúp ở cả 3 loại đặc trưng** (+2.3 đến +4.9 điểm macro-F1 dev), chủ yếu nhờ recall OFFENSIVE tăng từ 13–22% lên 42–54%. Đổi lại precision giảm: mô hình gán nhãn độc hại nhiều hơn, kể cả cho câu CLEAN (trên test, 632/5548 câu CLEAN bị gán nhãn độc hại). Đây là đánh đổi cần cân nhắc khi đặt ngưỡng cho `moderation.py`.
2. **Char n-gram tốt hơn word n-gram trên dev, ghép cả hai tốt nhất.** Dev là văn bản sạch; độ bền trước teencode sẽ đo bằng bộ test nhiễu (tuần 4).
3. **OFFENSIVE là lớp khó nhất** (F1 test 0.435): hay bị nhầm với HATE và CLEAN. Phù hợp với việc phân biệt hai lớp này dựa vào việc có đối tượng bị nhắm tới hay không, điều mà túi từ (bag-of-words) khó nắm bắt.
4. **Trọng số lớn nhất khớp với phân tích từ khóa** (`notebooks/02_keywords_vihsd.ipynb`): OFFENSIVE dựa vào từ chửi viết tắt; HATE dựa vào đại từ coi thường, so sánh với con vật, và từ viết tắt chỉ nhóm chính trị (rủi ro thiên lệch cần kiểm tra ở tuần 6).
5. **Char n-gram dùng được emoji** (emoji cười nằm trong trọng số lớn của CLEAN), thông tin mà `keywords.py` đã bỏ qua. Có dấu hiệu tương quan giả (cụm từ là tên chủ đề cụ thể trong top CLEAN) và dạng không dấu của cụm từ công kích ở OFFENSIVE (ủng hộ lớp khôi phục dấu ở tuần 3).

## 5. Lưu ý khi đọc kết quả

- **Baseline vượt mBERT của bài báo 1.6 điểm.** Không kết luận TF-IDF tốt hơn mBERT: (a) 11.9% câu test trùng với train (D-001), TF-IDF ghi nhớ tốt những câu này; (b) dự án dùng cân bằng nhãn, bài báo có thể không; (c) khác tiền xử lý và cách huấn luyện. Cần đo thêm macro-F1 trên phần test **không** trùng với train (TODO trong `data/README.md` mục 8).
- Chênh lệch giữa các cấu hình đầu bảng trên dev nhỏ (0.633 so với 0.619) so với kích thước dev (212 câu OFFENSIVE); chưa đo khoảng tin cậy.
- Dev cũng có 12% câu trùng với train; ảnh hưởng như nhau với cả 6 cấu hình nên phép so sánh vẫn công bằng.
