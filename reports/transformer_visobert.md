# ViSoBERT fine-tune (cấu hình A)

> Cấu hình A: không chuẩn hóa teencode, văn bản đưa thẳng vào tokenizer của ViSoBERT.
> Mọi số dưới đây đo thật trên Kaggle bằng `python -m src.train`, file gốc ở `reports/<cấu hình>/dev_metrics.json`.
> Chưa chạy test. Test chỉ chạy một lần sau khi đã chốt cấu hình bằng dev (CLAUDE.md mục 5).

## 1. Thiết lập

- Mô hình: `uitnlp/visobert` (kiến trúc XLM-RoBERTa), thêm đầu phân loại 3 nhãn khởi tạo ngẫu nhiên, fine-tune toàn bộ (D-015).
- Tokenizer: transformers 4.57.6 (bản 5.x tách token sai, D-016); `max_length = 128`, chọn từ phân bố độ dài trên train (D-017).
- Siêu tham số theo giá trị phổ biến, chưa dò: learning rate 2e-5, batch 32, 4 epoch, warmup 10%, weight decay 0.01, fp16.
- Phần cứng: 1 GPU Tesla T4 trên Kaggle (D-018), torch 2.11.0+cu128. Thời gian train khoảng 5 phút.
- Chọn epoch: đo dev cuối mỗi epoch, giữ epoch có macro-F1 dev cao nhất. Vì dev vừa dùng để chọn epoch vừa để báo cáo, điểm dev hơi lạc quan.

## 2. `visobert_base` (loss thường, seed 42), đo ngày 30/10/2026

Diễn biến theo epoch (đo trên dev):

| Epoch | Loss train | Loss dev | Macro-F1 dev | Accuracy dev |
|---|---|---|---|---|
| 1 | 0.372 | 0.304 | 0.692 | 0.884 |
| **2** | 0.203 | 0.348 | **0.709** | 0.888 |
| 3 | 0.100 | 0.442 | 0.700 | 0.887 |
| 4 | 0.052 | 0.545 | 0.705 | 0.887 |

Mô hình được giữ: epoch 2. Macro-F1 train 0.906, dev 0.709.

So sánh trên dev với baseline (`reports/baseline_tfidf.md`):

| Trên dev (2672 câu) | TF-IDF word+char | TF-IDF word+char, cân bằng nhãn | **ViSoBERT** |
|---|---|---|---|
| Macro-F1 | 0.611 | 0.633 | **0.709** |
| F1 CLEAN | 0.933 | 0.915 | **0.947** |
| F1 OFFENSIVE | 0.329 | 0.429 | **0.521** |
| F1 HATE | 0.570 | 0.555 | **0.660** |
| Câu CLEAN bị gán độc hại (/2190) | 58 | 264 | 93 |
| Câu OFFENSIVE bắt được (/212) | 47 | 99 | 91 |
| Câu HATE bắt được (/270) | 139 | 187 | 186 |

Confusion matrix của ViSoBERT trên dev (hàng = nhãn thật, cột = nhãn đoán; thứ tự CLEAN, OFFENSIVE, HATE):

| | CLEAN | OFFENSIVE | HATE |
|---|---|---|---|
| **CLEAN** | 2097 | 34 | 59 |
| **OFFENSIVE** | 72 | 91 | 49 |
| **HATE** | 72 | 12 | 186 |

## 3. Nhận xét

1. **Hơn baseline khoảng 7.6 điểm macro-F1 trên dev** (0.709 so với 0.633). Chưa phải con số cuối: một seed, và epoch được chọn theo dev.
2. **Không cần cân bằng nhãn vẫn bắt được gần bằng TF-IDF cân bằng nhãn** (91 so với 99 câu OFFENSIVE, 186 so với 187 câu HATE) nhưng gán oan ít hơn nhiều (93 so với 264 câu CLEAN). TF-IDF phải đổi precision lấy recall; ViSoBERT tăng được cả hai.
3. **OFFENSIVE vẫn là lớp khó nhất** (F1 0.521, recall 0.43): bị nhầm sang CLEAN (72) và HATE (49), cùng kiểu lỗi với baseline.
4. **Quá khớp sau epoch 1:** loss train giảm từ 0.37 xuống 0.05, loss dev tăng từ 0.30 lên 0.54, macro-F1 dev gần như đứng yên. Mô hình ngày càng tự tin thái quá kể cả khi sai. Ảnh hưởng tới ngưỡng độ tin cậy của `moderation.py` (hiệu chỉnh xác suất), cần xem lại ở tuần 7.
5. Trên train vẫn sai 186 + 253 câu OFFENSIVE (recall train 0.73): mô hình không học thuộc hết, một phần có thể do nhãn mâu thuẫn trong train (`data/README.md` mục 8).

## 4. Lưu ý

- Kaggle dùng scikit-learn 1.6.1 (máy: 1.9.1); chỉ dùng để tính chỉ số, không ảnh hưởng mô hình.
- Mô hình (`models/visobert_base/`) chỉ nằm trong phiên Kaggle, không commit.
- Việc tiếp theo: so sánh loss thường với loss cân bằng nhãn, chạy nhiều seed, rồi mới chạy test.
