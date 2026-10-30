# ViSoBERT fine-tune (cấu hình A)

> Cấu hình A: không chuẩn hóa teencode, văn bản đưa thẳng vào tokenizer của ViSoBERT.
> Mọi số dưới đây đo thật trên Kaggle bằng `python -m src.train`, file gốc ở `reports/<cấu hình>/dev_metrics.json`.
> Test chỉ chạy một lần, cho cấu hình đã được chọn trên dev theo quy tắc đặt trước (D-019, mục 5 dưới đây).

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

## 4. Loss thường và loss cân bằng nhãn, 3 seed trên dev (D-019)

Quy tắc chọn được ghi trước khi chạy: trung bình macro-F1 dev cao hơn; nếu chênh lệch nhỏ hơn độ lệch chuẩn lớn hơn thì coi như hòa và chọn `visobert_base`. Độ lệch chuẩn là độ lệch chuẩn mẫu (ddof=1).

| Lần chạy | Macro-F1 dev | F1 OFFENSIVE | F1 HATE | Recall OFFENSIVE | Câu CLEAN bị gán độc hại (/2190) |
|---|---|---|---|---|---|
| `visobert_base_s42` | 0.7092 | 0.521 | 0.660 | 0.429 | 93 |
| `visobert_base_s43` | 0.7174 | 0.540 | 0.664 | 0.495 | 92 |
| `visobert_base_s44` | 0.7097 | 0.528 | 0.657 | 0.472 | 101 |
| `visobert_balanced_s42` | 0.7251 | 0.584 | 0.652 | 0.590 | 184 |
| `visobert_balanced_s43` | 0.7268 | 0.559 | 0.673 | 0.524 | 109 |
| `visobert_balanced_s44` | 0.7165 | 0.543 | 0.661 | 0.505 | 116 |

| Cấu hình | Macro-F1 dev | F1 CLEAN | F1 OFFENSIVE | F1 HATE |
|---|---|---|---|---|
| `visobert_base` | 0.7121 ± 0.0046 | 0.9465 ± 0.0018 | 0.5297 ± 0.0093 | 0.6602 ± 0.0037 |
| `visobert_balanced` | **0.7228 ± 0.0055** | 0.9444 ± 0.0048 | **0.5622 ± 0.0206** | 0.6618 ± 0.0103 |

Chênh lệch 0.0107 lớn hơn 0.0055, nên chọn **`visobert_balanced`**. `visobert_base_s42` chạy lại cho đúng từng chữ số như lần chạy đầu (mục 2): kết quả tái lập được trên cùng GPU và cùng phiên bản thư viện.

## 5. Kết quả test của `visobert_balanced` (chạy một lần, ngày 30/10/2026)

| Seed | Macro-F1 test (6680) | Đã gặp trong train (805) | Chưa gặp (5875) | Câu CLEAN bị gán độc hại (/5548) |
|---|---|---|---|---|
| 42 | 0.6965 | 0.6947 | 0.6958 | 512 |
| 43 | 0.6900 | 0.6636 | 0.6917 | 320 |
| 44 | 0.6954 | 0.6816 | 0.6960 | 326 |
| **Trung bình ± độ lệch chuẩn** | **0.6940 ± 0.0035** | | **0.6945 ± 0.0024** | |

F1 từng lớp (trung bình ± độ lệch chuẩn): CLEAN 0.940 ± 0.005, OFFENSIVE 0.490 ± 0.018, HATE 0.652 ± 0.005.
Precision / recall trung bình: CLEAN 0.950 / 0.930, OFFENSIVE 0.493 / 0.489, HATE 0.610 / 0.706.

So sánh với baseline trên test (confusion matrix của ViSoBERT là trung bình 3 seed; hàng = nhãn thật, cột = nhãn đoán):

| | TF-IDF + LR, cân bằng (1 lần chạy) | ViSoBERT, cân bằng (trung bình 3 seed) |
|---|---|---|
| Macro-F1 | 0.643 | **0.694 ± 0.004** |
| Macro-F1 phần chưa gặp | 0.641 | **0.695 ± 0.002** |
| Câu CLEAN bị gán độc hại (/5548) | 632 | **386** |
| Câu OFFENSIVE bắt đúng (/444) | 218 | 217 |
| Câu HATE bắt đúng (/688) | 475 | 486 |

| ViSoBERT (trung bình) | CLEAN | OFFENSIVE | HATE |
|---|---|---|---|
| **CLEAN** | 5162 | 160 | 226 |
| **OFFENSIVE** | 136 | 217 | 91 |
| **HATE** | 138 | 64 | 486 |

Mốc tham khảo: luôn đoán CLEAN 0.303; mBERT trong bài báo gốc 0.627 (Luu et al., 2021; khác cách huấn luyện và không có số trên phần chưa gặp).

## 6. Nhận xét sau khi có test

1. **ViSoBERT hơn baseline khoảng 5 điểm macro-F1 trên test** (0.694 so với 0.643), và khoảng 5.4 điểm trên phần chưa gặp. Khoảng cách lớn hơn nhiều lần độ lệch chuẩn giữa các seed.
2. **Phần lợi chủ yếu là gán oan ít hơn, không phải bắt được nhiều hơn:** số câu OFFENSIVE và HATE bắt đúng gần như bằng baseline, nhưng số câu CLEAN bị gán độc hại giảm từ 632 xuống khoảng 386.
3. **Điểm test thấp hơn dev khoảng 2.9 điểm** (0.723 so với 0.694), tụt nhiều nhất ở OFFENSIVE (0.562 so với 0.490). Dev đã được dùng hai lần để chọn (epoch tốt nhất, rồi giữa hai cấu hình) nên lạc quan; dev chỉ có 212 câu OFFENSIVE nên F1 lớp này dao động mạnh. Con số để báo cáo là test.
4. **Không có dấu hiệu học vẹt câu trùng:** phần chưa gặp (0.695) gần bằng toàn bộ test (0.694). Phần đã gặp thấp hơn nhưng chỉ khoảng 40 câu mỗi lớp độc hại nên không đọc sâu.
5. **Ngưỡng hoạt động thay đổi theo seed:** macro-F1 gần như nhau nhưng seed 42 gán oan 512 câu CLEAN trên test, hai seed còn lại khoảng 320. Seed 42 giữ epoch 2 (ít huấn luyện hơn), cũng là seed gán oan nhiều nhất trên dev (184). Với `moderation.py`, cần đặt ngưỡng xác suất trên dev cho đúng mô hình được triển khai, không dùng chung một ngưỡng cho mọi lần train.
6. **OFFENSIVE vẫn là lớp yếu nhất** (F1 0.49): hay nhầm sang CLEAN (136) và HATE (91). Phân tích lỗi ở tuần 6 nên tập trung vào lớp này.

## 7. Lưu ý

- Kaggle dùng scikit-learn 1.6.1 (máy: 1.9.1); chỉ dùng để tính chỉ số, không ảnh hưởng mô hình.
- Mô hình chỉ nằm trong phiên Kaggle, không commit; train lại cho ra cùng kết quả (mục 4).
- Siêu tham số chưa dò; 3 seed vẫn ít, độ lệch chuẩn còn là ước lượng thô.
- Báo cáo `visobert_smoke` (chạy thử 500 câu) không được lưu vì không dùng để so sánh.
