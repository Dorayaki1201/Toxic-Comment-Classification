# Cấu hình A trên bộ test nhiễu (D-026)

> Cấu hình A: không chuẩn hóa, văn bản đưa thẳng vào mô hình. Mô hình đã đóng băng, không chỉnh gì theo kết quả ở đây.
> Bộ nhiễu: 9 bộ + đối chứng `nfc_only` từ ViHSD test (6680 câu), seed nhiễu 1105 (D-025, `manifest.json`).
> File gốc: `reports/tfidf_lr_wordchar_balanced/noisy_test_metrics.json`, `reports/visobert_balanced/noisy_test_summary.json` (và `visobert_balanced_s4x/noisy_test_metrics.json`).
> Lệnh: `python -m src.evaluate --config <config> --noisy-test [--seed S]`; ViSoBERT chạy trên Kaggle (`docs/kaggle.md` mục 5).

## 1. Mô hình được đo

| Mô hình | Lần chạy | Tham chiếu để tính mức tụt | Điểm tham chiếu |
|---|---|---|---|
| `tfidf_lr_wordchar_balanced` | máy cá nhân, 06/11/2026; tất định (D-013) | `original` (baseline tự NFC) | 0.6431, khớp `test_metrics.json` |
| `visobert_balanced` seed 42/43/44 | Kaggle, 07/11/2026, Tesla T4, torch 2.11.0+cu128, transformers 4.57.6; train lại vì mô hình cũ không còn | `nfc_only` (transformer không tự NFC) | 0.6951 ± 0.0037 |

**Tái tạo ViSoBERT (3 cổng của D-026):**

- Cổng 1 (môi trường): khớp phiên bản và GPU của lần train cũ.
- Cổng 2 (dev): s43 và s44 cho `dev_metrics.json` giống hệt từng byte; **s42 lệch** đúng 1 câu train (một câu CLEAN chuyển sang bị đoán HATE) và 1 câu dev (một câu CLEAN trước bị đoán HATE, nay đoán đúng). Macro-F1 dev s42: 0.7251 (cũ) → 0.7256 (mới).
- Cổng 3 (test gốc): s43 0.6900 và s44 0.6954 khớp `test_metrics.json`; s42 0.6965 (cũ) → 0.6962 (mới).
- Nguyên nhân có thể là phép tính không tất định trên GPU (chưa bật `torch.use_deterministic_algorithms`); chưa kiểm chứng. Lần chạy lại `visobert_base_s42` ngày 30/10 thì khớp, nên lệch chỉ xảy ra thỉnh thoảng.
- Theo quy tắc ghi trước ở D-026: dùng 3 mô hình mới, giữ nguyên các file dev/test cũ. Test gốc của bộ mới: **0.6939 ± 0.0034** (phần chưa gặp 0.6944 ± 0.0023), so với bộ cũ 0.6940 ± 0.0035: giống nhau tới chữ số thứ 3.

## 2. Kết quả

Mức tụt = điểm tham chiếu − điểm bộ nhiễu (dương = tụt), tính bằng điểm macro-F1 tuyệt đối. ViSoBERT: trung bình ± độ lệch chuẩn mẫu qua 3 seed, mức tụt tính trong từng seed rồi mới gộp. "Trên câu bị đổi" = chỉ những câu mà nhiễu thực sự thay đổi, so với chính các câu đó ở bộ tham chiếu.

| Bộ | % từ đổi | Baseline macro-F1 | Baseline tụt | ViSoBERT macro-F1 | ViSoBERT tụt | ViSoBERT tụt trên câu bị đổi |
|---|---|---|---|---|---|---|
| tham chiếu | 0 | 0.6431 | — | 0.6951 ± 0.0037 | — | — |
| abbreviate_medium | 8.69 | 0.6403 | +0.003 | 0.6856 ± 0.0027 | +0.010 ± 0.001 | +0.016 |
| teencode_chars_medium | 7.73 | 0.6451 | −0.002 | 0.6806 ± 0.0055 | +0.015 ± 0.006 | +0.025 |
| remove_diacritics_medium | 19.93 | 0.6184 | +0.025 | 0.6777 ± 0.0057 | +0.017 ± 0.004 | +0.021 |
| elongate_medium | 25.95 | 0.6371 | +0.006 | 0.6370 ± 0.0106 | +0.058 ± 0.012 | +0.064 |
| substitute_digits_medium | 23.93 | 0.6228 | +0.020 | 0.6376 ± 0.0067 | +0.058 ± 0.007 | +0.064 |
| insert_chars_medium | 26.01 | 0.5782 | +0.065 | 0.4852 ± 0.0144 | +0.210 ± 0.017 | +0.229 |
| all_light | 8.57 | 0.6303 | +0.013 | 0.6520 ± 0.0088 | +0.043 ± 0.010 | +0.067 |
| all_medium | 26.68 | 0.6062 | +0.037 | 0.5944 ± 0.0114 | +0.101 ± 0.012 | +0.109 |
| all_heavy | 45.81 | 0.5757 | +0.067 | 0.5135 ± 0.0058 | +0.182 ± 0.007 | +0.185 |

F1 từng lớp (ViSoBERT: trung bình 3 seed):

| Bộ | Baseline OFFENSIVE | Baseline HATE | ViSoBERT OFFENSIVE | ViSoBERT HATE |
|---|---|---|---|---|
| tham chiếu | 0.435 | 0.576 | 0.494 | 0.652 |
| insert_chars_medium | 0.347 | 0.501 | 0.241 | 0.434 |
| all_medium | 0.391 | 0.519 | 0.353 | 0.541 |
| all_heavy | 0.351 | 0.475 | 0.271 | 0.418 |

Tỉ lệ đổi nhãn dự đoán so với tham chiếu (all_light / all_medium / all_heavy): baseline 4.0% / 9.7% / 14.3%; ViSoBERT 5.8% / 15.0% / 23.0%. Phần chưa gặp trong train cho điểm gần như trùng toàn bộ tập ở mọi bộ (chênh dưới 0.004), nên câu trùng train không làm sai lệch kết luận.

Đối chứng NFC với ViSoBERT: test gốc chưa NFC thấp hơn `nfc_only` 0.0013 ± 0.0007, khoảng 0.1% câu đổi nhãn. NFC có ảnh hưởng nhưng nhỏ so với nhiễu.

## 3. Nhận xét

1. **ViSoBERT kém bền hơn baseline trước nhiễu.** Ở `all_medium`, ViSoBERT tụt 0.101 so với 0.037 của baseline và **đã thấp hơn baseline** (0.594 so với 0.606); ở `all_heavy` khoảng cách lớn hơn (0.514 so với 0.576). Lợi thế 5 điểm trên test sạch mất hết khi khoảng 1/4 số từ bị đổi.
2. **ViSoBERT vẫn hơn ở các biến đổi giống teencode thật** (viết tắt, `teencode_chars`, bỏ dấu, `all_light`), có thể vì nó được tiền huấn luyện trên văn bản mạng xã hội. Nó yếu nhất ở các biến đổi phá vỡ chữ: chèn ký tự (tụt 0.21), kéo dài và số thay chữ (tụt 0.058).
3. **Giả thuyết (chưa kiểm chứng):** tokenizer subword của ViSoBERT cắt từ bị chèn ký tự thành các mảnh vô nghĩa, còn char n-gram của baseline vẫn giữ được một phần chuỗi con. Kiểm chứng ở phần phân tích lỗi tuần 6 (so số subword trước/sau nhiễu), không chỉnh gì ở đây.
4. Baseline trên `teencode_chars_medium` tăng 0.002 dù 3.0% câu đổi nhãn: lỗi mới và lỗi được sửa gần như bù trừ. Chênh lệch này nằm trong mức dao động của một seed nhiễu, **không** kết luận nhiễu giúp mô hình.
5. Ý nghĩa cho các cấu hình sau: chuẩn hóa (B) nhắm đúng các dạng ViSoBERT yếu nhất (chèn ký tự, kéo dài, số thay chữ là lớp luật D-021). Tuy vậy theo D-026, mọi chỉnh sửa cho B/C/D phải dựa trên train/dev, không dựa trên bảng này.

## 4. Giới hạn

- Một seed nhiễu; chưa có khoảng tin cậy. Tuần 6 dùng `noisy_test_predictions.csv` để bootstrap/so cặp.
- Các bộ có % từ bị đổi khác nhau (cột 2): so mức tụt giữa các bộ phải đọc kèm cột này.
- Nhiễu sinh ngẫu nhiên đều trên các từ, không nhắm vào từ độc hại như người lách luật thật (D-023).
- ViSoBERT seed 42 là mô hình tái tạo hơi khác mô hình đã báo cáo ở D-019 (mục 1).
