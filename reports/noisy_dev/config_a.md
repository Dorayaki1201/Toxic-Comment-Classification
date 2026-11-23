# Cấu hình A trên bộ dev nhiễu (D-027)

> Cấu hình A: không chuẩn hóa, văn bản đưa thẳng vào mô hình. Mô hình đã đóng băng.
> Bộ nhiễu: 9 bộ + đối chứng `nfc_only` từ ViHSD dev (2672 câu), seed nhiễu 2611 (`manifest.json`), cùng bộ sinh với bộ test nhiễu (D-025).
> File gốc: `reports/tfidf_lr_wordchar_balanced/noisy_dev_metrics.json`, `reports/visobert_balanced/noisy_dev_summary.json` (và `visobert_balanced_s4x/noisy_dev_metrics.json`).
> Lệnh: `python -m src.evaluate --config <config> --noisy-dev [--seed S]`; ViSoBERT chạy trên Kaggle (`docs/kaggle.md` mục 6).
> **Vai trò:** đây là mốc để chọn cấu hình B/C/D. Mọi lựa chọn cho B/C/D so với bảng này, không so với bộ test nhiễu.

## 1. Mô hình được đo

| Mô hình | Lần chạy | Tham chiếu để tính mức tụt | Điểm tham chiếu |
|---|---|---|---|
| `tfidf_lr_wordchar_balanced` | máy cá nhân, commit 20/11/2026; tất định (D-013) | `original` (baseline tự NFC) | 0.6330, khớp `dev_metrics.json` |
| `visobert_balanced` seed 42/43/44 | Kaggle (file kết quả ghi ngày 22/11/2026), Tesla T4, torch 2.11.0+**cu130**, transformers 4.57.6; dùng 3 mô hình đã train ở D-026, không train lại | `nfc_only` (transformer không tự NFC) | 0.7225 ± 0.0059 |

**Cổng "đúng mô hình"** (điểm dev gốc đoán lại so với `dev_metrics.json`):

| Seed | `dev_metrics.json` | Lần chạy này | Câu lệch | Đã dự đoán trước? |
|---|---|---|---|---|
| 42 | 0.7251 | 0.7256 | 1 (CLEAN trước bị đoán HATE, nay đoán đúng) | Có: mô hình tái tạo ở D-026 khác đúng câu này |
| 43 | 0.7268 | 0.7281 | 1 (OFFENSIVE trước bị đoán CLEAN, nay đoán đúng) | **Không** |
| 44 | 0.7165 | 0.7165 | 0 | – |

- s43 lệch ngoài dự đoán của D-027. Khác biệt môi trường duy nhất tìm được: lúc train dùng torch `2.11.0+cu128`, lần này dùng `2.11.0+cu130` (cùng phiên bản PyTorch, build cho CUDA khác). Sai số làm tròn nhỏ có thể làm một câu nằm sát ranh giới đổi nhãn. Đây là giả thuyết, chưa kiểm chứng.
- Lệch 1/2672 câu thì không thể là nạp nhầm mô hình (nạp nhầm sẽ lệch hàng chục câu trở lên). Theo quy tắc của D-026, không chạy lại để tìm lần chạy khớp. Mức tụt vẫn hợp lệ vì tham chiếu và bộ nhiễu được đoán trong cùng một lần chạy, cùng môi trường.

## 2. Kết quả

Mức tụt = điểm tham chiếu − điểm bộ nhiễu (dương = tụt), tính bằng điểm macro-F1 tuyệt đối. ViSoBERT: trung bình ± độ lệch chuẩn mẫu qua 3 seed; mức tụt tính trong từng seed rồi mới gộp. "Tụt trên câu bị đổi" = chỉ tính những câu mà nhiễu thực sự thay đổi.

| Bộ | % từ đổi | Baseline macro-F1 | Baseline tụt | ViSoBERT macro-F1 | ViSoBERT tụt | ViSoBERT tụt trên câu bị đổi | ViSoBERT % đổi nhãn |
|---|---|---|---|---|---|---|---|
| tham chiếu | 0 | 0.6330 | — | 0.7225 ± 0.0059 | — | — | — |
| abbreviate_medium | 8.58 | 0.6204 | +0.013 | 0.7118 ± 0.0045 | +0.011 ± 0.001 | +0.012 | 1.9 |
| teencode_chars_medium | 7.60 | 0.6181 | +0.015 | 0.7117 ± 0.0043 | +0.011 ± 0.003 | +0.020 | 3.2 |
| remove_diacritics_medium | 20.34 | 0.6180 | +0.015 | 0.7021 ± 0.0039 | +0.020 ± 0.006 | +0.024 | 5.2 |
| elongate_medium | 25.89 | 0.6171 | +0.016 | 0.6392 ± 0.0119 | +0.083 ± 0.017 | +0.094 | 9.7 |
| substitute_digits_medium | 23.38 | 0.6228 | +0.010 | 0.6604 ± 0.0049 | +0.062 ± 0.001 | +0.070 | 8.4 |
| insert_chars_medium | 26.14 | 0.5702 | +0.063 | 0.5014 ± 0.0210 | +0.221 ± 0.023 | +0.242 | 31.1 |
| all_light | 8.82 | 0.6120 | +0.021 | 0.6676 ± 0.0060 | +0.055 ± 0.009 | +0.090 | 5.8 |
| all_medium | 27.11 | 0.5939 | +0.039 | 0.5999 ± 0.0122 | +0.123 ± 0.017 | +0.132 | 15.9 |
| all_heavy | 45.53 | 0.5756 | +0.057 | 0.5228 ± 0.0080 | +0.200 ± 0.010 | +0.203 | 24.3 |

F1 từng lớp (ViSoBERT: trung bình 3 seed):

| Bộ | Baseline OFFENSIVE | Baseline HATE | ViSoBERT OFFENSIVE | ViSoBERT HATE |
|---|---|---|---|---|
| tham chiếu | 0.429 | 0.555 | 0.562 | 0.661 |
| insert_chars_medium | 0.361 | 0.470 | 0.285 | 0.435 |
| all_medium | 0.390 | 0.492 | 0.386 | 0.529 |
| all_heavy | 0.371 | 0.459 | 0.319 | 0.400 |

Tỉ lệ đổi nhãn của baseline (all_light / all_medium / all_heavy): 4.3% / 10.7% / 15.3%. Ở mọi bộ, điểm trên phần câu chưa gặp trong train chênh với điểm toàn tập dưới 0.005, nên câu trùng train không làm sai lệch kết luận.

## 3. Dev nhiễu có đại diện cho test nhiễu không

So mức tụt với `reports/noisy_test/config_a.md`, **chỉ để kiểm tra tính đại diện**, không dùng để chọn gì.

| Bộ | Baseline tụt (dev / test) | ViSoBERT tụt (dev / test) |
|---|---|---|
| insert_chars_medium | 0.063 / 0.065 | 0.221 / 0.210 |
| elongate_medium | 0.016 / 0.006 | 0.083 / 0.058 |
| substitute_digits_medium | 0.010 / 0.020 | 0.062 / 0.058 |
| all_light | 0.021 / 0.013 | 0.055 / 0.043 |
| all_medium | 0.039 / 0.037 | 0.123 / 0.101 |
| all_heavy | 0.057 / 0.067 | 0.200 / 0.182 |

- **Xu hướng giống nhau:** chèn ký tự gây hại nặng nhất cho cả hai mô hình; ViSoBERT tụt khoảng 3 lần baseline ở `all_medium`; mức tụt tăng dần theo mức nhẹ → vừa → nặng.
- **Khác biệt:** ở mọi bộ trong bảng, ViSoBERT tụt trên dev nhiều hơn trên test từ 0.004 đến 0.025 (rõ nhất ở `elongate_medium`). Với baseline, mức tụt của từng biến đổi đơn lẻ lệch nhau khoảng ±0.01 theo cả hai chiều. Dev nhỏ hơn test 2.5 lần (212 câu OFFENSIVE, 270 câu HATE) và chỉ có một seed nhiễu, nên mức lệch này nằm trong biên dao động có thể chấp nhận. Theo D-027, ghi lại và không sửa bộ cho khớp.
- **Kết luận:** dev nhiễu dùng làm thước đo cho B/C/D được, với lưu ý rằng chênh lệch dưới khoảng 0.01 giữa hai cấu hình trên một bộ nhiễu đơn lẻ có thể chỉ là dao động.

## 4. Nhận xét cho B/C/D

1. Mục tiêu: thu hẹp khoảng 0.123 ± 0.017 điểm mà ViSoBERT mất ở `all_medium` (từ 0.7225 còn 0.5999). Trên dev, ViSoBERT nhiễu chỉ còn ngang baseline nhiễu (0.600 so với 0.594), dù trên dev sạch ViSoBERT hơn khoảng 0.09.
2. Nhóm đáng làm nhất là chèn ký tự, kéo dài và số thay chữ: chúng gây tụt lớn nhất, và đều là những gì lớp luật D-021 xử lý. Viết tắt, teencode và bỏ dấu gây tụt nhỏ hơn (0.011–0.020).
3. Mức tụt của `all_light` trên câu bị đổi (0.090) lớn hơn nhiều so với trên toàn tập (0.055), do chỉ khoảng một nửa số câu bị đổi. Khi so B/C/D, nên đọc cả hai cột.

## 5. Giới hạn

- Một seed nhiễu, chưa có khoảng tin cậy; dev nhỏ nên F1 của OFFENSIVE và HATE dao động mạnh.
- Dùng dev nhiễu nhiều lần để chọn có thể quá khớp dev: mọi lần thử cho B/C/D phải được ghi vào `docs/decisions.md`.
- ViSoBERT s42 và s43 khác `dev_metrics.json` 1 câu mỗi seed (mục 1).
- Nhiễu sinh ngẫu nhiên đều trên các từ, không nhắm vào từ độc hại như người lách luật thật (D-023).
