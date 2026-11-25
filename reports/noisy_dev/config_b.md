# Cấu hình B trên bộ dev nhiễu (D-028)

> Cấu hình B: `normalize()` đặt trước mô hình A đã đóng băng, không train lại.
> Hai biến thể: `rules` = chỉ lớp luật (D-021); `full` = lớp luật + từ điển teencode (D-022).
> Bộ nhiễu: giống `config_a.md` (9 bộ + `nfc_only` từ ViHSD dev, 2672 câu, seed nhiễu 2611).
> File gốc: `reports/tfidf_lr_wordchar_balanced/noisy_dev_b_<biến thể>_metrics.json`, `reports/visobert_balanced/noisy_dev_b_<biến thể>_summary.json` (và `visobert_balanced_s4x/noisy_dev_b_<biến thể>_metrics.json`).
> Lệnh: `python -m src.evaluate --config <config> --noisy-dev --normalize <rules|full> [--seed S]`; ViSoBERT chạy trên Kaggle (`docs/kaggle.md` mục 7).
> **Chưa chạy trên test nhiễu.** Biến thể đã chọn sẽ chạy trên test nhiễu đúng một lần ở tuần 6.

## 1. Lần chạy và các cổng kiểm tra

| Mô hình | Lần chạy | Tham chiếu của A | Tham chiếu của B |
|---|---|---|---|
| `tfidf_lr_wordchar_balanced` | máy cá nhân; tất định (D-013) | `original` | `original` đã chuẩn hóa |
| `visobert_balanced` seed 42/43/44 | Kaggle (25/11/2026), Tesla T4, torch 2.11.0+cu130, transformers 4.57.6 (`kaggle_env_config_b.txt`); mô hình của D-026, không train lại | `nfc_only` | `original` đã chuẩn hóa |

Mọi lần chạy (2 mô hình × 2 biến thể, ViSoBERT 3 seed) đều qua 3 cổng:

- **Hash đóng băng:** `src/normalize.py` và `resources/teencode.tsv` khớp hash ghi trong D-028 (code dừng nếu khác).
- **A đúng như đã commit:** trong cùng lần chạy, A được đoán lại trên cả 10 bộ. Không câu nào khác với `noisy_dev_predictions.csv` của D-027 (0/26 720 câu mỗi lần chạy). Môi trường trùng với lần chạy A (cùng build cu130).
- **`nfc_only` và `original` sau chuẩn hóa giống hệt nhau** (vì `normalize` đã có NFC), nên B có đúng một tham chiếu.

## 2. Kết quả chính

Lợi ích = F1_B − F1_A, tính trong từng seed rồi mới gộp (trung bình ± độ lệch chuẩn mẫu, kèm số seed có lợi ích dương). "% lấp lại" = lợi ích trung bình / mức tụt trung bình của A (`config_a.md`). "Khôi phục" = % câu bị nhiễu đổi mà `normalize(nhiễu) == normalize(gốc)`; số này không phụ thuộc mô hình. Ô "—": A chỉ tụt khoảng 0.01–0.02 và lợi ích gần 0, nên tỉ lệ này không có nghĩa.

### ViSoBERT (3 seed), biến thể `rules`

| Bộ | A macro-F1 | B macro-F1 | Lợi ích B − A | Seed dương | % lấp lại | Khôi phục |
|---|---|---|---|---|---|---|
| dev sạch (`original`) | 0.7234 ± 0.0061 | 0.7214 ± 0.0039 | −0.0020 ± 0.0029 | 1/3 | — | — |
| abbreviate_medium | 0.7118 ± 0.0045 | 0.7108 ± 0.0025 | −0.0010 ± 0.0020 | 1/3 | — | 0% |
| teencode_chars_medium | 0.7117 ± 0.0043 | 0.7108 ± 0.0028 | −0.0009 ± 0.0019 | 1/3 | — | 0% |
| remove_diacritics_medium | 0.7021 ± 0.0039 | 0.7013 ± 0.0040 | −0.0008 ± 0.0004 | 0/3 | — | 0% |
| elongate_medium | 0.6392 ± 0.0119 | 0.7060 ± 0.0047 | **+0.0668 ± 0.0154** | 3/3 | 80% | 44% |
| substitute_digits_medium | 0.6604 ± 0.0049 | 0.6799 ± 0.0063 | +0.0195 ± 0.0029 | 3/3 | 31% | 5% |
| insert_chars_medium | 0.5014 ± 0.0210 | 0.5767 ± 0.0254 | **+0.0753 ± 0.0092** | 3/3 | 34% | 13% |
| all_light | 0.6676 ± 0.0060 | 0.6903 ± 0.0039 | +0.0228 ± 0.0039 | 3/3 | 42% | 19% |
| **all_medium** | 0.5999 ± 0.0122 | **0.6536 ± 0.0118** | **+0.0536 ± 0.0030** | 3/3 | **44%** | 11% |
| all_heavy | 0.5228 ± 0.0080 | 0.6113 ± 0.0072 | +0.0885 ± 0.0008 | 3/3 | 44% | 6% |

**Chi phí trên dev sạch** = F1_A(`nfc_only`) − F1_B(gốc) = **+0.0010 ± 0.0020** (dương = chuẩn hóa làm hại câu sạch). Lớp luật đổi 2.6% câu dev sạch.

### So sánh hai biến thể

| | ViSoBERT `rules` | ViSoBERT `full` | Baseline `rules` | Baseline `full` |
|---|---|---|---|---|
| B ở `all_medium` | **0.6536 ± 0.0118** | 0.6511 ± 0.0141 | 0.6129 | 0.6092 |
| Lợi ích ở `all_medium` | +0.0536 ± 0.0030 (3/3) | +0.0511 ± 0.0024 (3/3) | +0.0190 | +0.0154 |
| Chi phí trên dev sạch | +0.0010 ± 0.0020 | +0.0097 ± 0.0060 | −0.0008 | +0.0004 |
| Lợi ích trên dev sạch (B − A, `original`) | −0.0020 ± 0.0029 (1/3) | −0.0106 ± 0.0047 (0/3) | +0.0008 | −0.0004 |
| % câu dev sạch bị `normalize` đổi | 2.6% | 25.0% | 2.6% | 25.0% |
| Lợi ích ở `abbreviate_medium` | −0.0010 | −0.0068 (0/3) | −0.0002 | +0.0050 |

Baseline chạy 1 lần (tất định), không có độ lệch chuẩn. Toàn bộ 10 bộ của baseline nằm trong file metrics.

### F1 từng lớp ở `all_medium` (`rules`)

| Mô hình | Lớp | A | B | dev sạch (A) |
|---|---|---|---|---|
| ViSoBERT (3 seed) | CLEAN | 0.885 ± 0.023 | 0.913 ± 0.015 | 0.945 |
| | OFFENSIVE | 0.386 ± 0.017 | 0.456 ± 0.015 | 0.563 |
| | HATE | 0.529 ± 0.003 | 0.592 ± 0.006 | 0.662 |
| Baseline | CLEAN | 0.899 | 0.907 | 0.915 |
| | OFFENSIVE | 0.390 | 0.406 | 0.429 |
| | HATE | 0.492 | 0.525 | 0.555 |

Lợi ích lớn nhất rơi vào OFFENSIVE (+0.07) và HATE (+0.06), đúng hai lớp quan trọng.

## 3. Áp quy tắc chọn (ghi trước trong D-028)

1. Chọn theo ViSoBERT, macro-F1 trung bình 3 seed ở `all_medium`: `rules` 0.6536, `full` 0.6511.
2. Chênh **0.0025 < 0.01**, nên chọn **`rules`** (`rules` cũng là biến thể cao hơn).
3. Chi phí trên dev sạch của `rules` là **0.0010 ≤ 0.01**, nên không có đánh đổi cần ghi.
4. Baseline chỉ báo cáo: cũng xếp `rules` trên `full` ở `all_medium` (0.6129 so với 0.6092).

B-retrain không bị kích hoạt, vì nó chỉ được cân nhắc khi **cả hai** biến thể tốn hơn 0.01 trên dev sạch.

**Kết luận: cấu hình B = `normalize(text)` (chỉ lớp luật) + mô hình A đóng băng.**

## 4. So với dự đoán ghi trước

| Dự đoán (D-028) | Thực tế | Đúng? |
|---|---|---|
| B khó thắng lớn | ViSoBERT `all_medium` +0.054 (lấp 44% mức tụt), 3/3 seed dương; lớn hơn nhiều so với dự đoán | **Sai** (B tốt hơn dự đoán) |
| Lợi ích tập trung ở `insert_chars`, `elongate`, `substitute_digits` | Đúng ba bộ này dương ở cả 3 seed; các bộ đơn lẻ khác ≈ 0 | Đúng |
| Bỏ dấu và `teencode_chars` nằm ngoài khả năng lớp luật | Khôi phục 0%, lợi ích −0.001 | Đúng |
| `noise.py` có dạng luật cố ý bỏ qua (lặp 2 lần...) | `elongate`: 1218/1264 câu không khôi phục được có ký tự lặp đúng 2 lần (D-021 chỉ gộp từ 3 lần) | Đúng (đã kiểm với `elongate`; chưa phân tích `insert_chars`, `substitute_digits`) |
| Từ điển gần như không giúp hoặc làm hại trên dev sạch | ViSoBERT: −0.011 ± 0.005, 0/3 seed dương; baseline ≈ 0 | Đúng |

Ghi chú về con số tham khảo ViLexNorm (+0.6 điểm macro-F1 với PhoBERT): không so trực tiếp được. Bài đó đo trên ViHSD gốc, không phải bộ nhiễu. Trên dev sạch, B của dự án này cũng không tăng (−0.002).

## 5. Diễn giải

- **Vì sao lớp luật giúp ViSoBERT nhiều hơn baseline** (+0.054 so với +0.019 ở `all_medium`): baseline có char n-gram nên vốn ít bị nhiễu ký tự (mức tụt 0.039). Tokenizer của ViSoBERT thì cắt "n.g.u" thành nhiều mảnh lạ. Gỡ nhiễu ký tự trả lại token mô hình đã học.
- **Lợi ích lớn dù % khôi phục thấp** (`all_medium`: chỉ 11% câu về đúng nguyên văn nhưng lấp 44% mức tụt): không cần khôi phục cả câu. Chỉ cần gỡ được các từ bị nhiễu nặng (chèn ký tự, kéo dài ≥ 3 lần) là mô hình đã đoán đúng lại.
- **Vì sao từ điển làm hại ViSoBERT:** mô hình được train trên văn bản thô nên đã học "ko", "đc"... như từ bình thường. Từ điển đổi 25% câu dev sạch thành dạng mà mô hình ít gặp lúc train (lệch train/suy luận, đã nêu trong D-028). Trên những câu bị viết tắt, từ điển có giúp một chút (+0.009 trên phần câu bị đổi ở `abbreviate_medium`). Nhưng phần lợi đó nhỏ hơn phần hại trên các câu sạch.
- **Khoảng cách còn lại:** ViSoBERT + B ở `all_medium` vẫn thấp hơn dev sạch khoảng 0.068 (0.654 so với 0.722). Phần còn thiếu nằm ở bỏ dấu, viết tắt, teencode và các dạng luật bỏ qua. Đó là việc của C (seq2seq) và D (huấn luyện đối kháng).

## 6. Giới hạn

- Một seed nhiễu, dev nhỏ (212 câu OFFENSIVE, 270 câu HATE). Chênh dưới khoảng 0.01 trên một bộ có thể chỉ là dao động (`config_a.md` mục 3).
- Dev nhiễu đã được dùng 2 lần để chọn (2 biến thể). Quy tắc chọn được ghi trước khi chạy, nên không có bước chỉnh luật theo kết quả.
- Lớp luật và bộ sinh nhiễu do cùng một người viết. Luật không được chỉnh theo `noise.py` (đóng băng từ D-021/D-028), nhưng hai bên vẫn có thể có cùng giả định về cách người dùng lách luật. Nhiễu sinh đều trên mọi từ, không nhắm vào từ độc hại (D-023).
- Từ điển (`full`) chưa được thử với mô hình train trên văn bản đã chuẩn hóa (B-retrain). Kết luận "từ điển làm hại" chỉ đúng với mô hình đóng băng.
