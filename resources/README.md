# resources/

Tài nguyên tự tuyển chọn, được commit (khác `data/`, nơi chứa dữ liệu tải về và không commit).

> Cảnh báo nội dung: `teencode.tsv` (cột `replacement`) và `noise_abbrev.tsv` (cột `normalized`) chứa từ tục, vì đó là nghĩa đầy đủ của các viết tắt tục. Đây là phạm vi cần thiết của nhiệm vụ (CLAUDE.md mục 10.7).

## `teencode.tsv`: từ điển teencode/viết tắt (phần 3b, D-022)

**Nguồn:** chỉ tập **train** của ViHSD. Lệnh sinh ứng viên và ngữ cảnh:

```bash
python -m src.keywords --min-count 10 --contexts 10 --seed 42
```

**Quy tắc chấm (đặt trước khi chấm):** đọc 10 ngữ cảnh ngẫu nhiên (seed 42) của mỗi ứng viên; `keep` nếu ít nhất **9/10** ngữ cảnh cùng một nghĩa, ngược lại `reject` kèm lý do. Vòng đầu do Claude chấm (01/11/2026), người dùng duyệt lại các mục `keep`.

| Cột | Ý nghĩa |
|---|---|
| `token` | token viết thường, sau lớp luật 3a |
| `decision` | `keep` (dùng để chuẩn hóa) hoặc `reject` (giữ nguyên trong văn bản) |
| `replacement` | dạng đầy đủ; rỗng nếu `reject` |
| `agree` | số ngữ cảnh khớp nghĩa trên 10; rỗng nếu `reject` |
| `count`, `CLEAN`, `OFFENSIVE`, `HATE` | số câu train chứa token, tổng và theo nhãn |
| `source` | `auto` (không nguyên âm hoặc có f/j/w/z) hoặc `manual` (danh sách gõ tay) |
| `note` | lý do loại, hoặc ngữ cảnh lệch nghĩa của mục `keep` |

**Lưu ý:**
- `agree` được đo trên chính các ngữ cảnh dùng để quyết định, nên **cao hơn thực tế**. Precision thật đo lại trên mẫu khác: 0.98 (98/100 lần thay, seed 2027), chi tiết ở D-022 và `reports/teencode/`.
- Không dùng file này để sinh nhiễu ở `noise.py` (D-022), nếu không cấu hình B sẽ được chấm trên chính từ điển của nó.

## `noise_abbrev.tsv`: bảng viết tắt cho sinh nhiễu (D-024)

> **Giấy phép riêng:** file này là sản phẩm phái sinh từ ViLexNorm, phát hành theo **CC BY-NC-SA 4.0** (không theo giấy phép của code). Nguồn: Nguyen, Le & Nguyen (2024), *ViLexNorm: A Lexical Normalization Corpus for Vietnamese Social Media Text*, EACL 2024, <https://aclanthology.org/2024.eacl-long.85>. Trích dẫn đầy đủ ở `data/README.md`.

**Nguồn:** chỉ split **train** của ViLexNorm (commit `1070c6b`), không dùng ViHSD và không dùng `teencode.tsv`. Sinh lại bằng:

```bash
python -m src.noise --build-abbrev data/raw/vilexnorm/data/train.csv --min-count 3
```

| Cột | Ý nghĩa |
|---|---|
| `normalized` | từ chuẩn, viết thường |
| `variant` | dạng viết tắt/teencode, viết thường |
| `count` | số lần cặp xuất hiện trong ViLexNorm train; `abbreviate` chọn dạng theo trọng số này |

**Lưu ý:**
- File được sinh tự động và **không sửa tay**, để tái lập được và giữ độc lập với người làm dự án.
- Đã bỏ các cặp mà 5 biến đổi theo ký tự tự sinh ra được (kéo dài, bỏ dấu, số, teencode ký tự), và rút chữ lặp từ 3 lần trước khi đếm.
- Còn lỗi gõ lọt qua ngưỡng ("cubgx") và chơi chữ phụ thuộc ngữ cảnh ("bình" → "pink", từ "tide pink" = "thái bình").
