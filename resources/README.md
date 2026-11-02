# resources/

Tài nguyên tự tuyển chọn, được commit (khác `data/`, nơi chứa dữ liệu tải về và không commit).

> Cảnh báo nội dung: `teencode.tsv` chứa từ tục ở cột `replacement`, vì đó là nghĩa đầy đủ của các viết tắt tục. Đây là phạm vi cần thiết của nhiệm vụ (CLAUDE.md mục 10.7).

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
- `agree` được đo trên chính các ngữ cảnh dùng để quyết định, nên **cao hơn thực tế**. Precision thật phải đo lại trên mẫu khác (seed khác).
- Không dùng file này để sinh nhiễu ở `noise.py` (D-022), nếu không cấu hình B sẽ được chấm trên chính từ điển của nó.
