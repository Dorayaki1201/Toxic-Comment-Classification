# CLAUDE.md

Tài liệu hướng dẫn cho Claude khi làm việc trong repo này. Đọc kỹ trước khi viết hoặc sửa code.

## 1. Tổng quan dự án

**Tên dự án:** Hệ thống phát hiện ngôn từ tiêu cực, thù ghét và tục tĩu trong bình luận tiếng Việt, có khả năng nhận diện viết tắt, teencode và biến thể lách kiểm duyệt.

**Mục đích:** Dự án cá nhân của sinh viên, dùng làm sản phẩm đưa vào CV (xin thực tập AI/NLP). Một người làm toàn bộ từ A đến Z: dữ liệu, huấn luyện, đánh giá, demo, tài liệu.

**Vì vậy mọi quyết định phải ưu tiên:**
1. Số liệu thật, đo được, tái lập được (không bịa số).
2. Người làm phải hiểu 100% code mình nộp (sẽ bị hỏi phỏng vấn).
3. Phạm vi vừa sức một người, hoàn thành được, đừng làm quá rộng.

## 2. Bài toán

Phân loại một bình luận thành 3 nhãn (theo định nghĩa của ViHSD):
- `CLEAN`: bình thường
- `OFFENSIVE`: có từ thô tục nhưng không nhắm vào cá nhân hay nhóm cụ thể
- `HATE`: công kích nhắm vào một cá nhân hoặc một nhóm người (có thể không chứa từ thô tục)

**Phần nâng cao (điểm nhấn của dự án):** nhận diện và chuẩn hóa viết tắt, teencode, ký tự chèn, không dấu, ký tự kéo dài, rồi chứng minh bằng số liệu rằng hệ thống bền hơn trước các biến thể đó.

Đầu ra mong muốn của hệ thống:
```
Đầu vào:  "thằng này ngu vcl"
Đầu ra:   nhãn, độ tin cậy, văn bản sau chuẩn hóa, (tùy chọn) các từ gây độc hại
```

Mô hình chỉ phân loại. Việc xử lý tiếp (cho đăng, làm mờ, chặn, đưa vào hàng chờ) là logic nghiệp vụ riêng ở `src/moderation.py`.

## 3. Dữ liệu

Dùng dữ liệu công khai, không đẩy dữ liệu lên GitHub, chỉ ghi hướng dẫn tải trong `data/README.md`.

| Dữ liệu | Vai trò |
|---|---|
| ViHSD | Chính: train/dev/test, 3 nhãn |
| ViCTSD | Kiểm tra chéo (cross-dataset) |
| ViHOS | Nhãn mức đoạn từ, phục vụ highlight/giải thích |
| ViTHSD | Tùy chọn: thù ghét theo đối tượng |

Quy tắc:
- Kiểm tra giấy phép và phiên bản mới nhất của từng tập trước khi dùng, ghi vào `data/README.md`.
- Giữ nguyên split gốc nếu có. Nếu tự chia, cố định `seed` và lưu chỉ mục chia.
- **Tuyệt đối không để lọt dữ liệu test vào train** (kể cả khi sinh nhiễu hoặc xây từ điển teencode: chỉ được thống kê trên tập train).
- Nếu tự thu thập thêm dữ liệu: tuân thủ điều khoản nền tảng, ẩn danh thông tin cá nhân (tên, số điện thoại, link trang cá nhân).

## 4. Hướng kỹ thuật

### 4.1 Mô hình (so sánh từ đơn giản đến mạnh)
1. Baseline: TF-IDF (word + char n-gram) + Logistic Regression / SVM
2. PhoBERT (cần dữ liệu đã tách từ)
3. ViSoBERT (huấn luyện trên dữ liệu mạng xã hội)
4. (Tùy chọn) XLM-R để so sánh đa ngữ; LLM zero/few-shot qua API để so sánh

### 4.2 Bốn cấu hình cần so sánh (cốt lõi của phần nâng cao)
- **A**: Mô hình thường, không chuẩn hóa
- **B**: Chuẩn hóa bằng luật + từ điển, rồi phân loại
- **C**: Chuẩn hóa bằng seq2seq (ViT5 hoặc BARTpho), rồi phân loại
- **D**: Huấn luyện đối kháng (tăng cường dữ liệu bằng văn bản đã bị nhiễu), không chuẩn hóa

### 4.3 Chuẩn hóa văn bản (`src/normalize.py`)
Làm theo thứ tự từ dễ đến khó, xong lớp nào ổn rồi mới sang lớp sau:
1. Lớp luật: từ điển teencode/viết tắt, ký tự kéo dài ("điiiên" → "điên"), ký tự chèn ("đ.ịt", "d!t"), số thay chữ.
2. Khôi phục dấu tiếng Việt cho văn bản không dấu.
3. Seq2seq chuẩn hóa, huấn luyện trên cặp (nhiễu, chuẩn) sinh tự động.
4. (Nâng cao) Phát hiện từ lóng mới: so sánh tần suất/log-odds giữa lớp độc hại và sạch, rồi duyệt thủ công.

### 4.4 Sinh nhiễu (`src/noise.py`)
Các biến đổi: bỏ dấu, viết tắt, teencode, chèn ký tự, kéo dài ký tự, thay số, đổi emoji. Mỗi biến đổi có tham số mức độ (nhẹ/vừa/nặng) và `seed` để tái lập. Dùng cho 2 việc: tạo bộ test nhiễu (không được dùng để train cấu hình D) và tăng cường dữ liệu train cho cấu hình D (chỉ biến đổi trên tập train).

## 5. Đánh giá (bắt buộc đúng quy trình)

- Chỉ số chính: **macro-F1**; luôn báo cáo kèm F1 từng lớp (đặc biệt `HATE`, `OFFENSIVE`). Không dùng accuracy làm chỉ số chính.
- Có confusion matrix và **phân tích lỗi** (gom nhóm lỗi: mỉa mai, teencode, thiếu ngữ cảnh...).
- **Bộ test nhiễu**: biến đổi tập test gốc ở nhiều mức; báo cáo mức tụt điểm so với test gốc.
- **Cross-dataset**: train trên ViHSD, test trên ViCTSD.
- **Thiên lệch**: kiểm tra mô hình có gán nhãn độc hại oan cho câu chỉ nhắc đến một nhóm người không (bộ câu mẫu trung tính).
- Đánh giá riêng module chuẩn hóa (word accuracy, CER/WER).
- Chạy mỗi cấu hình quan trọng với **nhiều seed** (tối thiểu 3) và báo cáo trung bình ± độ lệch chuẩn nếu đủ tài nguyên.
- Model chọn dựa trên tập dev. Tập test chỉ chạy cuối cùng, không chỉnh siêu tham số theo test.

## 6. Cấu trúc repo

```
du-an-toxic-vi/
├── CLAUDE.md
├── README.md              # mô tả, kết quả, cách chạy, ảnh demo
├── requirements.txt
├── pyproject.toml         # cấu hình ruff/pytest
├── data/
│   └── README.md          # hướng dẫn tải dữ liệu, giấy phép (không commit dữ liệu)
├── notebooks/             # khám phá dữ liệu, thử nghiệm (đánh số 01_, 02_...)
├── src/
│   ├── data.py            # nạp, chia, tiền xử lý dữ liệu
│   ├── normalize.py       # chuẩn hóa teencode/viết tắt
│   ├── noise.py           # sinh nhiễu
│   ├── models.py          # baseline + transformer
│   ├── train.py
│   ├── evaluate.py        # macro-F1, robustness, bias, cross-dataset
│   ├── moderation.py      # logic xử lý sau khi phát hiện
│   └── utils.py           # seed, logging, đọc/ghi cấu hình
├── configs/               # file cấu hình thí nghiệm (yaml)
├── app/
│   └── app.py             # Gradio demo (+ trang bình luận mô phỏng)
├── tests/                 # test cho normalize, noise, moderation
└── reports/               # bảng kết quả, biểu đồ, phân tích lỗi
```

## 7. Công nghệ và môi trường

- Python 3.10+; PyTorch, HuggingFace Transformers/Datasets/Evaluate, scikit-learn, pandas
- Tách từ tiếng Việt: VnCoreNLP hoặc underthesea
- Theo dõi thí nghiệm: Weights & Biases (hoặc TensorBoard)
- Demo: Gradio, đưa lên HuggingFace Spaces; API (tùy chọn): FastAPI
- Huấn luyện: Kaggle Notebooks (chính), Colab (dự phòng). Viết code: VS Code. Lưu code: GitHub. Lưu mô hình: HuggingFace Hub.

Luôn lưu checkpoint thường xuyên (phiên Kaggle/Colab có thể bị ngắt). Ghi phiên bản thư viện trong `requirements.txt`.

## 8. Lệnh thường dùng

Cập nhật mục này khi lệnh thực tế thay đổi.

```bash
# Cài đặt
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell)
# source .venv/bin/activate     # Linux/macOS, Kaggle, Colab
pip install -r requirements.txt

# Kiểm tra chất lượng code
ruff check .
ruff format .
pytest -q

# Huấn luyện (ví dụ)
python -m src.train --config configs/visobert_base.yaml

# Đánh giá
python -m src.evaluate --config configs/visobert_base.yaml --split test

# Chạy demo
python app/app.py
```

## 9. Quy ước code

- Python, type hints cho hàm công khai, docstring ngắn gọn bằng tiếng Việt.
- Tên biến, hàm, file bằng tiếng Anh; chú thích và README bằng tiếng Việt.
- Hàm nhỏ, một việc; logic dùng lại phải nằm trong `src/`, không copy giữa notebook.
- Mọi yếu tố ngẫu nhiên phải nhận `seed`. Gọi `set_seed()` ở đầu mỗi lần chạy.
- Cấu hình thí nghiệm nằm trong `configs/`, không hard-code siêu tham số trong code.
- Không commit: dữ liệu, checkpoint nặng, khóa API, token. Dùng `.gitignore` và biến môi trường / secrets của Spaces.
- Commit nhỏ, thông điệp rõ (ví dụ `feat: add elongated-char normalizer`, `fix: leak of dev ids into train`).
- **Không** thêm dòng `Co-Authored-By` hay chữ ký AI nào vào commit message hoặc mô tả Pull Request.
- Quy trình Git: `main` luôn chạy được; mỗi phần lớn làm trên nhánh `feat/...`, merge vào `main` qua Pull Request; gắn tag ở mỗi mốc (`v0.1-baseline`...).
- Viết test cho `normalize.py`, `noise.py`, `moderation.py` (các phần logic thuần, dễ test).

## 10. Quy tắc cho Claude khi hỗ trợ dự án này

1. **Giải thích khi viết code.** Người dùng là sinh viên làm một mình và phải bảo vệ được code. Với mỗi đoạn quan trọng, nói ngắn gọn nó làm gì và vì sao chọn cách đó.
2. **Không bịa số liệu.** Không tự điền kết quả F1, độ chính xác hay số liệu so sánh. Chỗ nào chưa chạy thì để `0.xx` hoặc `TODO`. Chỉ ghi số đã đo thật, kèm cấu hình và seed.
3. **Không rò rỉ dữ liệu.** Luôn kiểm tra: xây từ điển, sinh nhiễu, chọn siêu tham số chỉ dựa trên train/dev, không dùng test.
4. **Làm theo từng bước nhỏ.** Hoàn thành và kiểm tra một phần (có thể chạy được) rồi mới sang phần sau; không sinh cả dự án một lần.
5. **Ưu tiên giải pháp đơn giản trước.** Baseline và lớp luật trước, mô hình phức tạp sau. Nếu một phần quá nặng so với tài nguyên (GPU miễn phí), nói rõ và đề xuất phương án nhẹ hơn.
6. **Nói rõ khi không chắc.** Về tên thư viện, API, phiên bản, giấy phép dữ liệu hay hạn mức GPU: nhắc người dùng kiểm tra lại nguồn chính thức thay vì khẳng định.
7. **Dữ liệu nhạy cảm.** Nội dung thô tục/thù ghét chỉ dùng ở mức cần thiết cho nhiệm vụ. Không tạo thêm nội dung thù ghét mới ngoài mục đích kiểm thử/sinh nhiễu có kiểm soát; trong README và báo cáo, tránh trích nguyên văn khi không cần.
8. **Chạy kiểm tra trước khi báo xong.** Sau khi sửa code, chạy `ruff` và `pytest` (nếu có) và nêu kết quả. Không nói "đã xong" khi chưa chạy.
9. **Giữ phạm vi.** Không thêm tính năng ngoài kế hoạch (mục 11) trừ khi người dùng yêu cầu.
10. **Cập nhật tài liệu.** Khi đổi cấu trúc, lệnh hoặc quyết định kỹ thuật, cập nhật CLAUDE.md và README tương ứng.
11. **Chế độ học: người dùng tự làm, Claude hướng dẫn.** Người dùng mới dùng Claude Code và muốn học, không muốn được làm hộ từ A đến Z.
    - Bước setup (VS Code, Claude Code, Git, môi trường): hướng dẫn từng bước cụ thể (chạy lệnh nào, tạo file gì).
    - Bước dữ liệu/code: giải thích khái niệm là gì và **vì sao** làm vậy, hướng dẫn người dùng tự viết rồi review; không đưa sẵn lời giải hoàn chỉnh trừ khi được yêu cầu.
    - Luôn kèm nguồn học: bài báo khoa học, tài liệu chính thức, website, video. Nói rõ khi không chắc nguồn còn tồn tại.
    - Làm từng target một; trước mỗi bước nói rõ làm gì, trên công cụ nào; chờ người dùng xong mới sang target sau.

## 11. Kế hoạch và tiến độ (đánh dấu khi xong)

**Tuần 1: Dữ liệu và baseline**
- [ ] Tải ViHSD, thống kê nhãn, độ dài, từ khóa
- [ ] Baseline TF-IDF + SVM/LR, ghi macro-F1 trên dev/test
- [x] Khởi tạo repo, cấu trúc thư mục, `requirements.txt`

**Tuần 2: Mô hình transformer**
- [ ] Tách từ + fine-tune PhoBERT
- [ ] Fine-tune ViSoBERT
- [ ] Xử lý mất cân bằng nhãn (class weight / focal loss / oversampling)

**Tuần 3: Chuẩn hóa bằng luật**
- [ ] Xây từ điển teencode/viết tắt (chỉ từ dữ liệu train)
- [ ] Xử lý ký tự kéo dài, ký tự chèn, số thay chữ
- [ ] Test đơn vị cho `normalize.py`

**Tuần 4: Sinh nhiễu và bộ test nhiễu**
- [ ] `noise.py` với các mức độ và seed
- [ ] Tạo bộ test nhiễu từ test gốc
- [ ] Đo mức tụt điểm của cấu hình A

**Tuần 5: Cấu hình C và D**
- [ ] Seq2seq chuẩn hóa (ViT5/BARTpho) trên cặp nhiễu-chuẩn sinh tự động
- [ ] Huấn luyện đối kháng (cấu hình D)

**Tuần 6: So sánh và phân tích**
- [ ] Bảng so sánh A/B/C/D trên test gốc và test nhiễu
- [ ] Cross-dataset, kiểm tra thiên lệch
- [ ] Phân tích lỗi, (tùy chọn) tìm từ lóng mới

**Tuần 7: Demo và đóng gói**
- [ ] `moderation.py` + Gradio demo (kèm trang bình luận mô phỏng)
- [ ] Đưa mô hình lên HuggingFace Hub, demo lên Spaces
- [ ] README hoàn chỉnh (kết quả, cách chạy, hạn chế, cảnh báo nội dung), báo cáo ngắn
- [ ] Viết dòng CV với số liệu thật

**Nếu thiếu thời gian, ưu tiên cắt theo thứ tự này (cắt từ dưới lên):**
Tìm từ lóng mới → Cấu hình C (seq2seq) → ViTHSD/đa nhãn → trang bình luận mô phỏng. **Không cắt:** baseline, ViSoBERT/PhoBERT, chuẩn hóa bằng luật, bộ test nhiễu, cấu hình D, phân tích lỗi, demo.

## 12. Logic xử lý sau khi phát hiện (`src/moderation.py`)

| Tình huống | Hành động |
|---|---|
| `CLEAN` | Cho đăng |
| `OFFENSIVE`, độ tin cậy cao | Làm mờ từ tục, cảnh báo người viết |
| `HATE`, độ tin cậy cao | Ẩn và đẩy vào hàng chờ kiểm duyệt |
| Độ tin cậy thấp | Đưa vào hàng chờ người kiểm duyệt |

Ngưỡng độ tin cậy là tham số có thể chỉnh trong demo. Hệ thống hỗ trợ người kiểm duyệt, không tự động xử phạt người dùng.

## 13. Đạo đức và giới hạn (phải có trong README)

- Cảnh báo nội dung nhạy cảm; không đăng lại nguyên văn thô tục không cần thiết.
- Ẩn danh dữ liệu tự thu thập, tuân thủ điều khoản nền tảng và giấy phép dữ liệu.
- Nêu rõ giới hạn: mô hình có thể nhầm (mỉa mai, đùa giữa bạn bè, phụ thuộc ngữ cảnh) và có thể thiên lệch theo dữ liệu huấn luyện.
- Không dùng để tự động trừng phạt người dùng mà không có người kiểm tra.

## 14. Tiêu chí hoàn thành (Definition of Done)

- [ ] Có baseline và ít nhất một mô hình transformer, so sánh bằng macro-F1 + F1 từng lớp
- [ ] Có 4 cấu hình A/B/C/D (hoặc ghi rõ cấu hình nào đã cắt và vì sao)
- [ ] Có bộ test nhiễu và số liệu mức tụt điểm
- [ ] Có phân tích lỗi và kiểm tra thiên lệch
- [ ] Có demo chạy được trên HuggingFace Spaces
- [ ] Repo GitHub sạch, README rõ, kết quả tái lập được
- [ ] Mọi con số trong CV đều đối chiếu được với `reports/`

## 15. Mẫu dòng ghi trong CV (chỉ điền sau khi đo thật)

> *Hệ thống phát hiện ngôn từ thù ghét/tục tĩu tiếng Việt chống lách kiểm duyệt*: Xây pipeline chuẩn hóa teencode, viết tắt và phân loại ViSoBERT trên ViHSD; macro-F1 đạt 0.xx trên tập gốc, giữ 0.xx trên tập bị nhiễu (mô hình thường chỉ còn 0.xx); sinh dữ liệu nhiễu tự động để huấn luyện đối kháng; demo Gradio trên HuggingFace Spaces. Công cụ: PyTorch, Transformers, VnCoreNLP.
