# Dữ liệu

> **Cảnh báo nội dung:** các bộ dữ liệu dưới đây chứa ngôn từ thô tục, xúc phạm và thù ghét. File này không trích nguyên văn bình luận.

Dữ liệu **không được commit** lên GitHub. File này ghi nguồn, giấy phép, cách tải và những gì đã kiểm tra được về dữ liệu.

## Tổng quan

| Dữ liệu | Vai trò | Nguồn | Giấy phép | Ngày kiểm tra |
|---|---|---|---|---|
| ViHSD | Train/dev/test chính, 3 nhãn | [uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd) | Chỉ dùng cho nghiên cứu (xem mục 2) | 04/10/2026 |
| ViCTSD | Kiểm tra chéo | TODO | TODO | TODO |
| ViHOS | Nhãn mức đoạn từ | TODO | TODO | TODO |
| ViTHSD | Tùy chọn: thù ghét theo đối tượng | TODO | TODO | TODO |

## Quy tắc dùng dữ liệu trong dự án

- Không commit dữ liệu, kể cả một phần. `.gitignore` đã chặn `data/*`, chỉ cho phép `data/README.md`.
- Không upload dữ liệu qua giao diện web của GitHub: cách này bỏ qua `.gitignore`.
- Trước khi commit notebook phải xóa output (*Clear All Outputs*), vì output có thể chứa nguyên văn bình luận.
- Giữ nguyên split train/dev/test gốc để kết quả so sánh được với bài báo.
- Tập test chỉ dùng để đánh giá cuối cùng. Xây từ điển teencode, sinh nhiễu để huấn luyện, chọn siêu tham số: chỉ dùng train và dev.

---

# ViHSD

## 1. Nguồn và trích dẫn

- **Bài báo:** Luu, Nguyen & Nguyen (2021), *A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts*, hội nghị IEA/AIE 2021. Bản arXiv: 2103.11528.
- **Repo của tác giả:** <https://github.com/sonlam1102/vihsd>
- **Nơi tải chính thức:** <https://huggingface.co/datasets/uitnlp/vihsd>. Đây là bản chính thức vì README trên GitHub của tác giả trỏ tới link này, và người đăng là tổ chức UIT NLP.

```bibtex
@inbook{Luu_2021,
   title={A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts},
   ISBN={9783030794576},
   ISSN={1611-3349},
   url={http://dx.doi.org/10.1007/978-3-030-79457-6_35},
   DOI={10.1007/978-3-030-79457-6_35},
   booktitle={Advances and Trends in Artificial Intelligence. Artificial Intelligence Practices},
   publisher={Springer International Publishing},
   author={Luu, Son T. and Nguyen, Kiet Van and Nguyen, Ngan Luu-Thuy},
   year={2021},
   pages={415–426} }
```

## 2. Giấy phép và điều kiện sử dụng

Kiểm tra ngày 04/10/2026. Hai nguồn chính thức ghi khác nhau:

| Nguồn | Nội dung |
|---|---|
| README trên GitHub | Dữ liệu chỉ dùng cho mục đích nghiên cứu; phải trích dẫn bài báo. Repo không có file `LICENSE`. |
| Trang HuggingFace | Nhãn License ghi `mit`. Dataset bị khóa (gated): phải đăng nhập và đồng ý điều kiện mới tải được. |

Dự án này làm theo **điều kiện chặt hơn**:

- Chỉ dùng cho nghiên cứu và học tập, phi thương mại.
- Luôn trích dẫn bài báo gốc.
- Không đăng lại dữ liệu ở bất kỳ đâu (GitHub, HuggingFace Spaces...).
- Mô hình huấn luyện trên ViHSD khi công bố phải ghi rõ nguồn dữ liệu và điều kiện "chỉ dùng cho nghiên cứu".

## 3. Cách tải

1. Đăng nhập HuggingFace, mở <https://huggingface.co/datasets/uitnlp/vihsd>, đọc và đồng ý điều kiện truy cập.
2. Vào tab **Files and versions**, tải `train.csv`, `dev.csv`, `test.csv`.
3. Đặt 3 file vào `data/raw/vihsd/`.
4. Chạy `git status` để chắc chắn không file CSV nào xuất hiện trong danh sách thay đổi.

TODO: bổ sung cách tải bằng code (HF access token) khi huấn luyện trên Kaggle/Colab.

## 4. Cấu trúc file

Ba file CSV, mã hóa UTF-8, tổng dung lượng khoảng 2.24 MB. Mỗi file có 2 cột:

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `free_text` | chuỗi | Nội dung bình luận |
| `label_id` | số nguyên | Nhãn: `0` = CLEAN, `1` = OFFENSIVE, `2` = HATE |

**Cách xác định ý nghĩa mã nhãn:** tài liệu của tác giả không ghi rõ `0/1/2` ứng với nhãn nào. Kết luận trên có được bằng cách đối chiếu số lượng mỗi giá trị `label_id` trong cả 3 tập với bảng thống kê trong bài báo; cả 9 con số đều khớp.

## 5. Định nghĩa nhãn (theo bài báo)

| Nhãn | Định nghĩa |
|---|---|
| CLEAN | Bình luận bình thường, không có từ ngữ thô tục và không công kích ai. |
| OFFENSIVE | Có từ ngữ thô tục nhưng không nhắm vào một cá nhân hay nhóm người cụ thể. |
| HATE | Công kích trực tiếp một cá nhân hoặc một nhóm người. Có thể không chứa từ thô tục nếu ý nghĩa vẫn là công kích. |

Điểm phân biệt OFFENSIVE và HATE là **có đối tượng bị nhắm tới hay không**, không phải mức độ thô tục.

## 6. Thu thập và gán nhãn (theo bài báo)

- **Nguồn:** bình luận trên các trang Facebook và kênh YouTube tiếng Việt. Tên người đăng đã được loại bỏ.
- **Thời gian thu thập:** bài báo, repo GitHub và dataset card đều không nêu (kiểm tra 04/10/2026).
- **Quy trình:** việc gán nhãn là do 2 người làm, mỗi người gán nhãn đọc hướng dẫn rồi gán nhãn; khi có bất đồng thì cần thêm người gán nhãn thứ 3; sử dụng đến người gán nhãn thứ 4 như người cuối cùng nếu cả 3 người trước vẫn có sự bất đồng
- **Độ đồng thuận:** Cohen's kappa = 0.52. Theo thang Landis & Koch (1977), đây là mức đồng thuận vừa phải (0.41–0.60).

Kappa ở mức vừa phải cho thấy chính con người cũng thường bất đồng về nhãn. Vì vậy nhãn có nhiễu, và một phần lỗi của mô hình thực chất là do nhãn gây tranh cãi.

## 7. Thống kê

Số liệu tự đo bằng [notebooks/01_explore_vihsd.ipynb](../notebooks/01_explore_vihsd.ipynb), pandas 3.0.6, trên bản tải ngày 04/10/2026.

### Số lượng và tỷ lệ nhãn

| Tập | CLEAN | OFFENSIVE | HATE | Tổng |
|---|---|---|---|---|
| train | 19886 (82.7%) | 1606 (6.7%) | 2556 (10.6%) | 24048 |
| dev | 2190 (82.0%) | 212 (7.9%) | 270 (10.1%) | 2672 |
| test | 5548 (83.1%) | 444 (6.6%) | 688 (10.3%) | 6680 |
| **Tổng** | 27624 | 2262 | 3514 | **33400** |

Tổng 33400 dòng và số lượng từng nhãn khớp với bài báo. Tỷ lệ nhãn giữa 3 tập gần như bằng nhau.

### Độ dài bình luận (số từ, tách theo khoảng trắng)

| Tập | Trung bình | Trung vị | Phân vị 95% | Phân vị 99% | Dài nhất |
|---|---|---|---|---|---|
| train | 11.5 | 8 | 32 | 60 | 1701 |
| dev | 11.5 | 8 | 33.45 | 61.29 | 130 |
| test | 11.5 | 8 | 33 | 60 | 411 |

Đây là số **từ**, không phải số token. Số token của PhoBERT/ViSoBERT sẽ lớn hơn và cần đo lại bằng tokenizer thật trước khi chọn `max_length`.

### Từ khóa đặc trưng theo nhãn

Đo bằng [notebooks/02_keywords_vihsd.ipynb](../notebooks/02_keywords_vihsd.ipynb) ngày 26/10/2026, **chỉ trên train** (24036 dòng sau khi bỏ dòng lỗi). Phương pháp: z-score log-odds có prior Dirichlet (Monroe et al., 2008), `prior_strength = 1000`, `min_count = 5`; mỗi token đếm tối đa 1 lần trong một bình luận; token = âm tiết viết thường (chuẩn hóa NFC), bỏ dấu câu và emoji. Lý do các lựa chọn: [docs/decisions.md](../docs/decisions.md) D-007 đến D-009. Mục này không trích nguyên văn từ thô tục.

| Loại | Nhãn | Số token xét | z > 1.96 | z > 5 | z lớn nhất |
|---|---|---|---|---|---|
| âm tiết | CLEAN | 3465 | 310 | 43 | 13.35 |
| âm tiết | OFFENSIVE | 3465 | 297 | 37 | 18.95 |
| âm tiết | HATE | 3465 | 484 | 80 | 19.42 |
| bigram | CLEAN | 7299 | 124 | 4 | 7.21 |
| bigram | OFFENSIVE | 7299 | 390 | 16 | 6.97 |
| bigram | HATE | 7299 | 759 | 24 | 8.90 |

**Nhận xét (dựa trên top 30 token theo z mỗi nhãn):**

- **Tần suất thô không phân biệt được nhãn.** Top tần suất của cả 3 nhãn đều là từ chức năng ("là", "có", "mà", "thì"...). Log-odds cho top khác hẳn nhau giữa các nhãn.
- **CLEAN:** từ xưng hô lịch sự và tiểu từ thân mật ("anh", "em", "cô", "thầy", "ơi", "ạ", "nha"), từ khen ngợi ("đẹp", "thuyết phục", "dễ thương"), và **nhiều tên người** (họ phổ biến, tên nhân vật/kênh cụ thể).
- **OFFENSIVE:** gần như toàn bộ là từ chửi thề. **14/30 token là dạng viết tắt** (cụm 1–4 phụ âm), và cùng một từ gốc xuất hiện dưới nhiều dạng: viết đủ, viết tắt 2 chữ, viết tắt 3–4 chữ, viết tắt có và không có dấu "đ".
- **HATE:** khác OFFENSIVE ở **từ chỉ đối tượng**: đại từ ngôi thứ hai và từ chỉ người mang sắc thái coi thường, so sánh với con vật, công kích trí tuệ, và **từ chỉ nhóm người** (chính trị, vùng miền, quốc gia, "ad"/quản trị viên). Từ chửi thề cũng có mặt nhưng z thấp hơn ở OFFENSIVE. Điều này khớp với định nghĩa nhãn: HATE có đối tượng bị nhắm tới.
- HATE có nhiều token đặc trưng nhất (484 âm tiết z > 1.96) dù ít bình luận hơn CLEAN gấp gần 8 lần.

**Hệ quả cho các bước sau:**

- **Chuẩn hóa (tuần 3):** từ điển teencode cần gom các dạng viết tắt của cùng một từ gốc về một dạng; danh sách token OFFENSIVE/HATE có z cao là điểm khởi đầu để duyệt thủ công (chỉ từ train).
- **Kiểm tra thiên lệch (tuần 6):** từ chỉ nhóm người có z cao ở HATE, nên mô hình có nguy cơ gán HATE oan cho câu trung tính chỉ nhắc đến nhóm đó. Bộ câu mẫu trung tính phải chứa các nhóm này.
- **Tương quan giả:** tên người và tên kênh có z cao ở CLEAN phản ánh *nguồn* bình luận, không phải ngôn từ; mô hình có thể học "tên này → CLEAN". Cần xem lại khi phân tích lỗi.

**Hạn chế:** z cao chỉ cho thấy token *đi kèm* nhãn, không *gây ra* nhãn. Ngưỡng 1.96 (khoảng 5% do ngẫu nhiên) áp cho từng token riêng lẻ; vì kiểm tra hàng nghìn token cùng lúc (so sánh bội), nên ngay cả khi không token nào thật sự đặc trưng vẫn có khoảng 5% × 3465 ≈ 170 âm tiết vượt 1.96 do ngẫu nhiên. Cột `z > 1.96` vì vậy chứa nhiều nhiễu; cột `z > 5` đáng tin hơn. Âm tiết đơn lẻ có thể nhập nhằng ("con", "cái" vừa là loại từ vừa là thành phần chửi). Bigram có thể ghép qua dấu câu. Emoji và chữ bị chèn ký tự không được phân tích. `prior_strength` chưa được dò.

### Kết quả tham chiếu từ bài báo

Mô hình tốt nhất tác giả báo cáo là mBERT (`bert-base-multilingual-cased`): accuracy 86.88%, macro-F1 62.69% trên tập test. Đây là số của bài báo, chưa phải kết quả của dự án này.

## 8. Quan sát và hạn chế

**Mất cân bằng nhãn.** CLEAN chiếm khoảng 83%, nhiều gấp khoảng 12.4 lần OFFENSIVE và 7.8 lần HATE (tính trên train). Một mô hình luôn đoán CLEAN đã đạt accuracy 83.1% trên test (5548/6680). Vì vậy dự án dùng macro-F1 làm chỉ số chính, không dùng accuracy.

**Trùng lặp giữa các tập (rò rỉ có sẵn trong split gốc).**

| Tập | Dòng trùng với train | Tỷ lệ | Số câu khác nhau |
|---|---|---|---|
| dev | 321 / 2672 | 12.0% | 318 |
| test | 797 / 6680 | 11.9% | 782 |

- Phần test trùng với train gồm 714 CLEAN, 40 OFFENSIVE, 43 HATE; độ dài trung vị 6 từ, nên phần lớn là câu cụ thể chứ không chỉ là emoji hay câu một từ.
- 725 / 797 câu đó có nhãn ở test khớp với nhãn trong train: mô hình chỉ cần ghi nhớ là đoán đúng, nên điểm test có thể cao hơn năng lực thật.
- Dự án vẫn giữ nguyên split gốc để so sánh được với bài báo, và mọi lần đánh giá báo thêm điểm trên phần đã gặp / chưa gặp trong train (D-014). Nếu so sau chuẩn hóa nhẹ (NFC, viết thường, gộp khoảng trắng) thì có 805 câu test đã gặp thay vì 797.
- Kết quả với baseline: macro-F1 trên phần test chưa gặp là 0.641, gần bằng 0.643 trên toàn bộ test; câu trùng gần như không thổi phồng điểm (chi tiết: `reports/baseline_tfidf.md` mục 4).

**Trùng lặp trong từng tập.** Train có 1489 dòng trùng (6.2%), dev 22 (0.8%), test 104 (1.6%).

**Nhãn mâu thuẫn.** Trong train có 131 câu xuất hiện nhiều lần với nhãn khác nhau (khoảng 9% trong 1434 câu bị lặp). Khi đọc thử 20 câu ngẫu nhiên mỗi nhãn (train, `random_state=42`), cũng gặp một số nhãn gây tranh cãi, nhất là câu đùa cợt và câu thiếu ngữ cảnh. Điều này phù hợp với kappa 0.52.

**Dòng lỗi.**

- Train có 2 dòng `free_text` rỗng; dev và test không có.
- Có những dòng mà nội dung chỉ là chuỗi `#ERROR!` (train: 10 dòng, test: 4 dòng; dev: 3 dòng). Nhiều khả năng đây là lỗi bảng tính làm mất nội dung gốc; nguyên nhân chưa được xác nhận.
- Không có biến thể khác (khoảng trắng thừa, viết thường, có chữ đi kèm): đã kiểm tra ngày 06/10/2026.
- **Cách xử lý** (`src/data.py`, xem [docs/decisions.md](../docs/decisions.md) D-004): train bỏ 12 dòng lỗi, còn 24036 dòng (CLEAN 19877, OFFENSIVE 1604, HATE 2555); dev và test giữ đủ 2672 và 6680 dòng, text của dòng lỗi được thay bằng chuỗi rỗng.

**Đặc điểm ngôn ngữ.** Mẫu đọc thử có nhiều viết tắt, teencode, chữ không dấu và ký tự kéo dài. Đây là lý do dự án cần phần chuẩn hóa văn bản.

**Thông tin cá nhân.** Tên người đăng đã được loại bỏ, nhưng tên người thật vẫn xuất hiện trong nội dung bình luận (do tag bạn bè hoặc nhắc tới người nổi tiếng). Không trích các bình luận này trong báo cáo hay demo.

**Thời gian thu thập không rõ.** Từ lóng và teencode thay đổi theo thời gian, nên mô hình có thể kém hơn trên bình luận mới.
