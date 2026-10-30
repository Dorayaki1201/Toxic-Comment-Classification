# Sổ quyết định kỹ thuật

Mỗi quyết định ghi: **bối cảnh** (vấn đề là gì), **quyết định** (chọn gì), **lý do**, **đánh đổi** (mất gì, phương án đã bỏ). Không xóa quyết định cũ; nếu đổi ý thì thêm quyết định mới và ghi "thay thế D-xxx".

Định dạng này dựa trên *Architecture Decision Records* (Michael Nygard, 2011): <https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions>.

---

## D-001. Giữ nguyên split train/dev/test gốc của ViHSD

- **Ngày:** 04/10/2026
- **Bối cảnh:** khi khám phá dữ liệu thấy 797/6680 dòng test (11.9%) trùng với train, 725 trong số đó có cùng nhãn. Mô hình chỉ cần ghi nhớ là đoán đúng, nên điểm test có thể cao hơn năng lực thật.
- **Quyết định:** giữ nguyên split gốc.
- **Lý do:** để so sánh được với kết quả của bài báo gốc và các nghiên cứu khác dùng ViHSD.
- **Đánh đổi:** điểm test có thể bị thổi phồng. Bù lại bằng cách báo cáo thêm điểm trên phần test không trùng với train (TODO, xem `data/README.md` mục 8). Phương án bỏ: tự chia lại không trùng, vì mất khả năng so sánh.

## D-002. Macro-F1 là chỉ số chính, không dùng accuracy

- **Ngày:** từ đầu dự án (CLAUDE.md mục 5); số liệu xác nhận ngày 04/10/2026
- **Bối cảnh:** CLEAN chiếm khoảng 83% dữ liệu. Mô hình luôn đoán CLEAN đã đạt accuracy 83.1% trên test.
- **Quyết định:** chỉ số chính là macro-F1, luôn báo cáo kèm F1 từng lớp.
- **Lý do:** macro-F1 cho mỗi lớp trọng số bằng nhau, nên mô hình bỏ qua OFFENSIVE/HATE sẽ bị phạt nặng.
- **Đánh đổi:** macro-F1 dao động mạnh hơn khi lớp nhỏ có ít mẫu, nên cần chạy nhiều seed.

## D-003. Đọc CSV với `keep_default_na=False`

- **Ngày:** 06/10/2026
- **Bối cảnh:** mặc định `pd.read_csv` biến cả chữ `"NA"`, `"null"`, `"None"`... thành NaN. Với văn bản tự do, một bình luận đúng là chữ `"null"` sẽ bị mất.
- **Quyết định:** đọc với `keep_default_na=False`, ô trống thành `""`, rồi tự quyết định thế nào là dòng lỗi.
- **Lý do:** đọc nguyên trạng, làm sạch tường minh trong code. Đo trên ViHSD thì hai cách cho cùng kết quả (2 dòng NaN ở train đều là ô trống thật), nhưng cùng hàm đọc sẽ dùng cho ViCTSD và câu người dùng gõ vào demo.
- **Đánh đổi:** không có NaN nên phải kiểm tra chuỗi rỗng thay vì dùng `isna()`.

## D-004. Dòng lỗi: bỏ ở train, giữ ở dev/test

- **Ngày:** 06/10/2026
- **Bối cảnh:** đo bằng code: train có 2 dòng rỗng + 10 dòng `#ERROR!`, dev có 3 dòng `#ERROR!`, test có 4 dòng `#ERROR!`. Không có biến thể khác.
- **Quyết định:** dòng lỗi = text sau khi bỏ khoảng trắng hai đầu là rỗng hoặc đúng bằng `#ERROR!`. Train bỏ các dòng này (còn 24036). Dev và test giữ đủ dòng (2672, 6680), thay text lỗi bằng `""`.
- **Lý do:** train sạch hơn để học. Dev/test giữ đúng số dòng gốc để so sánh được với bài báo (D-001) và không tự ý chỉnh tập đánh giá.
- **Đánh đổi:** 7 dòng rỗng ở dev/test vẫn mang nhãn gốc; dự đoán trên chuỗi rỗng không có ý nghĩa, nên nếu nhãn gốc là độc hại thì gần như chắc chắn bị tính sai. Ảnh hưởng rất nhỏ (3/2672 dòng dev, 4/6680 dòng test). Nhãn gốc của 7 dòng này chưa được thống kê. Phương án bỏ: bỏ ở cả 3 tập, vì lệch số với bài báo.

## D-005. `data.py` không sửa nội dung bình luận

- **Ngày:** 06/10/2026
- **Bối cảnh:** có thể tiện tay viết thường, bỏ khoảng trắng thừa ngay lúc nạp dữ liệu.
- **Quyết định:** `data.py` chỉ đảm bảo dữ liệu hợp lệ. Mọi biến đổi nội dung thuộc về `normalize.py`.
- **Lý do:** dự án cần so sánh cấu hình A (không chuẩn hóa) với B (có chuẩn hóa). Nếu `data.py` đã sửa chữ thì cấu hình A không còn là "không chuẩn hóa" thật.
- **Đánh đổi:** mọi bước sau phải gọi chuẩn hóa một cách tường minh.

## D-006. Test không đọc dữ liệu thật

- **Ngày:** 16/10/2026
- **Bối cảnh:** dữ liệu không được commit (giấy phép, nội dung nhạy cảm), nên không có trên máy người clone repo.
- **Quyết định:** test dùng bảng hoặc CSV nhỏ tự tạo trong thư mục tạm của pytest, chỉ chứa câu trung tính. Giá trị cần kiểm tra (như `"#ERROR!"`) được gõ thẳng trong test, không import hằng số từ code.
- **Lý do:** test chạy được ở mọi nơi. Gõ thẳng giá trị giúp bắt lỗi khi chính hằng số trong code bị sai.
- **Đánh đổi:** test không bắt được thay đổi của dữ liệu thật. Phần đó được kiểm tra bằng log số dòng khi nạp (24036 / 2672 / 6680).

## D-007. Thống kê từ khóa nằm trong module riêng `src/keywords.py`

- **Ngày:** 24/10/2026
- **Bối cảnh:** kế hoạch ban đầu đặt hàm thống kê trong `src/data.py`.
- **Quyết định:** tách thành `src/keywords.py`.
- **Lý do:** `data.py` là cửa vào dữ liệu; thống kê từ khóa là phân tích. Tuần 3 (từ điển teencode) và tuần 6 (từ lóng mới) sẽ dùng lại module này.
- **Đánh đổi:** thêm một file. Phương án bỏ: gộp vào `data.py`, khiến một module gánh hai vai trò.

## D-008. Phương pháp: log-odds có prior Dirichlet, đếm theo số bình luận

- **Ngày:** 24/10/2026
- **Bối cảnh:** tần suất thô cho top giống nhau ở mọi nhãn (từ phổ biến); tỷ lệ thô thì đẩy từ hiếm lên đầu. Ngoài ra có bình luận lặp một từ nhiều lần.
- **Quyết định:** dùng z-score log-odds với prior lấy theo tần suất chung (Monroe et al., 2008), tổng độ lớn prior `prior_strength = 1000`, bỏ token xuất hiện dưới `min_count = 5`. Mỗi token đếm tối đa 1 lần trong một bình luận.
- **Lý do:** z-score phạt token hiếm qua phương sai, nên không cần ngưỡng tùy tiện. Đếm theo bình luận bền trước spam và dễ diễn giải ("X% bình luận chứa token").
- **Đánh đổi:** đếm theo bình luận lệch khỏi giả định multinomial của bài báo gốc (một biến thể thực dụng). Giá trị `prior_strength` là lựa chọn, chưa dò; z-score chỉ cho thấy token *đi kèm* nhãn, không phải *gây ra* nhãn.

## D-009. Token là âm tiết, có thêm bigram

- **Ngày:** 24/10/2026
- **Bối cảnh:** từ tiếng Việt thường gồm nhiều âm tiết; tách theo khoảng trắng chỉ ra âm tiết.
- **Quyết định:** phân tích riêng hai bảng: âm tiết (unigram) và bigram (2 âm tiết liền nhau). Chuẩn hóa Unicode NFC và viết thường trước khi tách; bỏ dấu câu, emoji.
- **Lý do:** bigram bắt được nhiều từ ghép mà không cần công cụ tách từ (để tuần 2). NFC để cùng một chữ có dấu không bị tách khác nhau tùy cách gõ.
- **Đánh đổi:** bigram có thể ghép qua dấu câu (cặp vô nghĩa, phần lớn bị `min_count` lọc). Mất thông tin từ emoji và từ bị chèn ký tự (ví dụ chữ có dấu chấm xen giữa bị tách thành nhiều mảnh).

## D-010. Baseline là Pipeline TF-IDF + Logistic Regression của scikit-learn

- **Ngày:** 26/10/2026
- **Bối cảnh:** cần một mô hình đơn giản, chạy nhanh trên CPU để làm mốc so sánh và kiểm tra cả quy trình trước khi dùng GPU.
- **Quyết định:** gói TF-IDF và Logistic Regression trong một `Pipeline`; chọn Logistic Regression thay vì SVM tuyến tính.
- **Lý do:** `Pipeline.fit(train)` bảo đảm từ điển và IDF chỉ học trên train, dev/test chỉ được biến đổi, nên chống rò rỉ bằng cấu trúc code thay vì nhờ trí nhớ. Logistic Regression cho xác suất, cần cho `moderation.py` (xử lý theo độ tin cậy, CLAUDE.md mục 12) và để hiệu chỉnh ngưỡng về sau.
- **Đánh đổi:** SVM tuyến tính thường mạnh ngang hoặc hơn một chút trên văn bản nhưng không cho xác suất trực tiếp. Phương án bỏ: tự viết từng bước fit/transform (dễ lỡ fit trên dev/test).

## D-011. So sánh 6 biến thể trên dev, chạy test đúng một lần

- **Ngày:** 26/10/2026
- **Bối cảnh:** chưa biết đặc trưng nào tốt hơn (word, char hay cả hai) và có nên cân bằng nhãn không. Giả thuyết: char n-gram bền hơn trước teencode.
- **Quyết định:** 3 loại đặc trưng × {không cân bằng, `class_weight="balanced"`} = 6 file trong `configs/tfidf_lr_*.yaml`. Chọn theo macro-F1 trên dev; chạy test một lần cho biến thể được chọn, không chỉnh gì sau đó.
- **Lý do:** biến giả thuyết thành số đo; giữ test sạch (nhìn điểm test rồi chọn lại thì test thành dev).
- **Đánh đổi:** tham số còn lại **chưa dò**, chọn theo thông lệ: word n-gram 1–2, char n-gram 2–5 trong ranh giới từ (`char_wb`), `min_df = 2`, `sublinear_tf = true`, `C = 1.0`. Có thể chưa tối ưu.

## D-012. Tiền xử lý của baseline: chỉ NFC + viết thường, giữ token 1 ký tự

- **Ngày:** 26/10/2026
- **Bối cảnh:** `TfidfVectorizer` mặc định bỏ token 1 ký tự, trong khi teencode tiếng Việt có nhiều token như vậy (ví dụ "k" = "không").
- **Quyết định:** dùng `token_pattern = \b\w+\b` để giữ token 1 ký tự; tiền xử lý chỉ gồm chuẩn hóa Unicode NFC và viết thường.
- **Lý do:** đây là cấu hình A (không chuẩn hóa teencode), nên chỉ làm những biến đổi không đổi nghĩa. NFC tránh cùng một chữ có hai cách mã hóa.
- **Đánh đổi:** không có tách từ tiếng Việt (để PhoBERT ở tuần 2).

## D-013. Chọn `tfidf_lr_wordchar_balanced` làm baseline chính thức

- **Ngày:** 27/10/2026
- **Bối cảnh:** 6 cấu hình trên dev (`reports/baseline_tfidf.md`): `wordchar_balanced` cao nhất với macro-F1 0.633; cân bằng nhãn giúp ở cả 3 loại đặc trưng.
- **Quyết định:** chọn `tfidf_lr_wordchar_balanced` theo quy tắc của D-011; đã chạy test một lần: macro-F1 0.643.
- **Lý do:** quy tắc chọn được đặt trước khi thấy kết quả, tránh chọn theo cảm tính.
- **Đánh đổi:** khác biệt với cấu hình thứ hai (0.619) nhỏ và chưa có khoảng tin cậy. Từ nay không chỉnh baseline dựa trên điểm test; mọi cải tiến phải đánh giá lại trên dev như một mô hình mới. Logistic Regression với solver mặc định cho kết quả giống nhau mỗi lần chạy, nên yêu cầu chạy nhiều seed (CLAUDE.md mục 5) không làm thay đổi kết quả của baseline này; vẫn gọi `set_seed` cho đúng quy trình.

## D-014. Mọi lần đánh giá tự báo thêm điểm trên phần đã gặp / chưa gặp trong train

- **Ngày:** 28/10/2026
- **Bối cảnh:** khoảng 12% câu test trùng với train (D-001). Chưa biết điểm test 0.643 của baseline có bị thổi phồng do mô hình ghi nhớ những câu này không.
- **Quyết định:** `evaluate_saved_model` luôn ghi thêm 2 khối `seen_in_train` và `unseen_in_train` vào file chỉ số. "Trùng" nghĩa là giống một câu train sau khi chuẩn hóa NFC, viết thường và gộp khoảng trắng (`overlap_key`); chuỗi rỗng không bao giờ tính là trùng. Train dùng để so là train sau `load_vihsd` (đúng dữ liệu mô hình đã học).
- **Lý do:** đặt trong `evaluate.py` để mọi mô hình sau (PhoBERT, ViSoBERT...) tự có con số này, không phải nhớ đo riêng. Chuẩn hóa nhẹ bắt thêm các câu chỉ khác hoa/thường hoặc khoảng trắng, mà mô hình (vốn viết thường đầu vào) coi như một câu.
- **Chạy lại test có vi phạm D-011 không:** không. Mô hình giữ nguyên, điểm toàn bộ test ra lại đúng 0.6431; kết quả chỉ dùng để phân tích, không dùng để chọn hay chỉnh mô hình.
- **Kết quả (baseline):** 805/6680 câu test đã gặp (797 nếu so giống hệt). Phần chưa gặp: macro-F1 0.641 so với 0.643 toàn bộ, nên câu trùng gần như không thổi phồng điểm của baseline.
- **Đánh đổi:** phần đã gặp chỉ có 41 OFFENSIVE và 43 HATE nên macro-F1 trên phần này dao động mạnh. Phương án bỏ: so giống hệt chuỗi (nhất quán với số 797 nhưng bỏ sót biến thể hoa/thường); notebook đo một lần (mô hình sau dễ quên đo).

## D-015. Transformer dùng chung lệnh train/evaluate và chung giao diện với baseline

- **Ngày:** 28/10/2026
- **Bối cảnh:** tuần 2 thêm transformer (ViSoBERT trước, PhoBERT sau). `evaluate.py` (seen/unseen), `moderation.py` và demo cần dùng được mọi loại mô hình.
- **Quyết định:**
  - Config có trường `type: tfidf_lr | transformer`. `train.py` chọn nhánh theo trường này; `models.py` có `save_model`/`load_model` cho cả hai loại.
  - Code riêng của transformer nằm ở `src/transformer.py`, chỉ import khi cần, nên baseline không phụ thuộc torch.
  - Mọi mô hình có `predict(texts)` và `predict_proba(texts)`. Transformer được bọc trong `TransformerClassifier`.
  - Huấn luyện bằng `Trainer` của HuggingFace: đo dev mỗi epoch, giữ epoch có macro-F1 dev cao nhất.
  - Mất cân bằng nhãn: tùy chọn `class_weight: balanced` (cross-entropy có trọng số, cùng công thức với scikit-learn); so sánh ở phần 2b.
- **Lý do:** một đường đi duy nhất thì báo cáo cùng định dạng, và các module phía sau không cần biết bên trong là mô hình gì. Thứ tự ViSoBERT trước vì không cần tách từ (hạ tầng đơn giản hơn) và được huấn luyện trên dữ liệu mạng xã hội, gần với ViHSD.
- **Đánh đổi:** chọn epoch theo dev làm điểm dev hơi lạc quan; con số báo cáo cuối vẫn là test. Phương án bỏ: script riêng cho transformer (hai đường đi dễ lệch nhau); tự viết vòng lặp PyTorch (dễ sai, tốn thời gian).

## D-016. Dùng transformers 4.57.6, không dùng 5.x

- **Ngày:** 28/10/2026
- **Bối cảnh:** với transformers 5.21, tokenizer của `uitnlp/visobert` tách "mèo đáng yêu" thành `▁m èo ▁đ áng ▁y êu`, trong khi thư viện `sentencepiece` gốc tách thành `▁mèo ▁đáng ▁yêu` (các token này có trong từ vựng). Văn bản không dấu dạng NFD còn bị mất dấu. Nguyên nhân: lớp `XLMRobertaTokenizer` của bản 5.x cố định thuật toán Unigram, còn file của ViSoBERT là sentencepiece **BPE**. Bản 5.x cũng bỏ tokenizer chậm (`use_fast` bị bỏ qua), nên không có cách tránh.
- **Quyết định:** dùng transformers 4.57.6. Đã kiểm tra: cả tokenizer chậm và nhanh của bản này cho kết quả giống hệt `sentencepiece` gốc, mã token bằng mã sentencepiece + 1 (4 token đặc biệt đứng đầu, theo quy ước fairseq).
- **Lý do:** lỗi này không báo gì; mô hình vẫn chạy nhưng nhận đầu vào khác hẳn lúc được huấn luyện sẵn, dẫn tới kết luận sai rằng ViSoBERT kém.
- **Chặn tái diễn:** test `test_visobert_tokenizer_keeps_whole_vietnamese_syllables` (cần mạng, bật bằng `RUN_NETWORK_TESTS=1`), chạy trên Kaggle trước mỗi lần train và sau mỗi lần nâng cấp thư viện.
- **Đánh đổi:** không dùng được tính năng mới của 5.x. Phương án bỏ: ở lại 5.x và tự viết bộ chuyển đổi tokenizer BPE (nhiều code khó kiểm chứng).

## D-017. `max_length = 128` cho ViSoBERT

- **Ngày:** 30/10/2026
- **Bối cảnh:** đo số token (tokenizer ViSoBERT, kể cả token đặc biệt) trên 24036 câu **train** bằng `python -m src.transformer`: trung vị 11, phân vị 95% là 38, 99% là 69, dài nhất 7250. Tỷ lệ câu dài hơn mốc: 64 → 1.42%, 96 → 0.19%, 128 → 0.07%, 256 → 0.03%.
- **Quyết định:** `max_length = 128`.
- **Lý do:** chỉ khoảng 17 câu train bị cắt. Vì đệm theo từng batch (`DataCollatorWithPadding`), phần lớn batch vẫn ngắn, nên 128 chậm hơn 64 không đáng kể nhưng giữ nguyên được khoảng 320 câu. Lên 256 chỉ giữ thêm khoảng 10 câu.
- **Đánh đổi:** câu bị cắt mất phần cuối; nội dung độc hại nằm cuối một câu rất dài sẽ bị bỏ qua. Chỉ đo trên train, không nhìn dev/test.
