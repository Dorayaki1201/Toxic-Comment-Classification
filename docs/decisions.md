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

## D-018. Huấn luyện transformer trên đúng 1 GPU, ghi tên GPU vào báo cáo

- **Ngày:** 30/10/2026
- **Bối cảnh:** lần chạy thử `visobert_smoke` trên Kaggle (GPU T4 x2) chỉ có 8 bước train cho 500 câu với `batch_size: 32` (lẽ ra 16). `Trainer` tự dùng cả 2 GPU và đưa mỗi GPU 32 câu, nên batch thật là 64, khác với config, và kết quả phụ thuộc số GPU của máy.
- **Quyết định:** `train_transformer` báo lỗi nếu thấy hơn 1 GPU; trên Kaggle chạy với `CUDA_VISIBLE_DEVICES=0`. Báo cáo `dev_metrics.json` ghi thêm trường `device` (tên GPU).
- **Lý do:** config phải mô tả đúng thí nghiệm, và cùng config phải cho cùng thiết lập trên mọi máy. Với mô hình cỡ base, 2 GPU kiểu này không nhanh hơn đáng kể.
- **Đánh đổi:** không tận dụng GPU thứ hai. Phương án bỏ: chia `batch_size` cho số GPU (vẫn phụ thuộc phần cứng, khó giải thích hơn).

## D-019. So sánh loss thường và loss cân bằng nhãn cho ViSoBERT bằng 3 seed; quy tắc chọn đặt trước

- **Ngày:** 30/10/2026 (ghi **trước** khi chạy)
- **Bối cảnh:** ViSoBERT loss thường đạt macro-F1 dev 0.709 (seed 42). Chênh lệch giữa loss thường và loss cân bằng có thể chỉ khoảng 1 điểm, cỡ dao động do seed; chọn theo 1 seed dễ chọn nhầm sự may mắn.
- **Quyết định:**
  - Chạy `visobert_base` và `visobert_balanced` (chỉ khác `class_weight`), mỗi cấu hình 3 seed: 42, 43, 44 (`python -m src.train --seed`). Seed 42 được chạy lại dưới tên `_s42` để kiểm tra độ lặp lại so với lần chạy đầu.
  - Gộp kết quả bằng `python -m src.evaluate --summarize-seeds`: trung bình ± độ lệch chuẩn mẫu (ddof=1).
  - **Quy tắc chọn:** chọn cấu hình có trung bình macro-F1 dev cao hơn. Nếu hai trung bình chênh nhau ít hơn độ lệch chuẩn lớn hơn trong hai cấu hình, coi như hòa và chọn `visobert_base` (đơn giản hơn, không cần trọng số).
  - Sau khi chọn: chạy test **một lần** cho 3 mô hình của cấu hình được chọn, báo cáo trung bình ± độ lệch chuẩn của toàn bộ test và phần unseen. Cấu hình còn lại không chạy test.
- **Lý do:** đặt quy tắc trước khi thấy số để tránh chọn theo cảm tính; 3 seed là mức tối thiểu của CLAUDE.md mục 5. Một lần chạy khoảng 5 phút nên 6 lần vừa hạn mức GPU.
- **Đánh đổi:** 3 seed vẫn ít, độ lệch chuẩn ước lượng còn thô. Phương án bỏ: 1 seed mỗi cấu hình (dễ chọn nhầm); 6 file config riêng cho từng seed (dễ lệch nhau).
- **Kết quả trên dev (30/10/2026, ghi trước khi chạy test):** `visobert_base` 0.7121 ± 0.0046, `visobert_balanced` 0.7228 ± 0.0055. Chênh lệch 0.0107 lớn hơn độ lệch chuẩn lớn hơn (0.0055), nên không hòa: **chọn `visobert_balanced`**. Phần hơn chủ yếu ở F1 OFFENSIVE (0.5622 ± 0.0206 so với 0.5297 ± 0.0093); CLEAN và HATE gần như không đổi. Chạy lại `visobert_base` seed 42 cho đúng từng chữ số như lần đầu (0.709192...), xác nhận tái lập được trên cùng GPU.
- **Test (chạy một lần):** `visobert_balanced` 0.6940 ± 0.0035 (phần chưa gặp trong train 0.6945 ± 0.0024), so với baseline 0.643. Chi tiết: `reports/transformer_visobert.md`.

## D-020. Hoãn PhoBERT tới sau khi có bộ test nhiễu

- **Ngày:** 30/10/2026
- **Bối cảnh:** ViSoBERT đã xong (test 0.694 ± 0.004). PhoBERT không dùng trong sản phẩm cuối (demo, `moderation.py` dùng một mô hình); giá trị của nó chỉ nằm ở phép so sánh. PhoBERT cần tách từ (VnCoreNLP, chạy bằng Java) đúng như lúc tiền huấn luyện, thêm một chỗ có thể sai im lặng như D-016.
- **Quyết định:** hoãn PhoBERT, chuyển sang chuẩn hóa teencode (tuần 3) và bộ test nhiễu (tuần 4). Làm lại PhoBERT sau tuần 4 nếu còn thời gian, dưới dạng so sánh **độ bền trước nhiễu** giữa mô hình học trên văn bản chuẩn (PhoBERT) và văn bản mạng xã hội (ViSoBERT).
- **Lý do:** chuẩn hóa và bộ test nhiễu là điểm nhấn của dự án. So sánh PhoBERT với ViSoBERT trên test sạch nhiều khả năng lặp lại kết quả đã có trong bài báo ViSoBERT (cần kiểm tra lại); so sánh trên test nhiễu chỉ làm được sau tuần 4. Tiêu chí hoàn thành (CLAUDE.md mục 14) yêu cầu ít nhất một mô hình transformer, đã đạt.
- **Đánh đổi:** chưa có số so sánh hai mô hình tiền huấn luyện. Nếu thiếu thời gian, PhoBERT là mục bị cắt.

## D-021. Lớp luật của `normalize.py` (phần 3a): chỉ sửa ở chỗ chắc chắn là cố ý lách

- **Ngày:** 30/10/2026
- **Bối cảnh:** cấu hình B chuẩn hóa văn bản trước khi phân loại. Bình luận mạng có nhiều dạng "trông giống lách luật" nhưng là câu bình thường: thiếu khoảng trắng sau dấu câu ("ừ.thế", "hay!quá"), số tiền/giờ ("50k", "5h"), từ có hai chữ giống nhau liền nhau ("xoong"), đường link.
- **Quyết định:** nguyên tắc "thà bỏ sót còn hơn sửa sai". Thứ tự cố định: NFC → số thay chữ → ký tự chèn → ký tự kéo dài → khoảng trắng; đường link được giữ nguyên.
  - Số thay chữ: chỉ thay `0 1 3 4 $` khi nằm **giữa hai chữ cái** ("ch3t" → "chet"). Không thay "@" (phá email). Token chứa **f/j/w/z** (chữ không có trong bảng chữ cái tiếng Việt) bị bỏ qua: đó là viết tắt, tiếng Anh hoặc mã ("j4f", mã video).
  - Ký tự chèn `. - _ * !`: chỉ bỏ khi **một bên là phụ âm đứng một mình** ("đ.iên", "n.g.u", "c.h.ế.t"); "!" được hiểu là "i" ("d!t" → "dit"). Hai bên đều là âm tiết thật thì giữ ("ừ.thế").
  - Ký tự kéo dài: chữ cái lặp **từ 3 lần** rút về 1 ("điiiiên" → "điên"); lặp 2 lần giữ nguyên để không phá "xoong", "boong".
  - Không viết thường, không đổi emoji, không thay từ (từ điển teencode là phần 3b).
- **Lý do:** một luật sửa nhầm câu sạch làm hỏng cả những câu mô hình đang đoán đúng. Mỗi luật có test hai chiều ("phải sửa" và "không được đụng") và test chạy hai lần cho cùng kết quả.
- **Đánh đổi / hạn chế đã biết:** bỏ sót "đii" (lặp 2), "quáaaa" chỉ thành "quáa" ("á" và "a" là hai ký tự khác nhau), "đm.mày" giữ nguyên, số ở cuối từ ("ng0") không được thay, teencode dùng f/j/w/z mà có số thay chữ cũng bị bỏ qua. Tác động trên dữ liệu thật đo bằng `python -m src.normalize` trên train.
- **Sửa sau lần đo đầu trên train (30/10/2026):** đọc mẫu thấy luật số thay chữ biến viết tắt "j4f" thành "jaf" → thêm điều kiện f/j/w/z. Bản đo đầu còn áp từng luật lên cả câu, kể cả đường link, nên đếm sai (link YouTube bị tính là "đổi"); `normalize` thật vẫn giữ link. Công cụ đo giờ dùng chung `apply_outside_urls` với `normalize` để số đếm khớp hành vi thật.
- **Sửa sau khi chấm mẫu (31/10/2026, seed 42):** số thay chữ đúng 5/8, ký tự chèn đúng 7/10. Lỗi chung: luật áp lên thứ không phải một âm tiết tiếng Việt. Thêm 4 điều kiện:
  - số thay chữ bỏ qua token có **từ 2 chữ số** (mã như tên súng "M4A4");
  - số thay chữ bỏ qua token **dài hơn 7 chữ/số**, vì âm tiết dài nhất là "nghiêng" (câu viết liền có "1" là số thật);
  - gạch ngang chỉ bị bỏ khi có **từ 2 gạch** trong chuỗi ("n-g-u"); một gạch thường là tên riêng ("K-pop", tên ca sĩ dạng "K-…"), chấp nhận bỏ sót "n-gu";
  - kéo dài bỏ qua token **chỉ gồm một chữ lặp** ("kkkk" là tiếng cười, rút về "k" thành teencode của "không").
  - Vì các luật được sửa dựa trên chính các mẫu seed 42, precision báo cáo phải đo lại trên mẫu mới (seed khác).
- **Sửa sau khi chấm mẫu seed 7 (31/10/2026):** số thay chữ đúng 4/4, kéo dài đúng 9/10 (mẫu còn lại: tên riêng tiếng Anh bị kéo dài "Willl" thành "Wil", không đổi nhãn). Ký tự chèn có lỗi sai thật: **`*` là ký tự che, không phải ký tự chèn**. Nó đứng thay cho một chữ bị giấu, nên bỏ đi thì từ tục bị che thành một từ sạch khác nghĩa, xóa mất tín hiệu mà mô hình học được từ token có `*`. Cùng một ký tự có thể dùng để chèn ("n*g*u") hoặc che ("đ*o"), và hai loại sai có giá khác nhau: không gộp thì mô hình vẫn thấy nguyên văn, còn gộp sai thì mất hẳn tín hiệu. Vì vậy `-` và `*` chỉ được coi là ký tự chèn khi xuất hiện **từ 2 lần** trong chuỗi (chèn thì tách từng chữ, che thì thay một chữ). Khôi phục chữ bị che để cho từ điển (phần 3b); phần mơ hồ luật không giải quyết được là lý do có cấu hình C (seq2seq, nhìn ngữ cảnh) và D (huấn luyện đối kháng). Hạn chế chấp nhận thêm: viết tắt tên "Q.Anh" thành "QAnh" (không đổi nghĩa); từ tiếng Anh có chữ đôi bị kéo dài rút về 1 chữ.
- **Sửa sau khi chấm mẫu seed 2026 (31/10/2026):** ký tự chèn đổi 58 câu; trong 10 mẫu có 1 đúng, 9 không ảnh hưởng, 0 sai. Phần lớn là **thiếu dấu cách sau dấu chấm** cạnh một phụ âm đơn ("k.tết", "M.n"), vì giả định "phụ âm đơn không thể là một chữ" sai với teencode ("k" = không, "T" = tôi). Một dấu `.` hoặc `_` giờ chỉ được gộp khi nó tách phụ âm đầu khỏi phần vần (bên trái là phụ âm đơn, bên phải bắt đầu bằng nguyên âm: "đ.iên"). Điều kiện mới hẹp hơn điều kiện cũ (chỉ gộp ít đi, không gộp thêm chỗ nào), nên không thể sinh lỗi "sai" mới so với lần chấm 0/10 sai này; không chấm thêm vòng nữa.
- **Thay đổi sau này:** việc khôi phục chữ bị `*` che không được làm ở 3b (xem D-022).

## D-022. Từ điển teencode (phần 3b): xây từ train, chấm tay theo ngưỡng đặt trước, không dùng để sinh nhiễu

- **Ngày:** 01–03/11/2026
- **Bối cảnh:** lớp luật 3a chỉ đổi 2.33% câu train; phần lớn biến thể trong bình luận là viết tắt và teencode ("ko", "đc", "vl"). Một token viết tắt có thể có nhiều nghĩa ("m" = mày/mình, "vc" = vãi cả/Việt Cộng/vợ chồng), và thay sai còn tệ hơn không thay: văn bản bị đổi nghĩa, có khi xóa đúng tín hiệu độc hại.
- **Quyết định:**
  - **Ứng viên** (`find_lexicon_candidates` trong `src/keywords.py`, chỉ trên train đã qua lớp luật 3a): token chỉ gồm chữ cái, có trong **ít nhất 10 câu train**, và không có nguyên âm ("đc") hoặc có chữ f/j/w/z ("loz"); cộng một danh sách gõ tay 17 token teencode có nguyên âm ("ko", "lun", "iu"...). Kết quả: **167 ứng viên**.
  - **Chấm:** mỗi ứng viên đọc 10 câu ngẫu nhiên (seed 42). Quy tắc đặt trước: `keep` nếu **ít nhất 9/10 câu cùng một nghĩa** và có dạng đầy đủ tiếng Việt; ngược lại `reject` kèm lý do. Vòng đầu do Claude chấm, người dùng duyệt lại mục `keep`. Kết quả: **83 keep, 84 reject**, lưu ở `resources/teencode.tsv` (có commit, có cột `note` ghi lý do).
  - **Thay thế** (`replace_teencode`, gọi qua `normalize(text, lexicon)`): chạy **sau** lớp luật 3a, chỉ thay nguyên token, tra không phân biệt hoa/thường, giữ chữ hoa đầu. Không thay token trong link, dính số ("5h"), đứng sau `: ; = ) ] . / @ -` (mặt cười ":v", tên miền "abc.vn", tag) hoặc có `-` ngay sau ("K-ICM"). `load_lexicon` từ chối file có dạng đầy đủ chứa một token khác của từ điển, để `normalize` luôn idempotent.
  - `normalize(text)` không truyền từ điển thì chỉ chạy 3a: công cụ tìm ứng viên cần thấy "ko" nguyên dạng.
  - **Không khôi phục chữ bị `*` che** ("x*u"): chỉ khoảng 0.2% câu train có dạng này, và một chữ bị che có nhiều cách điền. Thay đổi so với dự định ghi trong D-021.
  - **`noise.py` không được dùng file từ điển này để sinh nhiễu.** Nếu bộ test nhiễu được tạo bằng cách đảo ngược chính từ điển (thay "không" thành "ko"), cấu hình B sẽ khôi phục được gần như mọi chỗ, và điểm của B trên test nhiễu chỉ đo việc từ điển khớp với chính nó (đánh giá vòng tròn). Bảng biến đổi của `noise.py` phải được xây độc lập.
- **Lý do:** ngưỡng 10 câu giúp có đủ mẫu để đọc, và token hiếm hầu như không ảnh hưởng mô hình. Ngưỡng 9/10 theo nguyên tắc "thà bỏ sót còn hơn sửa sai" của D-021: reject nhầm thì token giữ nguyên như khi chưa có từ điển, keep nhầm thì văn bản bị đổi nghĩa. Thay theo ngữ cảnh ("t" trước động từ → "tôi") để cho cấu hình C.
- **Đo trên train** (`python -m src.normalize --lexicon --samples 100 --seed <seed>`). Precision tính trên **từng lần thay** lấy ngẫu nhiên, nên token thay nhiều có trọng số lớn đúng như tác động thật lên dữ liệu; không chắc nghĩa thì tính sai. Không dùng cột `agree` (đo trên chính mẫu seed 42 dùng để chọn, nên cao hơn thực tế).
  - **Seed 2026, bản đầu:** đổi 5711/24036 câu (23.76%), 8010 lần thay; precision **96/100**, Wilson 95% [0.902, 0.984]. Lỗi: "K-ICM" (tên ca sĩ) ×2, "vài trăm k" (nghìn), "cs" = ca sĩ.
  - **Sửa:** không thay token có `-` dính hai bên (38 lần thay trên train, đếm trước khi sửa; sau khi sửa số lần thay giảm đúng 38).
  - **Seed 2027, bản đã sửa** (seed chọn trước khi đo): đổi 5678 câu (23.62%), 7972 lần thay; precision **98/100**, Wilson 95% [0.930, 0.994]. Lỗi còn lại là nhập nhằng ngữ cảnh ("chj" trong câu không dấu, "bi k"). Hai khoảng tin cậy chồng nhau, nên không kết luận bản sửa tốt hơn về precision; chỉ kết luận đã bỏ được một nhóm lỗi đã biết.
  - Tỉ lệ câu bị đổi theo nhãn (seed 2027): CLEAN 20.8%, OFFENSIVE 38.1%, HATE 36.3%. Chuẩn hóa tác động nhiều nhất đúng ở nhóm câu độc hại.
  - Bảng chấm (chỉ số dòng, token, đúng/sai; không chép câu): `reports/teencode/precision_seed2026.tsv`, `precision_seed2027.tsv`.
- **Đánh đổi / hạn chế đã biết:** bỏ các token phổ biến nhưng nhập nhằng ("v" 693 câu, phần lớn là ":v"; "t", "m"); "k" vẫn có thể là "nghìn", "cs" có thể là "ca sĩ", "bít" phá từ thật "bưng bít", "KO" viết hoa (knock-out) thành "Không"; chữ cái đơn ("H") có thể là chữ viết tắt tên người; "thế.ko" (thiếu dấu cách) không được thay. Từ điển chỉ phản ánh teencode xuất hiện trong ViHSD train.

## D-023. Sinh nhiễu (`noise.py`): mô phỏng người viết thật, mỗi từ một biến đổi, `rate` = tỉ lệ từ bị đổi

- **Ngày:** 04/11/2026
- **Bối cảnh:** bộ test nhiễu đo độ bền của A/B/C/D; cấu hình D dùng nhiễu để tăng cường train. Ngoài vòng tròn với từ điển (D-022), lớp luật 3a cũng có rủi ro tương tự: nhiễu chỉ sinh đúng dạng luật biết gỡ ("n.g.u", số giữa hai chữ) thì B thắng chỉ vì gặp lại dạng nó được viết để gỡ.
- **Quyết định:**
  - 5 biến đổi theo ký tự, không cần bảng từ: `teencode_chars` (ph→f, qu→w, gi→j, gì→j, c→k trừ "ch", v→z, "i" cuối không dấu→j), `remove_diacritics` (từng từ), `elongate` (một nguyên âm lặp 2–5 lần), `substitute_digits` (a/e/i/o/s → 4/3/1/0/$, ở bất kỳ vị trí nào), `insert_chars` (một loại `. - _ *`, vào mọi khe hoặc một khe). Có cả dạng lớp luật cố ý bỏ qua (D-021): lặp 2 lần, số ở cuối từ, một dấu `-`, chữ "f"/"w"/"j"/"z".
  - **Mỗi từ bị chọn với xác suất `rate` và chịu đúng một biến đổi**, chọn ngẫu nhiên trong các biến đổi áp được lên từ đó. Mức: light 0.1, medium 0.3, heavy 0.5 (`configs/noise.yaml`).
  - Seed theo từng câu: `random.Random(f"{seed}-{i}")`. Đổi một câu không làm đổi nhiễu của câu khác; giá trị đầu của `rng_for(42, 0)` được khóa trong test.
  - Không xét nhãn khi chọn từ (ngẫu nhiên đều): nhiễu không mang thông tin nhãn.
  - **Hoãn "đổi emoji"** (CLAUDE.md mục 4.4): cần một bảng từ → emoji mang tính ngữ nghĩa, tùy ý, khó bảo vệ.
  - **Viết tắt theo bảng** (`abbreviate`): làm sau, xem D-024.
- **Sửa sau khi đọc mẫu train (04/11/2026, seed 42):** bản đầu cho mỗi biến đổi chạy độc lập với cùng `rate`, nên khi ghép 5 biến đổi xác suất một từ bị đổi là 1 − (1 − rate)^5 (≈ 41% / 83% / 97%), và một từ có thể bị chồng nhiều biến đổi ("w.o.o.o.o.o.c"). Sau khi sửa, `python -m src.noise` trên 2000 câu train đo được khoảng 8.9% / 27.2% / 46.4% từ bị đổi ở light/medium/heavy (đếm xấp xỉ). Test `test_rate_is_fraction_of_words_changed` và `test_each_word_gets_only_one_transform` khóa hành vi này.
- **Lý do:** một nghĩa duy nhất cho `rate` giúp so sánh được giữa bộ từng biến đổi và bộ ghép. Mỗi từ một biến đổi giữ được câu đọc được, gần với cách người dùng thật lách (che vài từ nhạy cảm, không phá cả câu).
- **Đánh đổi / rủi ro:** không sinh dạng ghép trên một từ ("đ.1.t"). Chọn từ ngẫu nhiên đều, không ưu tiên từ độc hại như người lách thật (để sau, có thể làm mức "targeted" ở tuần 6). `elongate` có thể kéo dài nguyên âm đầu từ ("aaassmin"), ít gặp ngoài đời. Cấu hình D (tuần 5) train trên nhiễu cùng bộ sinh với test nhiễu nên cũng có "lợi thế sân nhà"; cách giảm (ví dụ giữ riêng một loại biến đổi) sẽ quyết định ở tuần 5. Seed của bộ test nhiễu sẽ được ghi ở đây trước khi tạo.

## D-024. Biến đổi viết tắt: bảng rút tự động từ ViLexNorm train, không sửa tay

- **Ngày:** 04/11/2026
- **Bối cảnh:** `noise.py` cần biến đổi viết tắt ("không" → "ko") nhưng không được dùng `teencode.tsv` (D-022). Một bảng do người làm dự án tự viết vẫn phụ thuộc vào cùng một người đã chấm từ điển. ViLexNorm (Nguyen et al., EACL 2024) có 10.467 cặp câu (gốc, chuẩn hóa) do người khác gán, từ Facebook và TikTok; giấy phép CC BY-NC-SA 4.0, chỉ cho nghiên cứu (`data/README.md`).
- **Quyết định:**
  - Chỉ dùng **split train** (8372 cặp). Dev/test để dành đo ERR của `normalize.py`.
  - Tách hai câu theo dấu cách (bỏ dấu câu hai đầu), gióng hàng bằng `difflib.SequenceMatcher`, chỉ giữ chỗ thay **1 từ ↔ 1 từ** mà cả hai bên chỉ gồm chữ cái. Bỏ 1-n/n-1 (dễ gióng sai; ViLexNorm không gán viết tắt cụm từ).
  - Bỏ cặp mà 5 biến đổi theo ký tự tự sinh được (so sau khi viết thường, đổi số về chữ, bỏ dấu, gộp chữ lặp; và thử từng luật teencode), để mỗi loại nhiễu chỉ nằm ở một biến đổi. Rút chữ lặp từ 3 lần trước khi đếm ("lunnn" cộng vào "lun").
  - Giữ cặp gặp **ít nhất 3 lần**; `abbreviate` chọn dạng viết tắt theo tần suất. Bảng được commit ở `resources/noise_abbrev.tsv` với giấy phép CC BY-NC-SA 4.0 riêng; **không sửa tay**.
- **Kết quả** (`python -m src.noise --build-abbrev ... --min-count 3`, seed 42 cho phép xáo khi đo độ bão hòa): 12.276 cặp từ sau lọc; bảng có **207 từ chuẩn, 424 mục** (ngưỡng 2: 610 mục, ngưỡng 5: 277 mục).
  - **Độ bão hòa:** xây trên 25/50/75/100% train được 152/261/346/424 mục; cả 50 mục phổ biến nhất của bảng đầy đủ đều đã xuất hiện từ mức 25%. Phần phổ biến đã ổn định, phần đuôi hiếm vẫn còn tăng: thêm dữ liệu chỉ thêm dạng hiếm. Không tự sinh thêm vì làm mất tính độc lập.
  - **Trùng với `teencode.tsv`:** 40/424 mục có dạng viết tắt nằm trong mục `keep` (35 mục trùng cả cặp), nhưng chiếm **45.5% theo tần suất**. Nghĩa là khoảng một nửa số lần `abbreviate` sinh ra là dạng mà từ điển của B gỡ được ("ko", "đc"); nửa còn lại B không gỡ được, kể cả dạng rất phổ biến mà D-022 đã cố ý loại vì nhập nhằng ("t" = tôi/tao, "m" = mày/mình). Phải báo con số này kèm điểm của B trên bộ `abbreviate`.
  - **Trùng câu với ViHSD:** 0 câu (train/dev/test), so sau NFC, viết thường, gộp khoảng trắng.
  - Xem thử trên 2000 câu ViHSD train (seed 42): `abbreviate` @ medium đổi khoảng 9% số từ (chỉ từ có trong bảng mới đổi được).
- **Lý do:** bảng từ nguồn độc lập, gán bởi người khác, dễ bảo vệ hơn bảng tự viết. Không sửa tay để bảng sinh lại được đúng từng dòng từ dữ liệu gốc.
- **Đánh đổi / hạn chế đã biết:** đọc 40 mục ngẫu nhiên có tần suất ≤ 4: 1 mục rõ là lỗi gõ ("cubgx"); có chơi chữ phụ thuộc ngữ cảnh ("bình" → "pink" từ "tide pink" = "thái bình", áp lên tên "Bình" thì không tự nhiên); một số biến thể vùng miền (n/l: "nên" → "lên"). `abbreviate` áp theo từng từ, không xét ngữ cảnh. Bảng phản ánh teencode của Facebook/TikTok trong thời gian ViLexNorm thu thập.

## D-025. Bộ test nhiễu: 9 bộ từ ViHSD test, seed 1105, mỗi bộ một luồng ngẫu nhiên riêng, `noise.py` đóng băng

- **Ngày:** 05/11/2026 (seed ghi **trước** khi tạo bộ)
- **Bối cảnh:** cần bộ test nhiễu cố định để đo độ bền của A/B/C/D (CLAUDE.md mục 5). Bộ sinh nhiễu đã xong (D-023, D-024). Dữ liệu sinh ra từ ViHSD nên không được commit, nhưng vẫn phải tái lập được đúng từng byte trên máy khác (Kaggle).
- **Quyết định:**
  - **9 bộ** từ `load_vihsd("test")` (6680 dòng): 6 bộ một biến đổi ở mức medium (`<biến đổi>_medium`, rate 0.3) và 3 bộ ghép cả 6 biến đổi (`all_light`, `all_medium`, `all_heavy`). Giữ đúng số dòng, thứ tự và nhãn gốc; cột `row_id` trỏ về vị trí trong test gốc, để seen/unseen (D-014) được tính theo câu gốc.
  - **Seed 1105**, chưa từng dùng ở bước nào trước đó. Câu i của bộ `<tên>` dùng `rng_for(1105, "<tên>-<i>")`: các bộ độc lập với nhau, không tương quan một phần như khi dùng chung `rng_for(seed, i)`.
  - **Cấu hình D (tuần 5) phải dùng seed khác 1105** khi sinh nhiễu tăng cường: khoảng 805 câu test trùng với train (D-014), dùng chung seed và khóa thì D có thể học đúng dạng nhiễu sẽ gặp lúc test.
  - **Đối chứng NFC:** `perturb` đưa cả câu về NFC, còn `transformer.py` không tự làm việc này (baseline thì có, `models.py`). Nếu có câu test mà NFC làm thay đổi, tạo thêm bộ `nfc_only` (rate 0) để tách ảnh hưởng của NFC khỏi ảnh hưởng của nhiễu.
  - **`noise.py` đóng băng** từ lần tạo bộ đầu tiên. Lệnh tạo chỉ in thống kê (% câu, % từ bị đổi), không in câu test. Không sửa bộ sinh nhiễu dựa trên những gì thấy ở test nhiễu; nếu buộc phải sửa thì ghi quyết định mới và tạo phiên bản bộ mới.
  - **Không commit dữ liệu, commit manifest** `reports/noisy_test/manifest.json`: seed, mức và biến đổi, sha256 của `noise.py` và `noise_abbrev.tsv` (sau khi đổi CRLF thành LF), và cho từng bộ: số câu đổi, % câu, % từ đổi, sha256 của CSV. `python -m src.noise --make-test-sets --check` sinh lại trong bộ nhớ và so hash.
- **Kết quả tạo bộ (05/11/2026, `python -m src.noise --make-test-sets`, sau commit ghi seed):** `--check` khớp manifest. **141/6680 câu test thay đổi khi đưa về NFC**, nên có thêm bộ đối chứng `nfc_only` (tổng 10 bộ). Với transformer, mức tụt phải tính so với điểm trên `nfc_only`, không phải test gốc. Tỉ lệ câu / từ bị đổi (từ đếm xấp xỉ):

  | Bộ | rate | % câu đổi | % từ đổi |
  |---|---|---|---|
  | abbreviate_medium | 0.3 | 52.96 | 8.69 |
  | teencode_chars_medium | 0.3 | 51.38 | 7.73 |
  | remove_diacritics_medium | 0.3 | 74.31 | 19.93 |
  | elongate_medium | 0.3 | 84.64 | 25.95 |
  | substitute_digits_medium | 0.3 | 82.98 | 23.93 |
  | insert_chars_medium | 0.3 | 84.93 | 26.01 |
  | all_light | 0.1 | 55.34 | 8.57 |
  | all_medium | 0.3 | 85.36 | 26.68 |
  | all_heavy | 0.5 | 94.04 | 45.81 |

  Các bộ một biến đổi có % từ đổi khác nhau vì không phải từ nào cũng áp được mọi biến đổi (`abbreviate` chỉ đổi từ có trong bảng, `teencode_chars` chỉ đổi từ khớp luật). Khi so mức tụt giữa các bộ phải đọc kèm cột này.
- **Lý do:** seed ghi trước khi tạo thì không thể chọn seed "đẹp" sau khi thấy kết quả. Luồng riêng mỗi bộ cho một câu trả lời đơn giản khi bị hỏi ("các bộ độc lập"). Manifest có hash thay cho việc commit dữ liệu.
- **Đánh đổi:** chỉ một seed nhiễu nên chưa biết phương sai do nhiễu (có thể tạo thêm 2 seed cho `all_medium` ở tuần 6). Các mức không lồng nhau (từ bị đổi ở light không chắc nằm trong medium). Giả định nhiễu không đổi nhãn, dựa trên đọc mẫu train (D-023). Phương án bỏ: dùng chung `rng_for(seed, i)` cho mọi bộ; commit CSV.

## D-026. Đo mức tụt của cấu hình A trên bộ test nhiễu; quy tắc ghi trước khi chạy

- **Ngày:** 06/11/2026 (ghi **trước** mọi lần chạy trên bộ nhiễu, kể cả baseline trên máy)
- **Bối cảnh:** bộ test nhiễu đã có (D-025). Cần hàng A của bảng A/B/C/D: `tfidf_lr_wordchar_balanced` (tất định, D-013) và `visobert_balanced` seed 42/43/44 (D-019). Mô hình ViSoBERT **không còn** (output Kaggle trống, chưa từng Save Version), nên phải tái tạo.
- **Quyết định (cách đo):**
  - `python -m src.evaluate --config X --noisy-test [--seed S]` đoán trên test gốc (`original`) và mọi bộ trong manifest. Kết quả ghi vào `reports/<tên>/noisy_test_metrics.json` và `noisy_test_predictions.csv` (chỉ `row_id` + nhãn đoán, không có văn bản). **Không ghi đè `test_metrics.json`.**
  - **Tham chiếu để tính mức tụt:** ViSoBERT so với `nfc_only` (transformer không tự NFC, D-025), baseline so với `original`. Mức tụt = điểm tham chiếu − điểm bộ nhiễu, tính bằng điểm macro-F1 tuyệt đối. Với 3 seed: tính mức tụt trong từng seed rồi mới gộp trung bình ± độ lệch chuẩn mẫu.
  - Mỗi bộ báo: macro-F1, F1 từng lớp, confusion matrix; seen/unseen theo **câu gốc** qua `row_id` (D-014); chỉ số trên phần câu bị đổi (`changed`) so với chính những câu đó ở bộ tham chiếu; tỉ lệ đổi nhãn dự đoán so với bộ tham chiếu.
  - **Kiểm tra tự động:** sha256 từng CSV phải khớp manifest đã commit; `row_id` = 0..n−1 và nhãn trùng test gốc; baseline phải đoán giống hệt trên `nfc_only` và `original`; điểm `original` đoán lại được so với `test_metrics.json` đã commit (ghi `matches` vào báo cáo).
- **Quyết định (tái tạo ViSoBERT):**
  - Train lại `visobert_balanced` seed 42/43/44 với **đúng** config đó. Code huấn luyện (`transformer.py`, `train.py`, `models.py`, `data.py`, `utils.py`), config và `requirements.txt` không đổi từ commit `9131a36` sinh ra kết quả test. Trọng số `uitnlp/visobert` không đổi: commit mới nhất `196a62afad9c` (09/06/2024, chỉ sửa README), kiểm tra qua API HuggingFace ngày 06/11/2026. Notebook Kaggle cố định môi trường cũ (pin original environment), GPU T4.
  - Ba cổng kiểm tra: (1) phiên bản torch/transformers/GPU so với `dev_metrics.json` cũ (2.11.0+cu128, 4.57.6, Tesla T4); (2) `git diff` 3 file `dev_metrics.json` sau khi train (không chạm test); (3) điểm `original` so với `test_metrics.json` đã commit.
  - **Không chạy `--split test`** (sẽ ghi đè mốc đối chiếu).
  - **Quy tắc nếu số không khớp:** không thử seed khác, môi trường khác hay chạy lại nhiều lần để "tìm" bản khớp; không chọn lại cấu hình (đã chọn ở D-019). Dùng 3 mô hình mới; giữ nguyên các file `dev_metrics.json`/`test_metrics.json` cũ (cơ sở của D-019) và ghi số dev, test sạch của bộ mô hình mới vào mục kết quả ở đây; báo cả số cũ 0.694 lẫn số mới. Mức tụt vẫn hợp lệ vì được so trong cùng một mô hình. Con số trong CV khi đó lấy theo bộ mô hình mới.
  - Lần này giữ mô hình: chạy bằng Save Version → Save & Run All để `/kaggle/working` (gồm `models/`) được lưu làm output (`docs/kaggle.md`).
- **Có vi phạm D-011 không:** không. D-011 cấm chọn hoặc chỉnh mô hình theo test. Ở đây mô hình đã đóng băng, không có lựa chọn nào phụ thuộc kết quả, cùng loại với lần chạy lại ở D-014. Cổng (3) chứng minh mô hình đo là mô hình đã báo cáo.
- **Rủi ro mềm và cam kết:** biết A yếu ở biến đổi nào có thể khiến ta vô tình chỉnh B, C, D theo đúng điểm yếu đó, biến test nhiễu thành dev. Vì vậy:
  1. Kết quả chỉ dùng để báo cáo và phân tích; không đổi mô hình, `noise.py`, bộ nhiễu hay ngưỡng nào theo kết quả.
  2. Từ nay mọi thay đổi ở `normalize.py` và mọi lựa chọn cho C/D phải dựa trên train/dev, kèm bằng chứng từ train/dev ghi ở đây.
  3. Tuần 5 tạo **bộ dev nhiễu** (từ dev, seed khác 1105 và khác seed tăng cường của D) để chọn siêu tham số cho C/D.
- **Giới hạn đã biết khi đọc kết quả:** một seed nhiễu; các bộ có % từ bị đổi khác nhau (bảng D-025), nên so giữa các bộ phải đọc kèm cột đó; chưa có khoảng tin cậy (tuần 6, dùng `noisy_test_predictions.csv` để bootstrap/so cặp); phần câu bị đổi là tập con theo nhiễu, không theo nhãn.
- **Phương án bỏ:** một tham chiếu chung cho cả hai mô hình (trộn ảnh hưởng của NFC vào ViSoBERT); lấy trung bình điểm qua seed rồi mới trừ (phương sai lớn hơn); tính seen trên câu nhiễu (gần như mọi câu thành unseen).
- **Kết quả:** TODO (điền sau khi chạy, chi tiết ở `reports/noisy_test/config_a.md`).
