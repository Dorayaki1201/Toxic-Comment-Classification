# Chạy huấn luyện transformer trên Kaggle

Code nằm trên GitHub, dữ liệu nằm trong một Kaggle Dataset **riêng tư**, notebook Kaggle chỉ clone code, cài thư viện và gọi đúng các lệnh như trên máy. Không viết logic trong notebook (CLAUDE.md mục 9).

> Hạn mức GPU, tên GPU và giao diện Kaggle thay đổi theo thời gian; kiểm tra trên trang Kaggle khi làm.

## 1. Tạo Kaggle Dataset riêng tư cho ViHSD (làm một lần)

1. Kaggle → **Datasets** → **New Dataset**.
2. Tải lên 3 file `train.csv`, `dev.csv`, `test.csv` từ `data/raw/vihsd/` trên máy.
3. Đặt tên (ví dụ `vihsd`), để **Private**. Không công khai lại dữ liệu (xem giấy phép trong `data/README.md`).

## 2. Tạo notebook

1. **Code** → **New Notebook**.
2. Cột phải: **Add Input** → chọn dataset `vihsd` vừa tạo.
3. **Settings** → **Accelerator**: chọn GPU; bật **Internet** (để clone code và tải mô hình).

## 3. Các ô lệnh

```bash
# Ô 1: lấy code (đổi tên nhánh nếu cần)
!git clone --branch feat/transformer-visobert https://github.com/Dorayaki1201/Toxic-Comment-Classification.git repo
%cd repo
!git log --oneline -1
```

```bash
# Ô 2: cài đúng phiên bản (torch dùng bản GPU có sẵn của Kaggle)
!pip install -q "transformers==4.57.6" "accelerate==1.15.0" "sentencepiece==0.2.2"
!python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

```bash
# Ô 3: trỏ dữ liệu vào đúng chỗ config mong đợi (data/raw/vihsd), không sửa config.
# Kaggle đổi cấu trúc /kaggle/input theo thời gian, nên tìm train.csv thay vì đoán đường dẫn.
# Kết quả đúng: dòng cuối in ra dev.csv test.csv train.csv (nếu chỉ in "data/raw/vihsd" là liên kết hỏng).
!find /kaggle/input -name "train.csv"
!mkdir -p data/raw && rm -f data/raw/vihsd && ln -s "$(dirname "$(find /kaggle/input -name train.csv | head -1)")" data/raw/vihsd && ls data/raw/vihsd
```

```bash
# Ô 4: kiểm tra tokenizer không bị lỗi tách token (D-016)
!RUN_NETWORK_TESTS=1 python -m pytest -q -m network
```

Ô 5 và ô 6 chỉ dùng **1 GPU** (`CUDA_VISIBLE_DEVICES=0`): với nhiều GPU, `Trainer` chia batch cho từng GPU nên batch thật bị nhân lên mà không báo; code sẽ dừng nếu thấy hơn 1 GPU (D-018).

```bash
# Ô 5: chạy thử nhanh (500 câu, 1 epoch) để kiểm tra pipeline
!CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/visobert_smoke.yaml
```

```bash
# Ô 6: chạy thật (chỉ sau khi ô 5 ổn)
!CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/visobert_base.yaml
```

### Chạy nhiều seed (phần 2b, D-019)

```bash
# Lấy code của nhánh mới (nếu notebook đã clone từ trước)
!git fetch origin && git checkout feat/visobert-seeds && git pull && git log --oneline -1
```

```bash
# 2 cấu hình x 3 seed = 6 lần, mỗi lần khoảng 6 phút; dừng ngay nếu một lần lỗi
!for s in 42 43 44; do for c in visobert_base visobert_balanced; do CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/$c.yaml --seed $s || exit 1; done; done
```

```bash
# Gộp trung bình ± độ lệch chuẩn trên dev (chỉ đọc báo cáo, không nạp mô hình)
!python -m src.evaluate --config configs/visobert_base.yaml --split dev --summarize-seeds 42 43 44
!python -m src.evaluate --config configs/visobert_balanced.yaml --split dev --summarize-seeds 42 43 44
```

```bash
# Đóng gói báo cáo để tải về (Output -> /kaggle/working/repo/reports.zip)
!zip -qr reports.zip reports && ls -lh reports.zip
```

Giữ phiên mở cho tới khi chọn xong cấu hình: bước chạy test cần 3 mô hình trong `models/`. Nếu phiên mất, train lại 3 seed của cấu hình được chọn.

Ghi chú môi trường (lần chạy 30/10/2026): Kaggle dùng Python 3.13, torch 2.11.0+cu128. Ô 2 in cảnh báo xung đột `huggingface-hub` với `gradio`/`diffusers` cài sẵn: vô hại khi train vì không dùng hai thư viện này. **Rủi ro cho tuần 7:** `gradio` bản mới cần `huggingface-hub >= 1.16`, còn `transformers` 4.57 cần `< 1.0`; cần kiểm tra khi làm demo.

## 4. Lấy kết quả về máy

- File cần lấy: `reports/<tên>/dev_metrics.json` (nhỏ, commit vào repo như baseline).
- Mô hình ở `models/<tên>/` (lớn, **không commit**). Để giữ lại sau khi phiên kết thúc: **Save Version** → **Save & Run All**, khi đó thư mục `/kaggle/working` được lưu làm output của notebook.
- Ghi lại: loại GPU, thời gian chạy, phiên bản torch/transformers (đã có trong `dev_metrics.json`).

## 5. Tái tạo ViSoBERT + đánh giá trên bộ test nhiễu (D-026)

Chạy **một lần duy nhất** bằng **Save Version → Save & Run All**, để `/kaggle/working` (gồm `models/`) được lưu làm output. Trước đó: notebook đã cố định môi trường cũ, Accelerator **GPU T4**, Internet bật, dataset `vihsd` đã thêm làm Input, và nhánh `feat/noisy-eval-a` (đã có commit D-026) đã được push.

Mọi ô dùng hàm `run` thay cho `!`. Lý do: `!lệnh` lỗi **không** làm notebook dừng, trong khi Save & Run All cần dừng ngay khi một bước kiểm tra sai. `subprocess.run(..., check=True)` báo lỗi và dừng.

```python
# Ô 1: hàm chạy lệnh (dừng nếu lỗi) + lấy code
import os, subprocess


def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


run(
    "git clone --branch feat/noisy-eval-a https://github.com/Dorayaki1201/Toxic-Comment-Classification.git repo"
)
os.chdir("repo")
run("git log --oneline -1")
```

```python
# Ô 2: thư viện + cổng 1 (so với dev_metrics.json cũ: 2.11.0+cu128, 4.57.6, Tesla T4)
run('pip install -q "transformers==4.57.6" "accelerate==1.15.0" "sentencepiece==0.2.2"')
run(
    'python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"'
)
```

```python
# Ô 3: dữ liệu + test tokenizer (D-016)
run(
    'mkdir -p data/raw && rm -f data/raw/vihsd && ln -s "$(dirname "$(find /kaggle/input -name train.csv | head -1)")" data/raw/vihsd && ls data/raw/vihsd'
)
run("RUN_NETWORK_TESTS=1 python -m pytest -q -m network")
```

```python
# Ô 4: tạo bộ nhiễu, so với manifest ĐÃ COMMIT (lệnh tạo ghi đè manifest, nên phải git diff), rồi --check
run("python -m src.noise --make-test-sets")
run("git diff --exit-code -- reports/noisy_test/manifest.json")
run("python -m src.noise --make-test-sets --check")
```

```python
# Ô 5: tái tạo 3 mô hình, đúng config cũ (khoảng 3 x 6 phút)
for s in (42, 43, 44):
    run(
        f"CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/visobert_balanced.yaml --seed {s}"
    )
```

```python
# Ô 6: cổng 2 (dev, chưa chạm test): rỗng = khớp tuyệt đối với mô hình cũ.
# Không dừng nếu khác: theo quy tắc D-026, vẫn dùng mô hình mới và báo cả hai số.
dev_files = " ".join(f"reports/visobert_balanced_s{s}/dev_metrics.json" for s in (42, 43, 44))
run(f"git diff --stat -- {dev_files} | tee reports/noisy_test/kaggle_dev_diff.txt")
```

```python
# Ô 7: bộ test nhiễu (cổng 3 nằm trong báo cáo: original_matches_committed_test), gộp 3 seed
for s in (42, 43, 44):
    run(
        f"CUDA_VISIBLE_DEVICES=0 python -m src.evaluate --config configs/visobert_balanced.yaml --noisy-test --seed {s}"
    )
run(
    "python -m src.evaluate --config configs/visobert_balanced.yaml --noisy-test --summarize-seeds 42 43 44"
)
run("zip -qr reports.zip reports && ls -lh reports.zip")
```

Tải về: `reports.zip` (giải nén đè lên `reports/` ở máy, rồi `git status` / `git diff` để xem file nào đổi). `models/visobert_balanced_s4x/` ở lại trong output của version này. Ở lần sau, thêm output đó làm Input thì không phải train lại.
