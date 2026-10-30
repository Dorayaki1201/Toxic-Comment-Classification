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
# Ô 3: trỏ dữ liệu vào đúng chỗ config mong đợi (data/raw/vihsd), không sửa config
# Đường dẫn input có thể khác; xem ở cột phải và sửa cho khớp.
!mkdir -p data/raw && ln -s /kaggle/input/vihsd data/raw/vihsd && ls data/raw/vihsd
```

```bash
# Ô 4: kiểm tra tokenizer không bị lỗi tách token (D-016)
!RUN_NETWORK_TESTS=1 python -m pytest -q -m network
```

```bash
# Ô 5: chạy thử nhanh (500 câu, 1 epoch) để kiểm tra pipeline
!python -m src.train --config configs/visobert_smoke.yaml
```

```bash
# Ô 6: chạy thật (chỉ sau khi ô 5 ổn)
!python -m src.train --config configs/visobert_base.yaml
```

## 4. Lấy kết quả về máy

- File cần lấy: `reports/<tên>/dev_metrics.json` (nhỏ, commit vào repo như baseline).
- Mô hình ở `models/<tên>/` (lớn, **không commit**). Để giữ lại sau khi phiên kết thúc: **Save Version** → **Save & Run All**, khi đó thư mục `/kaggle/working` được lưu làm output của notebook.
- Ghi lại: loại GPU, thời gian chạy, phiên bản torch/transformers (đã có trong `dev_metrics.json`).
