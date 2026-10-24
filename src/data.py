"""Nạp dữ liệu ViHSD, kiểm tra định dạng và xử lý dòng lỗi.

Không sửa nội dung bình luận (viết thường, bỏ dấu, teencode...)
"""

from pathlib import Path

import pandas as pd

from src.utils import get_logger

LABEL_NAMES: dict[int, str] = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
TEXT_COL = "free_text"
LABEL_COL = "label_id"
SPLITS = ("train", "dev", "test")
ERROR_MARKER = "#ERROR!"

logger = get_logger(__name__)


def find_invalid_rows(df: pd.DataFrame) -> pd.Series:
    """Trả về mask True ở những dòng rỗng hoặc bằng #ERROR!"""
    stripped = df[TEXT_COL].str.strip()
    return (stripped == "") | (stripped == ERROR_MARKER)


def clean_split(df: pd.DataFrame, drop_invalid: bool) -> pd.DataFrame:
    """Trả về bảng mới, sau khi xử lý dòng lỗi.

    Nếu drop_invalid = True thì xóa dòng lỗi
    Nếu drop_invalid = False thì thay dòng lỗi bằng chuỗi ""
    """
    cleaned = df.copy()
    mask = find_invalid_rows(df)
    if drop_invalid:
        cleaned = cleaned[~mask]
        cleaned = cleaned.reset_index(drop=True)
        return cleaned
    cleaned.loc[mask, TEXT_COL] = ""
    return cleaned


def read_split(split: str, data_dir: str | Path = "data/raw/vihsd") -> pd.DataFrame:
    """Đọc nguyên trạng một tập (train/dev/test) từ CSV và kiểm tra định dạng.

    Báo ValueError nếu tên tập sai, thiếu cột, hoặc có nhãn ngoài LABEL_NAMES.
    """
    if split not in SPLITS:
        raise ValueError(f"Tập '{split}' không hợp lệ, chỉ nhận {SPLITS}")

    path = Path(data_dir) / f"{split}.csv"
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)

    if (TEXT_COL not in df.columns) or (LABEL_COL not in df.columns):
        raise ValueError(
            f"{path} phải có cột '{TEXT_COL}' và '{LABEL_COL}', nhận được {list(df.columns)}"
        )

    valid_labels = df[LABEL_COL].isin(LABEL_NAMES)
    if not valid_labels.all():
        bad_labels = df.loc[~valid_labels, LABEL_COL].unique().tolist()
        raise ValueError(f"{path} có nhãn ngoài {list(LABEL_NAMES)}: {bad_labels}")

    return df


def load_vihsd(split: str, data_dir: str | Path = "data/raw/vihsd") -> pd.DataFrame:
    """Đọc file dữ liệu

    Nếu tập dữ liệu là train thì bỏ dòng lỗi
    Nếu tập dữ liệu là dev hoặc test thì thay dòng lỗi bằng ""
    """
    df = read_split(split, data_dir)
    drop_invalid = split == "train"
    n_invalid = int(find_invalid_rows(df).sum())
    result = clean_split(df, drop_invalid=drop_invalid)

    action = "bỏ" if drop_invalid else "thay bằng chuỗi rỗng"
    logger.info(f"{split}: {action} {n_invalid} dòng lỗi, còn {len(result)} dòng")
    return result
