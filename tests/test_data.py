from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    LABEL_COL,
    TEXT_COL,
    clean_split,
    find_invalid_rows,
    load_vihsd,
    read_split,
)


def make_df(texts: list[str], labels: list[int] | None = None) -> pd.DataFrame:
    """Tạo bảng nhỏ có đủ 2 cột như ViHSD; không truyền nhãn thì đặt hết bằng 0."""
    if labels is None:
        labels = [0] * len(texts)
    return pd.DataFrame({TEXT_COL: texts, LABEL_COL: labels})


def write_split(data_dir: Path, split: str, df: pd.DataFrame) -> None:
    """Ghi bảng ra <data_dir>/<split>.csv giống cấu trúc file ViHSD."""
    df.to_csv(data_dir / f"{split}.csv", index=False, encoding="utf-8")


# --- find_invalid_rows ---


def test_find_invalid_rows_flags_empty_and_error_marker():
    df = make_df(["", "   ", "#ERROR!", "  #ERROR!  "])
    assert find_invalid_rows(df).tolist() == [True, True, True, True]


def test_find_invalid_rows_keeps_normal_text():
    df = make_df(["xin chào", "NA", "#ERROR! xin chào"])
    assert find_invalid_rows(df).tolist() == [False, False, False]


def test_find_invalid_rows_marks_correct_positions():
    df = make_df(["xin chào", "", "hôm nay trời đẹp", "#ERROR!"])
    assert find_invalid_rows(df).tolist() == [False, True, False, True]


# --- clean_split ---


def test_clean_split_drop_removes_invalid_rows_and_resets_index():
    df = make_df(["xin chào", "", "hôm nay trời đẹp", "#ERROR!"], labels=[0, 1, 2, 1])
    result = clean_split(df, drop_invalid=True)
    assert result[TEXT_COL].tolist() == ["xin chào", "hôm nay trời đẹp"]
    assert result[LABEL_COL].tolist() == [0, 2]
    assert result.index.tolist() == [0, 1]


def test_clean_split_keep_blanks_invalid_text_and_keeps_labels():
    df = make_df(["xin chào", "  ", "hôm nay trời đẹp", "#ERROR!"], labels=[0, 1, 2, 1])
    result = clean_split(df, drop_invalid=False)
    assert result[TEXT_COL].tolist() == ["xin chào", "", "hôm nay trời đẹp", ""]
    assert result[LABEL_COL].tolist() == [0, 1, 2, 1]


@pytest.mark.parametrize("drop_invalid", [True, False])
def test_clean_split_does_not_modify_input(drop_invalid):
    df = make_df(["xin chào", "", "#ERROR!"], labels=[0, 1, 2])
    original = df.copy()
    clean_split(df, drop_invalid=drop_invalid)
    pd.testing.assert_frame_equal(df, original)


# --- read_split ---


def test_read_split_keeps_na_like_words_as_text(tmp_path):
    write_split(tmp_path, "train", make_df(["xin chào", "", "NA", "null"]))
    df = read_split("train", tmp_path)
    assert df[TEXT_COL].tolist() == ["xin chào", "", "NA", "null"]


def test_read_split_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="không hợp lệ"):
        read_split("validation", tmp_path)


@pytest.mark.parametrize("missing_col", [TEXT_COL, LABEL_COL])
def test_read_split_rejects_missing_column(tmp_path, missing_col):
    df = make_df(["xin chào"]).drop(columns=missing_col)
    write_split(tmp_path, "dev", df)
    with pytest.raises(ValueError, match="phải có cột"):
        read_split("dev", tmp_path)


def test_read_split_rejects_unknown_label(tmp_path):
    write_split(tmp_path, "test", make_df(["xin chào", "hôm nay trời đẹp"], labels=[0, 3]))
    with pytest.raises(ValueError, match="nhãn ngoài"):
        read_split("test", tmp_path)


# --- load_vihsd ---


@pytest.mark.parametrize(
    ("split", "expected_rows"),
    [("train", 2), ("dev", 4), ("test", 4)],
)
def test_load_vihsd_drops_invalid_rows_only_in_train(tmp_path, split, expected_rows):
    df = make_df(["xin chào", "", "hôm nay trời đẹp", "#ERROR!"], labels=[0, 1, 2, 1])
    write_split(tmp_path, split, df)
    assert len(load_vihsd(split, tmp_path)) == expected_rows
