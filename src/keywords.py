"""Tìm token đặc trưng cho từng nhãn bằng log-odds có prior Dirichlet.

Phương pháp: Monroe, Colaresi & Quinn (2008), "Fightin' Words", mục 3.5.
Chỉ chạy trên tập train (xem docs/architecture.md mục 4).
"""

import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from src.data import LABEL_COL, TEXT_COL

TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Viết thường và tách thành âm tiết; bỏ dấu câu, emoji, ký tự đặc biệt."""
    text = unicodedata.normalize("NFC", text).lower()
    return TOKEN_PATTERN.findall(text)


def extract_ngrams(text: str, n: int = 1) -> list[str]:
    """Trả về các n-gram (n âm tiết liền nhau, nối bằng dấu cách) của một bình luận."""
    tokens = tokenize(text)
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def count_tokens_by_label(df: pd.DataFrame, n: int = 1) -> dict[int, Counter]:
    """Đếm, với mỗi nhãn, số bình luận chứa từng n-gram (mỗi bình luận tính tối đa 1 lần)."""
    counts: dict[int, Counter] = {}
    for text, label in zip(df[TEXT_COL], df[LABEL_COL], strict=True):
        counts.setdefault(int(label), Counter()).update(set(extract_ngrams(text, n)))
    return counts


def log_odds_by_label(
    counts: dict[int, Counter], prior_strength: float = 1000.0, min_count: int = 5
) -> pd.DataFrame:
    """Tính z-score log-odds của mỗi token cho từng nhãn, so với các nhãn còn lại.

    Prior lấy theo tần suất chung của token trên mọi nhãn, tổng độ lớn là prior_strength.
    Chỉ giữ token xuất hiện ít nhất min_count lần trên mọi nhãn.
    Trả về các cột: label, token, count, count_rest, z; sắp z giảm dần trong từng nhãn.
    """
    table = pd.DataFrame(counts).fillna(0)
    total = table.sum(axis=1)
    alpha = prior_strength * total / total.sum()

    results = []
    for label in sorted(table.columns):
        y_i = table[label]
        y_j = total - y_i
        n_i = y_i.sum()
        n_j = y_j.sum()

        log_odds_i = np.log((y_i + alpha) / (n_i + prior_strength - y_i - alpha))
        log_odds_j = np.log((y_j + alpha) / (n_j + prior_strength - y_j - alpha))
        delta = log_odds_i - log_odds_j
        variance = 1 / (y_i + alpha) + 1 / (y_j + alpha)

        results.append(
            pd.DataFrame(
                {
                    "label": label,
                    "token": table.index,
                    "count": y_i.astype(int).to_numpy(),
                    "count_rest": y_j.astype(int).to_numpy(),
                    "z": (delta / np.sqrt(variance)).to_numpy(),
                }
            )[(total >= min_count).to_numpy()]
        )

    result = pd.concat(results, ignore_index=True)
    return result.sort_values(["label", "z"], ascending=[True, False], ignore_index=True)
