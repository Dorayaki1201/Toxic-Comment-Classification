"""Tìm token đặc trưng cho từng nhãn bằng log-odds có prior Dirichlet, và ứng viên teencode.

Phương pháp log-odds: Monroe, Colaresi & Quinn (2008), "Fightin' Words", mục 3.5.
Chỉ chạy trên tập train (xem docs/architecture.md mục 4).
"""

import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from src.data import LABEL_COL, LABEL_NAMES, TEXT_COL
from src.normalize import FOREIGN_LETTERS, has_vowel

TOKEN_PATTERN = re.compile(r"\w+")

# Teencode có nguyên âm, bộ lọc tự động không bắt được (danh sách gõ tay, D-022). Mỗi token
# vẫn phải đạt min_count trên train mới thành ứng viên; có trong danh sách không có nghĩa là
# được nhận vào từ điển.
MANUAL_CANDIDATES = (
    "ko", "hok", "hông", "lun", "iu", "ok", "oke", "nhìu", "nhiu", "ak", "ah",
    "bik", "bít", "mik", "ny", "lol", "ae",
)  # fmt: skip


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


def is_lexicon_candidate(token: str) -> bool:
    """Token chỉ gồm chữ cái và không giống âm tiết tiếng Việt: không có nguyên âm ("đc"),
    hoặc có chữ f/j/w/z ("wa", "loz"). Token dính số ("50k") bị loại."""
    return token.isalpha() and (not has_vowel(token) or bool(FOREIGN_LETTERS & set(token)))


def find_lexicon_candidates(
    df: pd.DataFrame, min_count: int = 10, manual: tuple[str, ...] = MANUAL_CANDIDATES
) -> pd.DataFrame:
    """Ứng viên cho từ điển teencode: token tự lọc (`auto`) và token gõ tay (`manual`).

    Chỉ giữ token có trong ít nhất min_count bình luận. df phải là TRAIN, đã qua lớp luật
    `normalize` để "vlll" được đếm chung với "vl".
    Trả về các cột: token, source, count, CLEAN, OFFENSIVE, HATE; sắp count giảm dần.
    """
    table = pd.DataFrame(count_tokens_by_label(df)).fillna(0).astype(int)
    table = table.reindex(columns=list(LABEL_NAMES), fill_value=0).rename(columns=LABEL_NAMES)
    table.insert(0, "count", table.sum(axis=1))

    sources = {token: "auto" for token in table.index if is_lexicon_candidate(token)}
    for token in manual:
        if token in table.index:
            sources.setdefault(token, "manual")

    result = table.loc[list(sources)]
    result.insert(0, "source", [sources[token] for token in result.index])
    result = result[result["count"] >= min_count].rename_axis("token").reset_index()
    return result.sort_values(["count", "token"], ascending=[False, True], ignore_index=True)


def context_window(text: str, token: str, width: int = 80) -> str:
    """Đoạn quanh lần xuất hiện đầu tiên của token (đứng riêng), mỗi bên tối đa width ký tự."""
    match = re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text, re.IGNORECASE)
    if match is None:
        return text[: 2 * width]
    start, end = max(0, match.start() - width), match.end() + width
    return ("…" if start > 0 else "") + text[start:end] + ("…" if end < len(text) else "")


def sample_contexts(texts: pd.Series, token: str, k: int = 10, seed: int = 42) -> list[str]:
    """Lấy ngẫu nhiên (có seed) tối đa k bình luận chứa token, để đọc xem token nghĩa là gì."""
    matches = texts[texts.map(lambda text: token in tokenize(text))]
    return matches.sample(min(k, len(matches)), random_state=seed).tolist()


def main() -> None:
    """In ứng viên teencode trên TRAIN kèm ngữ cảnh ngẫu nhiên để chấm (D-022).

    Không có tùy chọn chọn tập: từ điển chỉ được xây từ train.
    """
    import argparse

    from src.data import load_vihsd
    from src.normalize import normalize

    parser = argparse.ArgumentParser(description="Ứng viên từ điển teencode trên train.")
    parser.add_argument("--data-dir", default="data/raw/vihsd")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--contexts", type=int, default=10, help="số ngữ cảnh mỗi ứng viên")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = load_vihsd("train", args.data_dir)
    train[TEXT_COL] = train[TEXT_COL].map(normalize)
    candidates = find_lexicon_candidates(train, args.min_count)
    print(f"Số ứng viên (>= {args.min_count} câu train): {len(candidates)}")
    print(candidates["source"].value_counts().to_string())
    for row in candidates.itertuples(index=False):
        print(f"\n== {row.token!r} [{row.source}] {row.count} câu | C/O/H = ", end="")
        print(f"{row.CLEAN}/{row.OFFENSIVE}/{row.HATE}")
        for text in sample_contexts(train[TEXT_COL], row.token, args.contexts, args.seed):
            print(f"  - {context_window(text, row.token)!r}")


if __name__ == "__main__":
    main()
