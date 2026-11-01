import unicodedata

import pandas as pd
import pytest

from src.data import LABEL_COL, TEXT_COL
from src.keywords import (
    context_window,
    count_tokens_by_label,
    extract_ngrams,
    find_lexicon_candidates,
    is_lexicon_candidate,
    log_odds_by_label,
    sample_contexts,
    tokenize,
)


def make_df(texts: list[str], labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame({TEXT_COL: texts, LABEL_COL: labels})


# --- tokenize / extract_ngrams ---


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Xin CHÀO, bạn!!! :)") == ["xin", "chào", "bạn"]


def test_tokenize_same_result_for_composed_and_decomposed_unicode():
    text = "hôm nay trời đẹp"
    decomposed = unicodedata.normalize("NFD", text)
    assert decomposed != text
    assert tokenize(decomposed) == tokenize(text)


def test_extract_ngrams_bigrams():
    assert extract_ngrams("hôm nay trời đẹp", n=2) == ["hôm nay", "nay trời", "trời đẹp"]


def test_extract_ngrams_too_short_returns_empty():
    assert extract_ngrams("chào", n=2) == []
    assert extract_ngrams("", n=1) == []


# --- count_tokens_by_label ---


def test_count_tokens_counts_each_comment_once():
    df = make_df(["mèo mèo mèo", "mèo và chó"], labels=[0, 0])
    counts = count_tokens_by_label(df)
    assert counts[0]["mèo"] == 2
    assert counts[0]["chó"] == 1


def test_count_tokens_separates_labels():
    df = make_df(["mèo", "chó", "chó"], labels=[0, 1, 1])
    counts = count_tokens_by_label(df)
    assert counts[0] == {"mèo": 1}
    assert counts[1] == {"chó": 2}


# --- log_odds_by_label ---


@pytest.fixture
def cat_dog_counts():
    """Nhãn 0 hay nói về mèo, nhãn 1 hay nói về chó; 'và' xuất hiện đều ở cả hai."""
    texts = ["mèo và cá"] * 20 + ["chó và xương"] * 20
    labels = [0] * 20 + [1] * 20
    return count_tokens_by_label(make_df(texts, labels))


def test_log_odds_top_token_matches_label(cat_dog_counts):
    result = log_odds_by_label(cat_dog_counts, min_count=1)
    top = result.groupby("label")["token"].first()
    assert top[0] in {"mèo", "cá"}
    assert top[1] in {"chó", "xương"}


def test_log_odds_shared_token_is_neutral(cat_dog_counts):
    result = log_odds_by_label(cat_dog_counts, min_count=1)
    z_shared = result.loc[result["token"] == "và", "z"]
    assert (z_shared.abs() < 0.5).all()


def test_log_odds_two_labels_are_mirror_images(cat_dog_counts):
    result = log_odds_by_label(cat_dog_counts, min_count=1)
    z0 = result[result["label"] == 0].set_index("token")["z"].sort_index()
    z1 = result[result["label"] == 1].set_index("token")["z"].sort_index()
    pd.testing.assert_series_equal(z0, -z1, check_names=False)


def test_log_odds_min_count_filters_rare_tokens():
    df = make_df(["mèo"] * 5 + ["chó hiếm"], labels=[0] * 5 + [1])
    result = log_odds_by_label(count_tokens_by_label(df), min_count=2)
    assert set(result["token"]) == {"mèo"}


def test_log_odds_output_columns_and_sorting(cat_dog_counts):
    result = log_odds_by_label(cat_dog_counts, min_count=1)
    assert list(result.columns) == ["label", "token", "count", "count_rest", "z"]
    for _, group in result.groupby("label"):
        assert group["z"].is_monotonic_decreasing


# --- ứng viên từ điển teencode ---


@pytest.mark.parametrize("token", ["đc", "vl", "k", "wa", "loz", "fb"])
def test_is_lexicon_candidate_accepts_non_syllables(token):
    assert is_lexicon_candidate(token)


@pytest.mark.parametrize("token", ["không", "được", "ngu", "50k", "5h", "ko"])
def test_is_lexicon_candidate_rejects_syllables_and_tokens_with_digits(token):
    # "ko" có nguyên âm nên chỉ vào qua danh sách gõ tay, không qua bộ lọc tự động.
    assert not is_lexicon_candidate(token)


def test_find_lexicon_candidates_counts_by_label_and_applies_min_count():
    df = make_df(["đc đc rồi", "đc nha", "dm mày", "hiếm xq"], labels=[0, 1, 2, 0])
    result = find_lexicon_candidates(df, min_count=2, manual=())
    assert result["token"].tolist() == ["đc"]
    row = result.iloc[0]
    assert (row["count"], row["CLEAN"], row["OFFENSIVE"], row["HATE"]) == (2, 1, 1, 0)


def test_find_lexicon_candidates_adds_manual_tokens_only_if_frequent():
    df = make_df(["ko biết", "ko sao", "hok nha"], labels=[0, 0, 0])
    result = find_lexicon_candidates(df, min_count=2, manual=("ko", "hok", "vắng"))
    assert result[["token", "source"]].values.tolist() == [["ko", "manual"]]


def test_find_lexicon_candidates_auto_wins_over_manual_and_sorts_by_count():
    df = make_df(["vl vl", "vl", "ko", "ko", "ko"], labels=[1, 1, 0, 0, 0])
    result = find_lexicon_candidates(df, min_count=1, manual=("ko", "vl"))
    assert result[["token", "source", "count"]].values.tolist() == [
        ["ko", "manual", 3],
        ["vl", "auto", 2],
    ]
    assert list(result.columns) == ["token", "source", "count", "CLEAN", "OFFENSIVE", "HATE"]


def test_find_lexicon_candidates_empty_when_nothing_matches():
    result = find_lexicon_candidates(make_df(["xin chào"], labels=[0]), min_count=1, manual=())
    assert result.empty


def test_sample_contexts_matches_whole_tokens_and_is_reproducible():
    texts = pd.Series(["k biết", "50k thôi", "kẹo ngon", "K nha", "ok k"])
    first = sample_contexts(texts, "k", k=10, seed=1)
    assert sorted(first) == ["K nha", "k biết", "ok k"]
    assert sample_contexts(texts, "k", k=2, seed=1) == sample_contexts(texts, "k", k=2, seed=1)


def test_context_window_centers_on_whole_token():
    text = "kẹo " * 30 + "k biết" + " xa" * 50
    window = context_window(text, "k", width=10)
    assert "k biết" in window
    assert window.startswith("…") and window.endswith("…")
    assert context_window("ngắn k", "k") == "ngắn k"
