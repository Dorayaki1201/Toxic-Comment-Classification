import unicodedata

import pandas as pd
import pytest

from src.data import LABEL_COL, TEXT_COL
from src.keywords import count_tokens_by_label, extract_ngrams, log_odds_by_label, tokenize


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
