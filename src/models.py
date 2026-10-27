"""Mô hình phân loại. Hiện có baseline TF-IDF (word và/hoặc char n-gram) + Logistic Regression."""

import unicodedata
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data import LABEL_NAMES


def basic_preprocess(text: str) -> str:
    """Chuẩn hóa Unicode NFC và viết thường; không sửa teencode hay dấu (cấu hình A)."""
    return unicodedata.normalize("NFC", text).lower()


def build_tfidf_lr(features: dict[str, Any], model: dict[str, Any], seed: int) -> Pipeline:
    """Tạo Pipeline: TF-IDF (word, char hoặc cả hai) -> Logistic Regression.

    features: {"word": {...} | None, "char": {...} | None}, tham số cho TfidfVectorizer.
    model: tham số cho LogisticRegression (C, class_weight, max_iter).
    Vì là Pipeline, fit() chỉ học từ điển/IDF trên dữ liệu được đưa vào (train).
    """
    vectorizers = []
    if features.get("word"):
        params = {**features["word"], "ngram_range": tuple(features["word"]["ngram_range"])}
        vectorizers.append(
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    preprocessor=basic_preprocess,
                    token_pattern=r"(?u)\b\w+\b",
                    **params,
                ),
            )
        )
    if features.get("char"):
        params = {**features["char"], "ngram_range": tuple(features["char"]["ngram_range"])}
        vectorizers.append(
            (
                "char",
                TfidfVectorizer(analyzer="char_wb", preprocessor=basic_preprocess, **params),
            )
        )
    if not vectorizers:
        raise ValueError("Cần bật ít nhất một loại đặc trưng: 'word' hoặc 'char'")

    classifier = LogisticRegression(random_state=seed, **model)
    return Pipeline([("features", FeatureUnion(vectorizers)), ("clf", classifier)])


def top_features(pipeline: Pipeline, label: int, k: int = 20) -> pd.DataFrame:
    """Trả về k đặc trưng có trọng số lớn nhất cho một nhãn (dùng để kiểm tra mô hình học gì)."""
    names = pipeline.named_steps["features"].get_feature_names_out()
    classifier = pipeline.named_steps["clf"]
    row = list(classifier.classes_).index(label)
    weights = pd.Series(classifier.coef_[row], index=names)
    top = weights.nlargest(k)
    return pd.DataFrame(
        {"label": LABEL_NAMES[label], "feature": top.index, "weight": top.to_numpy()}
    )
