"""Mô hình phân loại: baseline TF-IDF + Logistic Regression, và lưu/nạp mọi loại mô hình.

Phần transformer nằm ở src/transformer.py, chỉ import khi config có type: transformer.
Mọi mô hình trả về đều có predict(texts) và predict_proba(texts).
"""

import unicodedata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data import LABEL_NAMES

MODEL_TYPES = ("tfidf_lr", "transformer")


def model_type(config: dict[str, Any]) -> str:
    """Loại mô hình trong config (mặc định tfidf_lr cho các config tuần 1)."""
    kind = config.get("type", "tfidf_lr")
    if kind not in MODEL_TYPES:
        raise ValueError(f"type phải là một trong {MODEL_TYPES}, nhận được {kind!r}")
    return kind


def save_model(model: Any, config: dict[str, Any], models_dir: str | Path) -> Path:
    """Lưu mô hình: tfidf_lr -> models/<tên>.joblib, transformer -> thư mục models/<tên>/."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    if model_type(config) == "transformer":
        path = models_dir / config["name"]
        model.save(path)
    else:
        path = models_dir / f"{config['name']}.joblib"
        joblib.dump(model, path)
    return path


def load_model(config: dict[str, Any], models_dir: str | Path) -> Any:
    """Nạp lại mô hình đã lưu bằng save_model."""
    if model_type(config) == "transformer":
        from src.transformer import TransformerClassifier

        return TransformerClassifier.load(Path(models_dir) / config["name"])
    return joblib.load(Path(models_dir) / f"{config['name']}.joblib")


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
