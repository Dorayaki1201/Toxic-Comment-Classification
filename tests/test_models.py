import json
import unicodedata

import pandas as pd
import pytest
import yaml

from src.data import LABEL_COL, TEXT_COL
from src.models import basic_preprocess, build_tfidf_lr, top_features
from src.train import run

WORD = {"ngram_range": [1, 1], "min_df": 1}
CHAR = {"ngram_range": [2, 3], "min_df": 1}
MODEL = {"C": 10.0, "class_weight": None, "max_iter": 1000}

# Dữ liệu đồ chơi trung tính: nhãn chỉ để kiểm tra luồng code, không mang nghĩa thật.
TEXTS = ["mèo đáng yêu", "mèo ngủ ngon", "chó sủa to", "chó chạy nhanh", "cá bơi lội", "cá bơi xa"]
LABELS = [0, 0, 1, 1, 2, 2]


def test_basic_preprocess_lowercases_and_normalizes_unicode():
    decomposed = unicodedata.normalize("NFD", "Mèo ĐẸP")
    assert basic_preprocess(decomposed) == "mèo đẹp"


@pytest.mark.parametrize(
    "features",
    [{"word": WORD, "char": None}, {"word": None, "char": CHAR}, {"word": WORD, "char": CHAR}],
)
def test_build_tfidf_lr_learns_toy_data(features):
    model = build_tfidf_lr(features, MODEL, seed=0).fit(TEXTS, LABELS)
    assert model.predict(["mèo", "chó", "cá"]).tolist() == [0, 1, 2]
    probabilities = model.predict_proba(["mèo"])
    assert probabilities.shape == (1, 3)
    assert probabilities.sum() == pytest.approx(1.0)


def test_build_tfidf_lr_requires_some_features():
    with pytest.raises(ValueError, match="ít nhất một"):
        build_tfidf_lr({"word": None, "char": None}, MODEL, seed=0)


def test_vocabulary_comes_only_from_training_data():
    model = build_tfidf_lr({"word": WORD, "char": None}, MODEL, seed=0).fit(TEXTS, LABELS)
    model.predict(["rùa bò chậm"])
    vocabulary = model.named_steps["features"].transformer_list[0][1].vocabulary_
    assert "mèo" in vocabulary
    assert "rùa" not in vocabulary


def test_word_features_keep_single_character_tokens():
    model = build_tfidf_lr({"word": WORD, "char": None}, MODEL, seed=0)
    model.fit(["k biết", "e ơi", "a nhé", "k sao"], [0, 1, 2, 0])
    vocabulary = model.named_steps["features"].transformer_list[0][1].vocabulary_
    assert {"k", "e", "a"} <= set(vocabulary)


def test_top_features_returns_most_indicative_word():
    model = build_tfidf_lr({"word": WORD, "char": None}, MODEL, seed=0).fit(TEXTS, LABELS)
    top = top_features(model, label=1, k=3)
    assert top["feature"].iloc[0] == "word__chó"
    assert list(top.columns) == ["label", "feature", "weight"]
    assert (top["label"] == "OFFENSIVE").all()


def test_train_run_end_to_end_without_real_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for split in ["train", "dev"]:
        pd.DataFrame({TEXT_COL: TEXTS + ["#ERROR!"], LABEL_COL: LABELS + [0]}).to_csv(
            data_dir / f"{split}.csv", index=False
        )
    config = {
        "name": "toy",
        "seed": 42,
        "data_dir": str(data_dir),
        "features": {"word": WORD, "char": None},
        "model": MODEL,
    }
    config_path = tmp_path / "toy.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    run(config_path, models_dir=tmp_path / "models", reports_dir=tmp_path / "reports")

    assert (tmp_path / "models" / "toy.joblib").exists()
    report = json.loads((tmp_path / "reports" / "toy" / "dev_metrics.json").read_text("utf-8"))
    assert set(report) >= {"train", "dev", "dev_always_clean", "config_values"}
    assert report["train"]["macro_f1"] == 1.0
    assert sum(map(sum, report["dev"]["confusion_matrix"])) == 7
