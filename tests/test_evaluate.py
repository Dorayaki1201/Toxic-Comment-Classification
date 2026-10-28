import json
import unicodedata

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import LABEL_COL, TEXT_COL
from src.evaluate import (
    compute_metrics,
    evaluate_saved_model,
    find_seen,
    overlap_key,
    save_metrics,
    subset_metrics,
)
from src.models import build_tfidf_lr


def test_compute_metrics_perfect_prediction():
    y = [0, 0, 1, 2, 2]
    metrics = compute_metrics(y, y)
    assert metrics["macro_f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 2]]


def test_compute_metrics_always_clean_matches_hand_calculation():
    # 8 CLEAN, 1 OFFENSIVE, 1 HATE; luôn đoán CLEAN.
    y_true = [0] * 8 + [1, 2]
    y_pred = [0] * 10
    metrics = compute_metrics(y_true, y_pred)
    # F1 của CLEAN: precision 0.8, recall 1.0 -> 2*0.8*1/(1.8) = 0.888...; hai lớp kia = 0.
    clean_f1 = 2 * 0.8 * 1.0 / 1.8
    assert metrics["per_class"]["CLEAN"]["f1"] == pytest.approx(clean_f1)
    assert metrics["per_class"]["HATE"]["f1"] == 0.0
    assert metrics["macro_f1"] == pytest.approx(clean_f1 / 3)
    assert metrics["accuracy"] == pytest.approx(0.8)


def test_compute_metrics_keeps_all_three_labels_when_one_is_missing():
    metrics = compute_metrics([0, 0, 2], [0, 2, 2])
    assert metrics["labels"] == ["CLEAN", "OFFENSIVE", "HATE"]
    assert len(metrics["confusion_matrix"]) == 3
    assert metrics["per_class"]["OFFENSIVE"]["support"] == 0


def test_overlap_key_ignores_case_spacing_and_unicode_form():
    decomposed = unicodedata.normalize("NFD", "  Mèo   ĐẸP ")
    assert overlap_key(decomposed) == "mèo đẹp"


def test_find_seen_matches_light_variants_but_not_other_sentences():
    train = pd.Series(["mèo đẹp", "chó sủa"])
    texts = pd.Series(["mèo đẹp", "MÈO  đẹp", "mèo xấu", "chó sủa to"])
    assert find_seen(texts, train).tolist() == [True, True, False, False]


def test_find_seen_never_marks_empty_text_as_seen():
    train = pd.Series(["", "   ", "mèo"])
    assert find_seen(pd.Series(["", "mèo"]), train).tolist() == [False, True]


def test_subset_metrics_uses_only_masked_rows():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 2])
    result = subset_metrics(y_true, y_pred, np.array([True, True, True, False]))
    assert result["rows"] == 3
    assert result["accuracy"] == 1.0
    assert subset_metrics(y_true, y_pred, np.zeros(4, dtype=bool)) == {"rows": 0}


def test_evaluate_saved_model_reports_seen_and_unseen_parts(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_texts = ["mèo đáng yêu", "mèo ngủ ngon", "chó sủa to", "chó chạy nhanh", "cá bơi xa"]
    train_labels = [0, 0, 1, 1, 2]
    pd.DataFrame({TEXT_COL: train_texts, LABEL_COL: train_labels}).to_csv(
        data_dir / "train.csv", index=False
    )
    # 2 câu đã gặp ở train (1 câu khác hoa/thường), 2 câu mới, 1 dòng lỗi.
    test_texts = ["mèo đáng yêu", "CHÓ SỦA TO", "mèo lười", "cá bơi lội", "#ERROR!"]
    pd.DataFrame({TEXT_COL: test_texts, LABEL_COL: [0, 1, 0, 2, 0]}).to_csv(
        data_dir / "test.csv", index=False
    )
    model_cfg = {"C": 10.0, "class_weight": None, "max_iter": 1000}
    features = {"word": {"ngram_range": [1, 1], "min_df": 1}, "char": None}
    model = build_tfidf_lr(features, model_cfg, seed=0).fit(train_texts, train_labels)
    (tmp_path / "models").mkdir()
    joblib.dump(model, tmp_path / "models" / "toy.joblib")
    config = {"name": "toy", "seed": 0, "data_dir": str(data_dir)}
    config_path = tmp_path / "toy.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    metrics = evaluate_saved_model(
        config_path, "test", models_dir=tmp_path / "models", reports_dir=tmp_path / "reports"
    )

    assert metrics["rows"] == 5
    assert metrics["seen_in_train"]["rows"] == 2
    assert metrics["unseen_in_train"]["rows"] == 3
    saved = json.loads((tmp_path / "reports" / "toy" / "test_metrics.json").read_text("utf-8"))
    assert (
        saved["seen_in_train"]["confusion_matrix"] == metrics["seen_in_train"]["confusion_matrix"]
    )


def test_save_metrics_roundtrip_keeps_vietnamese(tmp_path):
    path = tmp_path / "nested" / "dev_metrics.json"
    save_metrics({"ghi_chú": "thử nghiệm", "macro_f1": 0.5}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "ghi_chú": "thử nghiệm",
        "macro_f1": 0.5,
    }
