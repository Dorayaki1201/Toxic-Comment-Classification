import json

import pytest

from src.evaluate import compute_metrics, save_metrics


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


def test_save_metrics_roundtrip_keeps_vietnamese(tmp_path):
    path = tmp_path / "nested" / "dev_metrics.json"
    save_metrics({"ghi_chú": "thử nghiệm", "macro_f1": 0.5}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "ghi_chú": "thử nghiệm",
        "macro_f1": 0.5,
    }
