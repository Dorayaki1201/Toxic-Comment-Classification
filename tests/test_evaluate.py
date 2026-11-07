import json
import unicodedata

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import LABEL_COL, TEXT_COL
from src.evaluate import (
    check_noisy_set,
    compute_metrics,
    evaluate_noisy,
    evaluate_saved_model,
    find_seen,
    load_noisy_set,
    noisy_set_metrics,
    overlap_key,
    prediction_flip_rate,
    reference_set_name,
    save_metrics,
    sha256_file,
    subset_metrics,
    summarize_noisy_seeds,
    summarize_seeds,
)
from src.models import build_tfidf_lr
from src.noise import NFC_ONLY, NoisySpec, build_noisy_set, csv_bytes


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


def write_seed_report(reports_dir, name, seed, split, macro_f1, unseen=None):
    per_class = {label: {"f1": macro_f1} for label in ["CLEAN", "OFFENSIVE", "HATE"]}
    metrics = {"macro_f1": macro_f1, "per_class": per_class}
    if unseen is not None:
        metrics["unseen_in_train"] = {"macro_f1": unseen}
    report = {"dev": metrics, "train": {"macro_f1": 1.0}} if split == "dev" else metrics
    save_metrics(report, reports_dir / f"{name}_s{seed}" / f"{split}_metrics.json")


def test_summarize_seeds_reads_dev_block_and_uses_sample_std(tmp_path):
    for seed, score in zip([1, 2, 3], [0.70, 0.72, 0.74], strict=True):
        write_seed_report(tmp_path, "toy", seed, "dev", score)
    summary = summarize_seeds("toy", [1, 2, 3], "dev", reports_dir=tmp_path)
    assert summary["macro_f1"]["mean"] == pytest.approx(0.72)
    # Độ lệch chuẩn mẫu (ddof=1) của 0.70, 0.72, 0.74 là 0.02.
    assert summary["macro_f1"]["std"] == pytest.approx(0.02)
    assert summary["per_class_f1"]["HATE"]["values"] == pytest.approx([0.70, 0.72, 0.74])
    assert "unseen_macro_f1" not in summary
    assert (tmp_path / "toy" / "dev_summary.json").exists()


def test_summarize_seeds_on_test_includes_unseen(tmp_path):
    for seed, score in zip([1, 2], [0.60, 0.64], strict=True):
        write_seed_report(tmp_path, "toy", seed, "test", score, unseen=score - 0.01)
    summary = summarize_seeds("toy", [1, 2], "test", reports_dir=tmp_path)
    assert summary["unseen_macro_f1"]["mean"] == pytest.approx(0.61)


def test_save_metrics_roundtrip_keeps_vietnamese(tmp_path):
    path = tmp_path / "nested" / "dev_metrics.json"
    save_metrics({"ghi_chú": "thử nghiệm", "macro_f1": 0.5}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "ghi_chú": "thử nghiệm",
        "macro_f1": 0.5,
    }


# --- Bộ test nhiễu (D-026) ---


def write_noisy_csv(path, df):
    path.write_bytes(csv_bytes(df))
    return sha256_file(path)


def toy_noisy(texts, labels, changed=None):
    return pd.DataFrame(
        {
            "row_id": range(len(texts)),
            TEXT_COL: texts,
            LABEL_COL: labels,
            "changed": changed if changed is not None else [False] * len(texts),
        }
    )


def test_load_noisy_set_keeps_empty_and_nan_strings(tmp_path):
    path = tmp_path / "set.csv"
    write_noisy_csv(path, toy_noisy(["", "nan", "NULL", "mèo"], [0, 0, 1, 2], [False, True] * 2))
    df = load_noisy_set(path)
    assert df[TEXT_COL].tolist() == ["", "nan", "NULL", "mèo"]
    assert df["changed"].tolist() == [False, True, False, True]


def test_check_noisy_set_accepts_aligned_set_and_rejects_broken_ones(tmp_path):
    labels = np.array([0, 1, 2])
    good = toy_noisy(["a", "b", "c"], labels.tolist())
    path = tmp_path / "good.csv"
    digest = write_noisy_csv(path, good)
    check_noisy_set(load_noisy_set(path), labels, path, digest)  # không lỗi

    with pytest.raises(ValueError, match="sha256"):
        check_noisy_set(load_noisy_set(path), labels, path, "0" * 64)

    shuffled = good.iloc[[1, 0, 2]].reset_index(drop=True)
    path = tmp_path / "shuffled.csv"
    with pytest.raises(ValueError, match="row_id"):
        check_noisy_set(shuffled, labels, path, write_noisy_csv(path, shuffled))

    relabeled = toy_noisy(["a", "b", "c"], [0, 1, 1])
    path = tmp_path / "relabeled.csv"
    with pytest.raises(ValueError, match="nhãn"):
        check_noisy_set(relabeled, labels, path, write_noisy_csv(path, relabeled))

    with pytest.raises(ValueError, match="dòng"):
        check_noisy_set(good, np.array([0, 1]), tmp_path / "good.csv", digest)


def test_prediction_flip_rate():
    assert prediction_flip_rate(np.array([0, 1, 2, 0]), np.array([0, 2, 2, 1])) == 0.5


def test_reference_set_is_nfc_only_for_transformer_only():
    names = ["original", "all_light", "nfc_only"]
    assert reference_set_name({"type": "transformer"}, names) == "nfc_only"
    assert reference_set_name({"type": "tfidf_lr"}, names) == "original"
    assert reference_set_name({"type": "transformer"}, ["original", "all_light"]) == "original"


def test_noisy_set_metrics_drop_and_changed_rows_match_hand_calculation():
    y_true = np.array([0, 0, 1, 2])
    reference = np.array([0, 0, 1, 2])  # đúng hết
    pred = np.array([0, 0, 0, 2])  # câu 2 bị nhiễu làm sai
    # Phần bị đổi có đủ 3 lớp: lớp vắng mặt sẽ có F1 = 0 và kéo macro-F1 xuống.
    changed = np.array([True, False, True, True])
    seen = np.array([True, False, False, False])
    m = noisy_set_metrics(y_true, pred, reference, seen, changed)
    assert m["drop_macro_f1"] == pytest.approx(1.0 - m["macro_f1"])
    assert m["flip_rate"] == 0.25
    assert m["seen_in_train"]["rows"] == 1
    part = m["changed_rows"]
    assert part["rows"] == 3
    assert part["reference_macro_f1"] == 1.0
    assert part["flip_rate"] == pytest.approx(1 / 3)
    # Phần bị đổi: đúng [0, 2], đoán sai câu OFFENSIVE thành CLEAN -> F1 = (2/3 + 0 + 1) / 3.
    assert part["macro_f1"] == pytest.approx((2 / 3 + 0 + 1) / 3)
    assert "changed_rows" not in noisy_set_metrics(y_true, pred, reference, seen, None)


def make_noisy_project(tmp_path):
    """Dữ liệu tí hon + baseline đã train + 2 bộ nhiễu + manifest, đúng cấu trúc thật."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_texts = ["mèo đáng yêu", "mèo ngủ ngon", "chó sủa to", "chó chạy nhanh", "cá bơi xa"]
    train_labels = [0, 0, 1, 1, 2]
    pd.DataFrame({TEXT_COL: train_texts, LABEL_COL: train_labels}).to_csv(
        data_dir / "train.csv", index=False
    )
    # Câu 0 và 1 đã gặp trong train; câu 1 viết dạng NFD để có bộ nfc_only.
    test_texts = ["mèo đáng yêu", unicodedata.normalize("NFD", "chó sủa to"), "mèo lười", "cá"]
    test = pd.DataFrame({TEXT_COL: test_texts, LABEL_COL: [0, 1, 0, 2]})
    test.to_csv(data_dir / "test.csv", index=False)

    model_cfg = {"C": 10.0, "class_weight": None, "max_iter": 1000}
    features = {"word": {"ngram_range": [1, 1], "min_df": 1}, "char": None}
    model = build_tfidf_lr(features, model_cfg, seed=0).fit(train_texts, train_labels)
    (tmp_path / "models").mkdir()
    joblib.dump(model, tmp_path / "models" / "toy.joblib")
    config_path = tmp_path / "toy.yaml"
    config = {"name": "toy", "seed": 0, "data_dir": str(data_dir)}
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    noisy_dir = tmp_path / "noisy"
    noisy_dir.mkdir()
    sets = {}
    for spec in (NoisySpec("all_heavy", ("remove_diacritics",), 1.0), NFC_ONLY):
        df = build_noisy_set(test, spec, seed=1)
        sets[spec.name] = {"sha256": write_noisy_csv(noisy_dir / f"{spec.name}.csv", df)}
    manifest_path = tmp_path / "manifest.json"
    manifest = {"seed": 1, "inputs": {}, "sets": sets}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    noise_cfg = {"test_sets": {"out_dir": str(noisy_dir), "manifest": str(manifest_path)}}
    noise_cfg_path = tmp_path / "noise.yaml"
    noise_cfg_path.write_text(yaml.safe_dump(noise_cfg), encoding="utf-8")
    return config_path, noise_cfg_path, manifest_path, noisy_dir


def run_noisy(tmp_path, config_path, noise_cfg_path):
    return evaluate_noisy(
        config_path,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        noise_config_path=noise_cfg_path,
    )


def test_evaluate_noisy_counts_seen_by_original_sentence(tmp_path):
    config_path, noise_cfg_path, _, _ = make_noisy_project(tmp_path)
    report = run_noisy(tmp_path, config_path, noise_cfg_path)

    assert report["reference_set"] == "original"
    assert set(report["sets"]) == {"original", "all_heavy", "nfc_only"}
    # Câu bỏ dấu không còn trùng train, nhưng câu GỐC thì có: seen phải giữ nguyên 2 câu.
    for metrics in report["sets"].values():
        assert metrics["seen_in_train"]["rows"] == 2
    assert report["sets"]["original"]["drop_macro_f1"] == 0.0
    # Chưa có test_metrics.json để đối chiếu.
    assert report["original_matches_committed_test"]["matches"] is None

    preds = pd.read_csv(tmp_path / "reports" / "toy" / "noisy_test_predictions.csv")
    assert preds.columns.tolist() == ["row_id", "original", "all_heavy", "nfc_only"]
    assert TEXT_COL not in preds.columns
    assert preds["nfc_only"].tolist() == preds["original"].tolist()


def test_evaluate_noisy_checks_against_committed_test_metrics(tmp_path):
    config_path, noise_cfg_path, _, _ = make_noisy_project(tmp_path)
    evaluate_saved_model(
        config_path, "test", models_dir=tmp_path / "models", reports_dir=tmp_path / "reports"
    )
    report = run_noisy(tmp_path, config_path, noise_cfg_path)
    assert report["original_matches_committed_test"]["matches"] is True

    path = tmp_path / "reports" / "toy" / "test_metrics.json"
    committed = json.loads(path.read_text("utf-8"))
    committed["macro_f1"] -= 0.01
    path.write_text(json.dumps(committed), encoding="utf-8")
    report = run_noisy(tmp_path, config_path, noise_cfg_path)
    assert report["original_matches_committed_test"]["matches"] is False


def test_evaluate_noisy_rejects_set_that_differs_from_manifest(tmp_path):
    config_path, noise_cfg_path, _, noisy_dir = make_noisy_project(tmp_path)
    path = noisy_dir / "all_heavy.csv"
    path.write_bytes(path.read_bytes().replace(b"meo", b"cho"))
    with pytest.raises(ValueError, match="sha256"):
        run_noisy(tmp_path, config_path, noise_cfg_path)


def test_evaluate_noisy_stops_when_baseline_differs_on_nfc_only(tmp_path):
    # Bộ nfc_only bị đổi nội dung nhưng manifest cũng được cập nhật theo: chỉ kiểm tra
    # "baseline đoán y hệt trên nfc_only và test gốc" mới bắt được.
    config_path, noise_cfg_path, manifest_path, noisy_dir = make_noisy_project(tmp_path)
    path = noisy_dir / "nfc_only.csv"
    df = load_noisy_set(path)
    df.loc[0, TEXT_COL] = "chó sủa to"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["sets"]["nfc_only"]["sha256"] = write_noisy_csv(path, df)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="nfc_only"):
        run_noisy(tmp_path, config_path, noise_cfg_path)


def write_noisy_report(reports_dir, name, seed, scores, reference="nfc_only"):
    sets = {}
    for set_name, (score, drop) in scores.items():
        per_class = {label: {"f1": score} for label in ["CLEAN", "OFFENSIVE", "HATE"]}
        sets[set_name] = {
            "macro_f1": score,
            "drop_macro_f1": drop,
            "flip_rate": 0.1,
            "unseen_in_train": {"macro_f1": score},
            "per_class": per_class,
        }
    report = {
        "reference_set": reference,
        "original_matches_committed_test": {"matches": True},
        "sets": sets,
    }
    save_metrics(report, reports_dir / f"{name}_s{seed}" / "noisy_test_metrics.json")


def test_summarize_noisy_seeds_pairs_drop_within_each_seed(tmp_path):
    # Điểm dao động giữa các seed, nhưng mỗi seed đều tụt đúng 0.10: std của mức tụt = 0.
    for seed, score in zip([1, 2, 3], [0.70, 0.72, 0.74], strict=True):
        scores = {"nfc_only": (score, 0.0), "all_heavy": (score - 0.10, 0.10)}
        write_noisy_report(tmp_path, "toy", seed, scores)
    summary = summarize_noisy_seeds("toy", [1, 2, 3], reports_dir=tmp_path)
    heavy = summary["sets"]["all_heavy"]
    assert heavy["macro_f1"]["std"] == pytest.approx(0.02)
    assert heavy["drop_macro_f1"]["mean"] == pytest.approx(0.10)
    assert heavy["drop_macro_f1"]["std"] == pytest.approx(0.0)
    assert summary["reference_set"] == "nfc_only"
    assert (tmp_path / "toy" / "noisy_test_summary.json").exists()


def test_summarize_noisy_seeds_rejects_mixed_reference_sets(tmp_path):
    write_noisy_report(tmp_path, "toy", 1, {"original": (0.7, 0.0)}, reference="nfc_only")
    write_noisy_report(tmp_path, "toy", 2, {"original": (0.7, 0.0)}, reference="original")
    with pytest.raises(ValueError, match="tham chiếu"):
        summarize_noisy_seeds("toy", [1, 2], reports_dir=tmp_path)
