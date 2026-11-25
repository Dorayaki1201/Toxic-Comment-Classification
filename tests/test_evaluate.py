import json
import unicodedata

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import LABEL_COL, TEXT_COL
from src.evaluate import (
    FROZEN_FOR_CONFIG_B,
    check_frozen_files,
    check_noisy_set,
    committed_macro_f1,
    compare_with_committed_predictions,
    compute_metrics,
    evaluate_config_b,
    evaluate_noisy,
    evaluate_saved_model,
    find_seen,
    load_noisy_set,
    make_normalizer,
    noisy_set_metrics,
    normalize_stats,
    overlap_key,
    prediction_flip_rate,
    reference_set_name,
    save_metrics,
    sha256_file,
    sha256_lf,
    subset_metrics,
    summarize_config_b_seeds,
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


def make_noisy_dev_project(tmp_path):
    """Như make_noisy_project nhưng bộ nhiễu sinh từ dev.csv và ghi vào mục dev_sets.

    test.csv vẫn tồn tại với câu khác hẳn: nếu code lỡ đọc test thì số dòng/nhãn sẽ lệch.
    """
    config_path, _, _, _ = make_noisy_project(tmp_path)
    data_dir = tmp_path / "data"
    dev_texts = ["chó chạy nhanh", "mèo ngủ ngon", "cá to", "mèo", "chó lười"]
    dev = pd.DataFrame({TEXT_COL: dev_texts, LABEL_COL: [1, 0, 2, 0, 1]})
    dev.to_csv(data_dir / "dev.csv", index=False)

    noisy_dir = tmp_path / "noisy_dev"
    noisy_dir.mkdir()
    spec = NoisySpec("all_heavy", ("remove_diacritics",), 1.0)
    sha = write_noisy_csv(noisy_dir / "all_heavy.csv", build_noisy_set(dev, spec, seed=2))
    manifest_path = tmp_path / "manifest_dev.json"
    manifest = {"seed": 2, "inputs": {}, "sets": {"all_heavy": {"sha256": sha}}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    noise_cfg = {"dev_sets": {"out_dir": str(noisy_dir), "manifest": str(manifest_path)}}
    noise_cfg_path = tmp_path / "noise_dev.yaml"
    noise_cfg_path.write_text(yaml.safe_dump(noise_cfg), encoding="utf-8")
    return config_path, noise_cfg_path


def test_evaluate_noisy_dev_reads_dev_and_writes_dev_files(tmp_path):
    config_path, noise_cfg_path = make_noisy_dev_project(tmp_path)
    report = evaluate_noisy(
        config_path,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        noise_config_path=noise_cfg_path,
        split="dev",
    )
    assert report["decision"] == "D-027" and report["rows"] == 5
    assert set(report["sets"]) == {"original", "all_heavy"}
    # "mèo ngủ ngon" và "chó chạy nhanh" có trong train.
    assert report["sets"]["all_heavy"]["seen_in_train"]["rows"] == 2
    assert report["original_matches_committed_dev"]["matches"] is None
    out = tmp_path / "reports" / "toy"
    assert (out / "noisy_dev_metrics.json").exists() and (
        out / "noisy_dev_predictions.csv"
    ).exists()
    assert not (out / "noisy_test_metrics.json").exists()


def test_committed_check_reads_dev_block_of_dev_metrics(tmp_path):
    # dev_metrics.json do train.py ghi: điểm dev nằm trong khối "dev", không ở gốc như test.
    save_metrics(
        {"train": {"macro_f1": 0.99}, "dev": {"macro_f1": 0.7}}, tmp_path / "dev_metrics.json"
    )
    assert committed_macro_f1(tmp_path, "dev") == 0.7
    save_metrics({"macro_f1": 0.6}, tmp_path / "test_metrics.json")
    assert committed_macro_f1(tmp_path, "test") == 0.6
    assert committed_macro_f1(tmp_path / "missing", "dev") is None


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


# --- Cấu hình B (D-028) ---


def test_sha256_lf_ignores_line_endings_and_frozen_check_detects_edits(tmp_path):
    (tmp_path / "lf.txt").write_bytes(b"a\nb\n")
    (tmp_path / "crlf.txt").write_bytes(b"a\r\nb\r\n")
    assert sha256_lf(tmp_path / "lf.txt") == sha256_lf(tmp_path / "crlf.txt")
    expected = {"crlf.txt": sha256_lf(tmp_path / "lf.txt")}
    check_frozen_files(expected, root=tmp_path)  # không lỗi
    (tmp_path / "crlf.txt").write_bytes(b"a\r\nc\r\n")
    with pytest.raises(ValueError, match="crlf.txt"):
        check_frozen_files(expected, root=tmp_path)


def test_repo_normalize_files_match_hashes_recorded_in_d028():
    # Bẫy: sửa normalize.py hoặc teencode.tsv là test này hỏng, vì B phải đo đúng bản đã ghi.
    check_frozen_files(FROZEN_FOR_CONFIG_B)


def test_make_normalizer_rules_skips_lexicon_and_full_uses_it():
    assert make_normalizer("rules")("ko n.g.u") == "ko ngu"
    assert make_normalizer("full")("ko n.g.u") == "không ngu"
    with pytest.raises(ValueError, match="Biến thể"):
        make_normalizer("seq2seq")


def test_normalize_stats_match_hand_calculation():
    raw = pd.Series(["c.h.ó", "mèo  ", "cho"])
    normalized = pd.Series(["chó", "mèo", "cho"])
    normalized_original = pd.Series(["chó", "mèo", "chó"])
    stats = normalize_stats(raw, normalized, normalized_original, np.array([True, False, True]))
    # Chỉ câu 0 bị luật đổi ("mèo  " chỉ khác khoảng trắng, không tính).
    assert stats["normalize_changed_rate"] == pytest.approx(1 / 3)
    assert stats["recovered_rate"] == pytest.approx(2 / 3)
    # Trong 2 câu bị nhiễu đổi, chỉ câu 0 được khôi phục; câu 2 mất dấu thì luật không gỡ được.
    assert stats["recovered_rate_changed_rows"] == pytest.approx(1 / 2)


def make_config_b_project(tmp_path):
    """Baseline tí hon + dev có câu NFD + 3 bộ dev nhiễu: chèn ký tự (luật gỡ được), bỏ dấu
    (luật không gỡ được), nfc_only."""
    config_path, _, _, _ = make_noisy_project(tmp_path)
    labels = [1, 0, 2, 0, 1]
    dev_texts = ["chó chạy nhanh", "mèo ngủ ngon", "cá to", unicodedata.normalize("NFD", "mèo")]
    dev = pd.DataFrame({TEXT_COL: [*dev_texts, "chó lười"], LABEL_COL: labels})
    dev.to_csv(tmp_path / "data" / "dev.csv", index=False)

    noisy_dir = tmp_path / "noisy_dev_b"
    noisy_dir.mkdir()
    inserted = toy_noisy(
        ["c.h.ó chạy nhanh", "mèo ngủ ngon", "cá to", "mèo", "c.h.ó lười"],
        labels,
        [True, False, False, False, True],
    )
    frames = {
        "insert_chars_medium": inserted,
        "all_heavy": build_noisy_set(dev, NoisySpec("all_heavy", ("remove_diacritics",), 1.0), 2),
        "nfc_only": build_noisy_set(dev, NFC_ONLY, seed=2),
    }
    sets = {
        set_name: {"sha256": write_noisy_csv(noisy_dir / f"{set_name}.csv", df)}
        for set_name, df in frames.items()
    }
    manifest_path = tmp_path / "manifest_dev_b.json"
    manifest_path.write_text(json.dumps({"seed": 2, "inputs": {}, "sets": sets}), "utf-8")
    noise_cfg = {"dev_sets": {"out_dir": str(noisy_dir), "manifest": str(manifest_path)}}
    noise_cfg_path = tmp_path / "noise_dev_b.yaml"
    noise_cfg_path.write_text(yaml.safe_dump(noise_cfg), encoding="utf-8")
    return config_path, noise_cfg_path, manifest_path, noisy_dir


def run_config_b(tmp_path, config_path, noise_cfg_path, variant="rules", split="dev"):
    return evaluate_config_b(
        config_path,
        variant,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        noise_config_path=noise_cfg_path,
        split=split,
    )


def test_config_b_equals_a_where_normalize_changes_nothing(tmp_path):
    config_path, noise_cfg_path, _, _ = make_config_b_project(tmp_path)
    report = run_config_b(tmp_path, config_path, noise_cfg_path)

    assert report["decision"] == "D-028" and report["variant"] == "rules"
    assert report["reference_set_a"] == "original"  # baseline tự NFC
    # Dev gốc không có gì để luật sửa, bộ bỏ dấu luật không gỡ được: B phải đoán y như A.
    preds = pd.read_csv(tmp_path / "reports" / "toy" / "noisy_dev_b_rules_predictions.csv")
    for set_name in ("original", "nfc_only", "all_heavy"):
        assert preds[f"b_{set_name}"].tolist() == preds[f"a_{set_name}"].tolist()
        assert report["sets"][set_name]["paired"]["gain_macro_f1"] == 0.0
    assert report["clean_cost_macro_f1"] == 0.0
    assert TEXT_COL not in preds.columns
    # Chưa chạy A (--noisy-dev) nên không có file để đối chiếu; B không ghi file nào của A.
    assert report["a_matches_committed_predictions"] == {"available": False}
    out = tmp_path / "reports" / "toy"
    assert (out / "noisy_dev_b_rules_metrics.json").exists()
    assert not (out / "noisy_dev_metrics.json").exists()
    assert not (out / "noisy_dev_predictions.csv").exists()


def test_config_b_recovers_inserted_chars_and_pairs_gain(tmp_path):
    config_path, noise_cfg_path, _, _ = make_config_b_project(tmp_path)
    report = run_config_b(tmp_path, config_path, noise_cfg_path)
    inserted = report["sets"]["insert_chars_medium"]

    assert inserted["normalize"]["recovered_rate"] == 1.0
    assert inserted["normalize"]["recovered_rate_changed_rows"] == 1.0
    # Khôi phục hoàn toàn -> B đoán y như trên dev gốc đã chuẩn hóa, không tụt.
    preds = pd.read_csv(tmp_path / "reports" / "toy" / "noisy_dev_b_rules_predictions.csv")
    assert preds["b_insert_chars_medium"].tolist() == preds["b_original"].tolist()
    assert inserted["b"]["drop_macro_f1"] == 0.0
    assert inserted["paired"]["gain_macro_f1"] == pytest.approx(
        inserted["b"]["macro_f1"] - inserted["a"]["macro_f1"]
    )
    assert "gain_changed_rows_macro_f1" in inserted["paired"]


def test_config_b_refuses_test_split(tmp_path):
    config_path, noise_cfg_path, _, _ = make_config_b_project(tmp_path)
    with pytest.raises(ValueError, match="D-028"):
        run_config_b(tmp_path, config_path, noise_cfg_path, split="test")


def test_config_b_stops_when_nfc_only_differs_after_normalize(tmp_path):
    config_path, noise_cfg_path, manifest_path, noisy_dir = make_config_b_project(tmp_path)
    path = noisy_dir / "nfc_only.csv"
    df = load_noisy_set(path)
    df.loc[1, TEXT_COL] = "mèo ngủ"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["sets"]["nfc_only"]["sha256"] = write_noisy_csv(path, df)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="nfc_only"):
        run_config_b(tmp_path, config_path, noise_cfg_path)


def test_config_b_counts_rows_where_a_differs_from_committed_predictions(tmp_path):
    config_path, noise_cfg_path, _, _ = make_config_b_project(tmp_path)
    evaluate_noisy(
        config_path,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        noise_config_path=noise_cfg_path,
        split="dev",
    )
    gate = run_config_b(tmp_path, config_path, noise_cfg_path)["a_matches_committed_predictions"]
    assert gate["available"] and gate["total_differing"] == 0 and gate["missing_sets"] == []

    path = tmp_path / "reports" / "toy" / "noisy_dev_predictions.csv"
    committed = pd.read_csv(path)
    committed.loc[0, "all_heavy"] = (committed.loc[0, "all_heavy"] + 1) % 3
    committed.to_csv(path, index=False)
    gate = compare_with_committed_predictions(path, {"all_heavy": np.zeros(5), "x": np.zeros(5)})
    assert gate["missing_sets"] == ["x"]
    gate = run_config_b(tmp_path, config_path, noise_cfg_path)["a_matches_committed_predictions"]
    assert gate["rows_differing"]["all_heavy"] == 1 and gate["total_differing"] == 1


def write_config_b_report(reports_dir, seed, a_score, gain, a_drop, variant="rules"):
    def side(score, drop):
        per_class = {label: {"f1": score} for label in ["CLEAN", "OFFENSIVE", "HATE"]}
        return {"macro_f1": score, "drop_macro_f1": drop, "per_class": per_class}

    report = {
        "variant": variant,
        "reference_set_a": "nfc_only",
        "frozen_inputs": {"src/normalize.py": "abc"},
        "clean_cost_macro_f1": 0.004,
        "a_matches_committed_predictions": {"available": True, "total_differing": 0},
        "sets": {
            "all_medium": {
                "a": side(a_score, a_drop),
                "b": side(a_score + gain, a_drop - gain),
                "paired": {"gain_macro_f1": gain},
                "normalize": {"recovered_rate": 0.3},
            }
        },
    }
    save_metrics(report, reports_dir / f"toy_s{seed}" / f"noisy_dev_b_{variant}_metrics.json")


def test_summarize_config_b_pairs_gain_within_each_seed(tmp_path):
    # A dao động giữa các seed nhưng B luôn hơn A đúng 0.02: std của lợi ích = 0.
    for seed, score in zip([1, 2, 3], [0.58, 0.60, 0.62], strict=True):
        write_config_b_report(tmp_path, seed, score, gain=0.02, a_drop=0.10)
    summary = summarize_config_b_seeds("toy", [1, 2, 3], "rules", reports_dir=tmp_path)
    medium = summary["sets"]["all_medium"]
    assert medium["b_macro_f1"]["std"] == pytest.approx(0.02)
    assert medium["gain_macro_f1"]["mean"] == pytest.approx(0.02)
    assert medium["gain_macro_f1"]["std"] == pytest.approx(0.0)
    assert medium["positive_gain_seeds"] == 3
    assert medium["recovered_share"] == pytest.approx(0.2)  # 0.02 / 0.10
    assert summary["clean_cost_macro_f1"]["mean"] == pytest.approx(0.004)
    assert summary["a_total_differing_from_committed"] == [0, 0, 0]
    assert (tmp_path / "toy" / "noisy_dev_b_rules_summary.json").exists()


def test_summarize_config_b_rejects_mixed_variants(tmp_path):
    write_config_b_report(tmp_path, 1, 0.6, gain=0.01, a_drop=0.1, variant="rules")
    write_config_b_report(tmp_path, 2, 0.6, gain=0.01, a_drop=0.1, variant="full")
    # Seed 2 ghi file tên "full": đổi tên để giả lập file sai nội dung.
    (tmp_path / "toy_s2" / "noisy_dev_b_full_metrics.json").rename(
        tmp_path / "toy_s2" / "noisy_dev_b_rules_metrics.json"
    )
    with pytest.raises(ValueError, match="variant"):
        summarize_config_b_seeds("toy", [1, 2], "rules", reports_dir=tmp_path)
