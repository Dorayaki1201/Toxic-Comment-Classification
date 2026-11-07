"""Tính chỉ số đánh giá (macro-F1, F1 từng lớp, confusion matrix) và đánh giá mô hình đã lưu.

Chạy trên test chỉ một lần, sau khi đã chốt cấu hình bằng dev:
    python -m src.evaluate --config configs/<tên>.yaml --split test
Bộ test nhiễu (D-025, D-026), mô hình đã đóng băng:
    python -m src.evaluate --config configs/<tên>.yaml --noisy-test [--seed S]
"""

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.data import LABEL_COL, LABEL_NAMES, TEXT_COL, load_vihsd
from src.models import load_model, model_type
from src.utils import get_logger, load_config, set_seed, with_seed

logger = get_logger(__name__)

LABEL_IDS = sorted(LABEL_NAMES)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Trả về macro-F1, accuracy, precision/recall/F1 từng lớp và confusion matrix.

    Luôn tính đủ 3 nhãn, kể cả khi một nhãn không xuất hiện trong y_true hoặc y_pred.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, zero_division=0
    )
    per_class = {
        LABEL_NAMES[label]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(LABEL_IDS)
    }
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_IDS).tolist(),
        "labels": [LABEL_NAMES[label] for label in LABEL_IDS],
    }


def overlap_key(text: str) -> str:
    """Khóa để so câu trùng: NFC, viết thường, gộp khoảng trắng. Không đưa vào mô hình."""
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def find_seen(texts: pd.Series, train_texts: pd.Series) -> pd.Series:
    """Trả mask bool: câu nào đã xuất hiện trong train (so theo overlap_key).

    Chuỗi rỗng (dòng lỗi ở dev/test) không bao giờ tính là trùng.
    """
    train_keys = set(train_texts.map(overlap_key)) - {""}
    return texts.map(overlap_key).isin(train_keys)


def subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Chỉ số trên phần dữ liệu được chọn bởi mask, kèm số dòng; phần rỗng chỉ trả số dòng."""
    rows = int(mask.sum())
    if rows == 0:
        return {"rows": 0}
    return {"rows": rows, **compute_metrics(y_true[mask].tolist(), y_pred[mask].tolist())}


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """Ghi chỉ số ra file JSON (UTF-8, dễ đọc), tạo thư mục nếu chưa có."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_saved_model(
    config_path: str | Path,
    split: str,
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
    seed: int | None = None,
) -> dict[str, Any]:
    """Nạp mô hình đã train theo config, đoán trên một tập, ghi chỉ số vào reports/<tên>/.

    Ngoài chỉ số trên toàn bộ tập, báo thêm hai phần: câu đã gặp trong train (seen)
    và câu chưa gặp (unseen), để biết điểm có bị thổi phồng do ghi nhớ hay không.
    seed khác None: dùng mô hình <tên>_s<seed> (đã train bằng src.train --seed).
    """
    config = load_config(config_path)
    if seed is not None:
        config = with_seed(config, seed)
    name = config["name"]
    model = load_model(config, models_dir)
    df = load_vihsd(split, config["data_dir"])
    train_texts = load_vihsd("train", config["data_dir"])[TEXT_COL]

    y_true = df[LABEL_COL].to_numpy()
    y_pred = np.asarray(model.predict(df[TEXT_COL]))
    seen = find_seen(df[TEXT_COL], train_texts).to_numpy()

    metrics = compute_metrics(y_true.tolist(), y_pred.tolist())
    metrics.update(
        {
            "config": name,
            "split": split,
            "rows": len(df),
            "sklearn_version": sklearn.__version__,
            "overlap_definition": "trùng train sau NFC + viết thường + gộp khoảng trắng",
            "seen_in_train": subset_metrics(y_true, y_pred, seen),
            "unseen_in_train": subset_metrics(y_true, y_pred, ~seen),
        }
    )
    save_metrics(metrics, Path(reports_dir) / name / f"{split}_metrics.json")
    logger.info(f"{name} | {split}: macro-F1 = {metrics['macro_f1']:.4f}")
    for part in ("seen_in_train", "unseen_in_train"):
        sub = metrics[part]
        score = f"macro-F1 = {sub['macro_f1']:.4f}" if sub["rows"] else "không có dòng"
        logger.info(f"{name} | {split} | {part}: {sub['rows']} dòng, {score}")
    return metrics


def _mean_std(values: list[float]) -> dict[str, Any]:
    """Trung bình và độ lệch chuẩn mẫu (ddof=1) của một chỉ số qua các seed."""
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {"mean": float(array.mean()), "std": std, "values": array.tolist()}


def summarize_seeds(
    name: str, seeds: list[int], split: str, reports_dir: str | Path = "reports"
) -> dict[str, Any]:
    """Gộp chỉ số của các lần chạy <tên>_s<seed> thành trung bình ± độ lệch chuẩn.

    Chỉ đọc file báo cáo đã có, không nạp mô hình. Với dev, đọc khối "dev" trong báo cáo của
    src.train. Ghi kết quả vào reports/<tên>/<split>_summary.json.
    """
    runs = []
    for seed in seeds:
        path = Path(reports_dir) / f"{name}_s{seed}" / f"{split}_metrics.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        runs.append(report.get("dev", report) if split == "dev" else report)

    summary: dict[str, Any] = {
        "config": name,
        "split": split,
        "seeds": list(seeds),
        "macro_f1": _mean_std([run["macro_f1"] for run in runs]),
        "per_class_f1": {
            label: _mean_std([run["per_class"][label]["f1"] for run in runs])
            for label in LABEL_NAMES.values()
        },
    }
    if all("unseen_in_train" in run for run in runs):
        summary["unseen_macro_f1"] = _mean_std([run["unseen_in_train"]["macro_f1"] for run in runs])
    save_metrics(summary, Path(reports_dir) / name / f"{split}_summary.json")

    macro = summary["macro_f1"]
    logger.info(
        f"{name} | {split} | {len(seeds)} seed: macro-F1 = {macro['mean']:.4f} ± {macro['std']:.4f}"
    )
    for label, stats in summary["per_class_f1"].items():
        logger.info(f"{name} | {split} | F1 {label} = {stats['mean']:.4f} ± {stats['std']:.4f}")
    return summary


# --- Bộ test nhiễu (D-025, D-026) ---

NOISY_COLUMNS = ("row_id", TEXT_COL, LABEL_COL, "changed")
ORIGINAL_SET = "original"  # test gốc, đoán lại trong cùng lần chạy để đối chiếu


def load_noisy_set(path: str | Path) -> pd.DataFrame:
    """Đọc một bộ nhiễu do src.noise ghi ra; câu rỗng hay chữ "nan" vẫn là chuỗi, không NaN."""
    df = pd.read_csv(
        path,
        dtype={TEXT_COL: str, "changed": bool},
        keep_default_na=False,
    )
    missing = [col for col in NOISY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path}: thiếu cột {missing}")
    return df


def sha256_file(path: str | Path) -> str:
    """sha256 của đúng các byte trên đĩa (src.noise ghi CSV bằng LF, không cần đổi xuống dòng)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_noisy_set(
    noisy: pd.DataFrame, labels: np.ndarray, path: str | Path, expected_sha256: str
) -> None:
    """Dừng (ValueError) nếu bộ nhiễu không khớp manifest hoặc không gióng hàng với test gốc.

    Kiểm tra: hash file đúng như manifest đã commit, đủ số dòng, row_id = 0..n-1 theo thứ tự,
    nhãn trùng nhãn test gốc từng dòng.
    """
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{path}: sha256 khác manifest (D-025), không dùng bộ này")
    if len(noisy) != len(labels):
        raise ValueError(f"{path}: {len(noisy)} dòng, test gốc có {len(labels)}")
    if not np.array_equal(noisy["row_id"].to_numpy(), np.arange(len(labels))):
        raise ValueError(f"{path}: row_id không phải 0..n-1 theo thứ tự")
    if not np.array_equal(noisy[LABEL_COL].to_numpy(), labels):
        raise ValueError(f"{path}: nhãn khác test gốc")


def prediction_flip_rate(pred_reference: np.ndarray, pred_noisy: np.ndarray) -> float:
    """Tỉ lệ câu có nhãn dự đoán khác với bộ tham chiếu (không cần nhãn thật)."""
    return float(np.mean(np.asarray(pred_reference) != np.asarray(pred_noisy)))


def reference_set_name(config: dict[str, Any], set_names: list[str]) -> str:
    """Bộ để tính mức tụt: transformer so với `nfc_only` (nó không tự NFC, D-025), baseline
    so với test gốc (baseline tự NFC trong models.py)."""
    if model_type(config) == "transformer" and "nfc_only" in set_names:
        return "nfc_only"
    return ORIGINAL_SET


def noisy_set_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    pred_reference: np.ndarray,
    seen: np.ndarray,
    changed: np.ndarray | None,
) -> dict[str, Any]:
    """Chỉ số của một bộ: toàn bộ, seen/unseen (theo câu gốc), phần câu bị đổi, mức tụt.

    `seen` phải đã được gióng theo row_id của câu gốc. Mức tụt = điểm bộ tham chiếu − điểm bộ
    này (dương = tụt). Phần câu bị đổi so với chính những câu đó ở bộ tham chiếu.
    """
    metrics = compute_metrics(y_true.tolist(), pred.tolist())
    reference = compute_metrics(y_true.tolist(), pred_reference.tolist())
    metrics["drop_macro_f1"] = reference["macro_f1"] - metrics["macro_f1"]
    metrics["flip_rate"] = prediction_flip_rate(pred_reference, pred)
    metrics["seen_in_train"] = subset_metrics(y_true, pred, seen)
    metrics["unseen_in_train"] = subset_metrics(y_true, pred, ~seen)
    if changed is not None and changed.any():
        noisy_part = subset_metrics(y_true, pred, changed)
        reference_part = subset_metrics(y_true, pred_reference, changed)
        metrics["changed_rows"] = {
            "rows": noisy_part["rows"],
            "macro_f1": noisy_part["macro_f1"],
            "reference_macro_f1": reference_part["macro_f1"],
            "drop_macro_f1": reference_part["macro_f1"] - noisy_part["macro_f1"],
            "flip_rate": prediction_flip_rate(pred_reference[changed], pred[changed]),
        }
    return metrics


def _committed_test_check(
    report_dir: Path, macro_f1: float, tolerance: float = 1e-9
) -> dict[str, Any]:
    """So điểm test gốc vừa đoán lại với test_metrics.json đã có: khớp = đúng mô hình đã báo cáo."""
    path = report_dir / "test_metrics.json"
    if not path.exists():
        return {"committed_macro_f1": None, "matches": None}
    committed = json.loads(path.read_text(encoding="utf-8"))["macro_f1"]
    return {"committed_macro_f1": committed, "matches": abs(committed - macro_f1) <= tolerance}


def evaluate_noisy(
    config_path: str | Path,
    seed: int | None = None,
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
    noise_config_path: str | Path = "configs/noise.yaml",
) -> dict[str, Any]:
    """Đánh giá một mô hình đã đóng băng trên test gốc + mọi bộ nhiễu trong manifest (D-026).

    Ghi reports/<tên>/noisy_test_metrics.json và noisy_test_predictions.csv (row_id + nhãn
    đoán từng bộ, không có văn bản). Không ghi đè test_metrics.json.
    """
    config = load_config(config_path)
    if seed is not None:
        config = with_seed(config, seed)
    set_seed(config["seed"])
    name = config["name"]
    settings = load_config(noise_config_path)["test_sets"]
    manifest = json.loads(Path(settings["manifest"]).read_text(encoding="utf-8"))
    noisy_dir = Path(settings["out_dir"])

    model = load_model(config, models_dir)
    test = load_vihsd("test", config["data_dir"])
    train_texts = load_vihsd("train", config["data_dir"])[TEXT_COL]
    y_true = test[LABEL_COL].to_numpy()
    # seen/unseen theo CÂU GỐC: câu nhiễu gần như không bao giờ trùng train (D-025).
    seen_original = find_seen(test[TEXT_COL], train_texts).to_numpy()

    predictions = {ORIGINAL_SET: np.asarray(model.predict(test[TEXT_COL]))}
    seen = {ORIGINAL_SET: seen_original}
    changed: dict[str, np.ndarray | None] = {ORIGINAL_SET: None}
    for set_name, info in manifest["sets"].items():
        path = noisy_dir / f"{set_name}.csv"
        noisy = load_noisy_set(path)
        check_noisy_set(noisy, y_true, path, info["sha256"])
        row_ids = noisy["row_id"].to_numpy()
        predictions[set_name] = np.asarray(model.predict(noisy[TEXT_COL]))
        seen[set_name] = seen_original[row_ids]
        changed[set_name] = noisy["changed"].to_numpy()

    if model_type(config) != "transformer" and "nfc_only" in predictions:
        # Baseline tự NFC, nên nfc_only phải cho đúng các dự đoán của test gốc.
        if not np.array_equal(predictions["nfc_only"], predictions[ORIGINAL_SET]):
            raise ValueError("Baseline đoán khác nhau trên nfc_only và test gốc: bộ bị lệch hàng?")

    reference = reference_set_name(config, list(predictions))
    sets = {
        set_name: noisy_set_metrics(
            y_true, pred, predictions[reference], seen[set_name], changed[set_name]
        )
        for set_name, pred in predictions.items()
    }
    check = _committed_test_check(Path(reports_dir) / name, sets[ORIGINAL_SET]["macro_f1"])
    report = {
        "config": name,
        "decision": "D-026",
        "rows": len(test),
        "reference_set": reference,
        "noise_seed": manifest["seed"],
        "noise_inputs": manifest["inputs"],
        "original_matches_committed_test": check,
        "sets": sets,
    }
    out_dir = Path(reports_dir) / name
    save_metrics(report, out_dir / "noisy_test_metrics.json")
    pd.DataFrame({"row_id": np.arange(len(test)), **predictions}).to_csv(
        out_dir / "noisy_test_predictions.csv", index=False, lineterminator="\n"
    )

    if check["matches"] is False:
        logger.warning(
            f"{name}: test gốc ra {sets[ORIGINAL_SET]['macro_f1']:.6f}, khác "
            f"test_metrics.json ({check['committed_macro_f1']:.6f}): KHÔNG phải mô hình đã báo cáo"
        )
    logger.info(f"{name} | tham chiếu: {reference}")
    for set_name, m in sets.items():
        logger.info(
            f"{name} | {set_name:26s} macro-F1 {m['macro_f1']:.4f} | tụt {m['drop_macro_f1']:+.4f}"
            f" | đổi nhãn {m['flip_rate']:.2%}"
        )
    return report


def summarize_noisy_seeds(
    name: str, seeds: list[int], reports_dir: str | Path = "reports"
) -> dict[str, Any]:
    """Gộp noisy_test_metrics.json của các seed: trung bình ± độ lệch chuẩn từng bộ.

    Mức tụt được tính trong từng seed trước rồi mới gộp (so cặp trong cùng một mô hình).
    """
    runs = [
        json.loads(
            (Path(reports_dir) / f"{name}_s{seed}" / "noisy_test_metrics.json").read_text("utf-8")
        )
        for seed in seeds
    ]
    references = {run["reference_set"] for run in runs}
    if len(references) != 1:
        raise ValueError(f"Các seed dùng bộ tham chiếu khác nhau: {references}")

    def stat(set_name: str, getter) -> dict[str, Any]:
        return _mean_std([getter(run["sets"][set_name]) for run in runs])

    sets = {}
    for set_name, first in runs[0]["sets"].items():
        sets[set_name] = {
            "macro_f1": stat(set_name, lambda m: m["macro_f1"]),
            "drop_macro_f1": stat(set_name, lambda m: m["drop_macro_f1"]),
            "flip_rate": stat(set_name, lambda m: m["flip_rate"]),
            "unseen_macro_f1": stat(set_name, lambda m: m["unseen_in_train"]["macro_f1"]),
            "per_class_f1": {
                label: stat(set_name, lambda m, label=label: m["per_class"][label]["f1"])
                for label in LABEL_NAMES.values()
            },
        }
        if "changed_rows" in first:
            sets[set_name]["changed_rows_drop_macro_f1"] = stat(
                set_name, lambda m: m["changed_rows"]["drop_macro_f1"]
            )
    summary = {
        "config": name,
        "seeds": list(seeds),
        "reference_set": references.pop(),
        "original_matches_committed_test": [
            run["original_matches_committed_test"]["matches"] for run in runs
        ],
        "sets": sets,
    }
    save_metrics(summary, Path(reports_dir) / name / "noisy_test_summary.json")
    for set_name, s in sets.items():
        m, d = s["macro_f1"], s["drop_macro_f1"]
        logger.info(
            f"{name} | {set_name:26s} macro-F1 {m['mean']:.4f} ± {m['std']:.4f}"
            f" | tụt {d['mean']:+.4f} ± {d['std']:.4f}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình đã train trên một tập.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["dev", "test"])
    parser.add_argument(
        "--noisy-test",
        action="store_true",
        help="đánh giá trên test gốc + bộ test nhiễu (D-026); không ghi đè test_metrics.json",
    )
    parser.add_argument("--seed", type=int, help="đánh giá mô hình <tên>_s<seed>")
    parser.add_argument(
        "--summarize-seeds",
        type=int,
        nargs="+",
        metavar="SEED",
        help="chỉ gộp các báo cáo <tên>_s<seed> đã có thành trung bình ± độ lệch chuẩn",
    )
    args = parser.parse_args()
    if args.noisy_test == bool(args.split):
        parser.error("chọn đúng một trong --split hoặc --noisy-test")
    if args.noisy_test:
        if args.summarize_seeds:
            summarize_noisy_seeds(load_config(args.config)["name"], args.summarize_seeds)
            return
        logger.warning(
            "Bộ test nhiễu suy ra từ TEST: mô hình đóng băng, không chỉnh gì theo kết quả (D-026)."
        )
        evaluate_noisy(args.config, seed=args.seed)
        return
    if args.summarize_seeds:
        name = load_config(args.config)["name"]
        summarize_seeds(name, args.summarize_seeds, args.split)
        return
    if args.split == "test":
        logger.warning("Đang chạy trên TEST: chỉ làm một lần, sau khi đã chốt cấu hình bằng dev.")
    evaluate_saved_model(args.config, args.split, seed=args.seed)


if __name__ == "__main__":
    main()
