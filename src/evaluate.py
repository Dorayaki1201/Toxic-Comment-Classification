"""Tính chỉ số đánh giá (macro-F1, F1 từng lớp, confusion matrix) và đánh giá mô hình đã lưu.

Chạy trên test chỉ một lần, sau khi đã chốt cấu hình bằng dev:
    python -m src.evaluate --config configs/<tên>.yaml --split test
"""

import argparse
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
from src.models import load_model
from src.utils import get_logger, load_config, with_seed

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình đã train trên một tập.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--seed", type=int, help="đánh giá mô hình <tên>_s<seed>")
    parser.add_argument(
        "--summarize-seeds",
        type=int,
        nargs="+",
        metavar="SEED",
        help="chỉ gộp các báo cáo <tên>_s<seed> đã có thành trung bình ± độ lệch chuẩn",
    )
    args = parser.parse_args()
    if args.summarize_seeds:
        name = load_config(args.config)["name"]
        summarize_seeds(name, args.summarize_seeds, args.split)
        return
    if args.split == "test":
        logger.warning("Đang chạy trên TEST: chỉ làm một lần, sau khi đã chốt cấu hình bằng dev.")
    evaluate_saved_model(args.config, args.split, seed=args.seed)


if __name__ == "__main__":
    main()
