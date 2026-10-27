"""Tính chỉ số đánh giá (macro-F1, F1 từng lớp, confusion matrix) và đánh giá mô hình đã lưu.

Chạy trên test chỉ một lần, sau khi đã chốt cấu hình bằng dev:
    python -m src.evaluate --config configs/<tên>.yaml --split test
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.data import LABEL_COL, LABEL_NAMES, TEXT_COL, load_vihsd
from src.utils import get_logger, load_config

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
) -> dict[str, Any]:
    """Nạp mô hình đã train theo config, đoán trên một tập, ghi chỉ số vào reports/<tên>/."""
    config = load_config(config_path)
    name = config["name"]
    model = joblib.load(Path(models_dir) / f"{name}.joblib")
    df = load_vihsd(split, config["data_dir"])

    metrics = compute_metrics(df[LABEL_COL].tolist(), model.predict(df[TEXT_COL]).tolist())
    metrics.update(
        {"config": name, "split": split, "rows": len(df), "sklearn_version": sklearn.__version__}
    )
    save_metrics(metrics, Path(reports_dir) / name / f"{split}_metrics.json")
    logger.info(f"{name} | {split}: macro-F1 = {metrics['macro_f1']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình đã train trên một tập.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    args = parser.parse_args()
    if args.split == "test":
        logger.warning("Đang chạy trên TEST: chỉ làm một lần, sau khi đã chốt cấu hình bằng dev.")
    evaluate_saved_model(args.config, args.split)


if __name__ == "__main__":
    main()
