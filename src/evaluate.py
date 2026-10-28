"""Tính chỉ số đánh giá (macro-F1, F1 từng lớp, confusion matrix) và đánh giá mô hình đã lưu.

Chạy trên test chỉ một lần, sau khi đã chốt cấu hình bằng dev:
    python -m src.evaluate --config configs/<tên>.yaml --split test
"""

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any

import joblib
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
) -> dict[str, Any]:
    """Nạp mô hình đã train theo config, đoán trên một tập, ghi chỉ số vào reports/<tên>/.

    Ngoài chỉ số trên toàn bộ tập, báo thêm hai phần: câu đã gặp trong train (seen)
    và câu chưa gặp (unseen), để biết điểm có bị thổi phồng do ghi nhớ hay không.
    """
    config = load_config(config_path)
    name = config["name"]
    model = joblib.load(Path(models_dir) / f"{name}.joblib")
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
