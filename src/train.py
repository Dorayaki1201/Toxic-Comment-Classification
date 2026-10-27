"""Huấn luyện mô hình theo một file cấu hình và đo trên train + dev (không chạm test).

python -m src.train --config configs/<tên>.yaml
"""

import argparse
from pathlib import Path
from typing import Any

import joblib
import sklearn

from src.data import LABEL_COL, TEXT_COL, load_vihsd
from src.evaluate import compute_metrics, save_metrics
from src.models import build_tfidf_lr
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)


def run(
    config_path: str | Path,
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Train trên train, đo train/dev, lưu mô hình vào models/ và chỉ số vào reports/<tên>/."""
    config = load_config(config_path)
    name = config["name"]
    set_seed(config["seed"])

    train = load_vihsd("train", config["data_dir"])
    dev = load_vihsd("dev", config["data_dir"])

    model = build_tfidf_lr(config["features"], config["model"], seed=config["seed"])
    model.fit(train[TEXT_COL], train[LABEL_COL])

    report = {
        "config": name,
        "config_values": config,
        "sklearn_version": sklearn.__version__,
        "train": compute_metrics(
            train[LABEL_COL].tolist(), model.predict(train[TEXT_COL]).tolist()
        ),
        "dev": compute_metrics(dev[LABEL_COL].tolist(), model.predict(dev[TEXT_COL]).tolist()),
        "dev_always_clean": compute_metrics(dev[LABEL_COL].tolist(), [0] * len(dev)),
    }
    save_metrics(report, Path(reports_dir) / name / "dev_metrics.json")

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(models_dir) / f"{name}.joblib")

    logger.info(
        f"{name} | macro-F1 train = {report['train']['macro_f1']:.4f}, "
        f"dev = {report['dev']['macro_f1']:.4f} "
        f"(luôn đoán CLEAN: {report['dev_always_clean']['macro_f1']:.4f})"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình theo file cấu hình.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
