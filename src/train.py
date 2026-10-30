"""Huấn luyện mô hình theo một file cấu hình và đo trên train + dev (không chạm test).

python -m src.train --config configs/<tên>.yaml
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

from src.data import LABEL_COL, TEXT_COL, load_vihsd
from src.evaluate import compute_metrics, save_metrics
from src.models import build_tfidf_lr, model_type, save_model
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)


def fit_model(
    config: dict[str, Any], train: pd.DataFrame, dev: pd.DataFrame, models_dir: Path
) -> tuple[Any, dict[str, str]]:
    """Train theo loại mô hình trong config; trả về mô hình và thông tin môi trường để ghi báo cáo.

    Transformer dùng dev để giữ epoch tốt nhất (chọn mô hình), không bao giờ dùng test.
    """
    if model_type(config) == "transformer":
        import torch
        import transformers

        from src.transformer import train_transformer

        checkpoints_dir = models_dir / f"{config['name']}_checkpoints"
        model = train_transformer(config, train, dev, checkpoints_dir)
        environment = {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        return model, environment

    model = build_tfidf_lr(config["features"], config["model"], seed=config["seed"])
    model.fit(train[TEXT_COL], train[LABEL_COL])
    return model, {}


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

    model, environment = fit_model(config, train, dev, Path(models_dir))

    report = {
        "config": name,
        "config_values": config,
        "sklearn_version": sklearn.__version__,
        **environment,
        "train": compute_metrics(
            train[LABEL_COL].tolist(), model.predict(train[TEXT_COL]).tolist()
        ),
        "dev": compute_metrics(dev[LABEL_COL].tolist(), model.predict(dev[TEXT_COL]).tolist()),
        "dev_always_clean": compute_metrics(dev[LABEL_COL].tolist(), [0] * len(dev)),
    }
    save_metrics(report, Path(reports_dir) / name / "dev_metrics.json")
    save_model(model, config, models_dir)

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
