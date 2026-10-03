"""Tiện ích dùng chung: cố định seed, đọc cấu hình, logging."""

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên để kết quả chạy lại giống hệt nhau.

    Gọi ở đầu mỗi lần chạy (train, sinh nhiễu, chia dữ liệu...).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch chỉ cài từ tuần 2; khi chưa có thì bỏ qua.
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict[str, Any]:
    """Đọc file cấu hình YAML và trả về dict.

    Báo lỗi nếu file rỗng hoặc không phải dạng key: value ở cấp ngoài cùng.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Cấu hình {path} phải là một mapping YAML, nhận được {type(config)}")
    return config


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Tạo logger in ra màn hình kèm thời gian; gọi nhiều lần không bị in trùng dòng."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger
