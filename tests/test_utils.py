import logging
import random

import numpy as np
import pytest

from src.utils import get_logger, load_config, set_seed


def test_set_seed_reproducible():
    set_seed(42)
    first = (random.random(), np.random.rand(3).tolist())
    set_seed(42)
    second = (random.random(), np.random.rand(3).tolist())
    assert first == second


def test_set_seed_different_seeds_differ():
    set_seed(1)
    a = np.random.rand(3).tolist()
    set_seed(2)
    b = np.random.rand(3).tolist()
    assert a != b


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("seed: 42\nmodel:\n  name: tfidf_lr\n  C: 1.0\n", encoding="utf-8")
    config = load_config(path)
    assert config == {"seed": 42, "model": {"name": "tfidf_lr", "C": 1.0}}


def test_load_config_keeps_vietnamese(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("note: bình luận tiếng Việt\n", encoding="utf-8")
    assert load_config(path)["note"] == "bình luận tiếng Việt"


def test_load_config_rejects_non_mapping(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_get_logger_no_duplicate_handlers():
    logger = get_logger("test_logger")
    get_logger("test_logger")
    assert len(logger.handlers) == 1
    assert logger.level == logging.INFO
