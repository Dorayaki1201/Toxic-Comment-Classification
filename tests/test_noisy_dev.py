"""Test cho src/noisy_dev.py (D-027). DataFrame nhỏ viết trong code, không đọc ViHSD."""

import json
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

from src.noise import compare_with_manifest, make_noisy_test_sets
from src.noisy_dev import build_manifest, check_preconditions, generator_inputs
from src.utils import load_config

CONFIG = {
    "levels": {"light": 0.1, "medium": 0.3, "heavy": 0.5},
    "transforms": ["teencode_chars", "remove_diacritics", "elongate", "insert_chars"],
    "dev_sets": {"seed": 2611},
}
INPUTS = {"src/noise.py": "aaa", "resources/noise_abbrev.tsv": "bbb"}
NFD = unicodedata.normalize("NFD", "thằng này")


def small_dev() -> pd.DataFrame:
    texts = ["thằng này ngu quá đi", "phải đi học thôi", NFD, ""]
    return pd.DataFrame({"free_text": texts, "label_id": [2, 0, 1, 0]})


def test_same_seed_as_test_is_rejected():
    with pytest.raises(ValueError, match="trùng seed test"):
        check_preconditions(1105, 1105, INPUTS, INPUTS)


def test_changed_generator_is_rejected():
    changed = {**INPUTS, "src/noise.py": "khác"}
    with pytest.raises(ValueError, match="đóng băng"):
        check_preconditions(2611, 1105, changed, INPUTS)


def test_valid_preconditions_pass():
    check_preconditions(2611, 1105, INPUTS, dict(INPUTS))


def test_manifest_describes_sets_and_detects_changes():
    dev = small_dev()
    sets = make_noisy_test_sets(dev, CONFIG, 2611, None)
    manifest = build_manifest(dev, sets, CONFIG, INPUTS, "small_dev()")
    assert manifest["decision"] == "D-027" and manifest["seed"] == 2611
    assert manifest["rows"] == 4 and manifest["nfc_changed_rows"] == 1
    # 4 bộ một biến đổi + 3 bộ ghép + nfc_only (vì có một câu NFD).
    assert len(manifest["sets"]) == 8 and manifest["sets"]["nfc_only"]["rate"] == 0.0
    assert manifest["sets"]["all_medium"]["transforms"] == CONFIG["transforms"]
    assert compare_with_manifest(sets, manifest) == []
    # Sinh với seed test thay vì seed dev: manifest phải bắt được.
    other = make_noisy_test_sets(dev, CONFIG, 1105, None)
    assert compare_with_manifest(other, manifest)


# --- Kiểm tra trên file thật đã commit (config và manifest test, không đọc dữ liệu). ---


def test_committed_config_seeds_are_distinct():
    config = load_config("configs/noise.yaml")
    seeds = [config["test_sets"]["seed"], config["dev_sets"]["seed"], 3907]  # 3907: D (D-027)
    assert len(set(seeds)) == 3


def test_generator_is_unchanged_since_noisy_test_sets():
    # noise.py và bảng viết tắt phải giống hệt lúc tạo bộ test nhiễu (đóng băng, D-025).
    config = load_config("configs/noise.yaml")
    manifest = json.loads(Path(config["test_sets"]["manifest"]).read_text(encoding="utf-8"))
    assert generator_inputs(config["abbreviations"]) == manifest["inputs"]
