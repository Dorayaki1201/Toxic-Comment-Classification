"""Bộ dev nhiễu từ ViHSD DEV (D-027): cùng bộ sinh và danh sách bộ với bộ test nhiễu, seed khác.

Không sửa src/noise.py (đóng băng, D-025): module này chỉ gọi các hàm thuần của nó. Trước khi
tạo bộ, kiểm tra seed dev khác seed test và bộ sinh (hash noise.py + bảng viết tắt) giống hệt
lúc tạo bộ test. Module chỉ đọc manifest test (hash, thống kê), không đọc dữ liệu test.

python -m src.noisy_dev          # -> data/noisy/vihsd_dev/<bộ>.csv + reports/noisy_dev/
python -m src.noisy_dev --check  # sinh lại trong bộ nhớ, so hash với manifest đã commit
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src import noise
from src.data import TEXT_COL, load_vihsd
from src.noise import (
    NFC_ONLY,
    compare_with_manifest,
    count_non_nfc,
    csv_bytes,
    load_abbreviations,
    make_noisy_test_sets,
    noisy_test_specs,
    set_summary,
    sha256_text_file,
)
from src.utils import load_config, set_seed

DECISION = "D-027"


def generator_inputs(abbreviations_path: str) -> dict[str, str]:
    """Hash của bộ sinh nhiễu, đặt khóa giống hệt `inputs` trong manifest test để so trực tiếp."""
    return {
        "src/noise.py": sha256_text_file(noise.__file__),
        abbreviations_path: sha256_text_file(abbreviations_path),
    }


def check_preconditions(
    dev_seed: int, test_seed: int, inputs: dict[str, str], test_inputs: dict[str, str]
) -> None:
    """Dừng (ValueError) nếu seed dev trùng seed test, hoặc bộ sinh khác lúc tạo bộ test.

    Bộ sinh khác thì dev nhiễu và test nhiễu không còn cùng loại nhiễu, chọn trên dev vô nghĩa.
    """
    if dev_seed == test_seed:
        raise ValueError(f"Seed dev ({dev_seed}) trùng seed test nhiễu: phải khác (D-027)")
    if inputs != test_inputs:
        raise ValueError(
            "noise.py hoặc bảng viết tắt khác lúc tạo bộ test nhiễu (đóng băng, D-025): "
            f"hiện tại {inputs}, manifest test {test_inputs}"
        )


def build_manifest(
    dev: pd.DataFrame,
    sets: dict[str, pd.DataFrame],
    config: dict[str, Any],
    inputs: dict[str, str],
    source: str,
) -> dict[str, Any]:
    """Manifest cùng cấu trúc manifest test: seed, bộ sinh, và thống kê + sha256 từng bộ."""
    specs = {spec.name: spec for spec in noisy_test_specs(config)}
    summaries = {}
    for name, noisy in sets.items():
        spec = specs.get(name, NFC_ONLY)
        summaries[name] = {
            "transforms": list(spec.transforms),
            "rate": spec.rate,
            **set_summary(noisy, dev),
        }
    return {
        "decision": DECISION,
        "source": source,
        "seed": config["dev_sets"]["seed"],
        "rng_key": "rng_for(seed, f'{tên bộ}-{i}')",
        "rows": len(dev),
        "nfc_changed_rows": count_non_nfc(dev[TEXT_COL]),
        "levels": config["levels"],
        "transforms": config["transforms"],
        "inputs": inputs,
        "sets": summaries,
    }


def run(config: dict[str, Any], data_dir: str, check: bool) -> None:
    """Tạo bộ dev nhiễu và ghi manifest, hoặc chỉ so với manifest đã có (check=True)."""
    settings = config["dev_sets"]
    seed, out_dir, manifest_path = settings["seed"], Path(settings["out_dir"]), settings["manifest"]
    test_manifest = json.loads(Path(config["test_sets"]["manifest"]).read_text(encoding="utf-8"))
    inputs = generator_inputs(config["abbreviations"])
    check_preconditions(seed, config["test_sets"]["seed"], inputs, test_manifest["inputs"])

    set_seed(seed)
    dev = load_vihsd("dev", data_dir)
    sets = make_noisy_test_sets(dev, config, seed, load_abbreviations(config["abbreviations"]))

    if check:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        mismatched = compare_with_manifest(sets, manifest)
        for name, noisy in sorted(sets.items()):
            path = out_dir / f"{name}.csv"
            if path.exists() and path.read_bytes() != csv_bytes(noisy):
                mismatched.append(f"{name} (file trên đĩa)")
        if mismatched:
            print(f"KHÔNG khớp manifest: {mismatched}")
            sys.exit(1)
        print(f"Khớp manifest: {len(sets)} bộ, seed {seed}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, noisy in sets.items():
        (out_dir / f"{name}.csv").write_bytes(csv_bytes(noisy))
    manifest = build_manifest(dev, sets, config, inputs, f"load_vihsd('dev', '{data_dir}')")
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, info in manifest["sets"].items():
        print(
            f"{name:28s} rate {info['rate']}: đổi {info['sentences_changed']}/{info['rows']} câu "
            f"({info['sentences_changed_pct']}%), ~{info['words_changed_pct_approx']}% từ"
        )
    print(f"\nCâu dev đổi khi đưa về NFC: {manifest['nfc_changed_rows']}")
    print(f"Đã ghi {len(sets)} bộ vào {out_dir}, manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo bộ dev nhiễu từ ViHSD DEV (D-027).")
    parser.add_argument("--config", default="configs/noise.yaml")
    parser.add_argument("--data-dir", default="data/raw/vihsd")
    parser.add_argument("--check", action="store_true", help="chỉ so với manifest, không ghi")
    args = parser.parse_args()
    run(load_config(args.config), args.data_dir, args.check)


if __name__ == "__main__":
    main()
