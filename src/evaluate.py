"""Tính chỉ số đánh giá (macro-F1, F1 từng lớp, confusion matrix) và đánh giá mô hình đã lưu.

Chạy trên test chỉ một lần, sau khi đã chốt cấu hình bằng dev:
    python -m src.evaluate --config configs/<tên>.yaml --split test
Bộ test nhiễu (D-025, D-026) và bộ dev nhiễu (D-027), mô hình đã đóng băng:
    python -m src.evaluate --config configs/<tên>.yaml --noisy-test [--seed S]
    python -m src.evaluate --config configs/<tên>.yaml --noisy-dev [--seed S]
Cấu hình B (D-028), chuẩn hóa trước mô hình A, chỉ trên dev nhiễu:
    python -m src.evaluate --config configs/<tên>.yaml --noisy-dev --normalize {rules,full}
"""

import argparse
import hashlib
import json
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
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
from src.normalize import collapse_whitespace, load_lexicon, normalize
from src.utils import get_logger, load_config, set_seed, with_seed

logger = get_logger(__name__)

LABEL_IDS = sorted(LABEL_NAMES)
# Bộ nhiễu theo tập gốc: mục trong configs/noise.yaml và quyết định tương ứng.
NOISY_SPLITS = {"test": ("test_sets", "D-026"), "dev": ("dev_sets", "D-027")}


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


# --- Bộ test nhiễu (D-025, D-026) và bộ dev nhiễu (D-027) ---

NOISY_COLUMNS = ("row_id", TEXT_COL, LABEL_COL, "changed")
ORIGINAL_SET = "original"  # tập gốc (test/dev), đoán lại trong cùng lần chạy để đối chiếu


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
    """Dừng (ValueError) nếu bộ nhiễu không khớp manifest hoặc không gióng hàng với tập gốc.

    Kiểm tra: hash file đúng như manifest đã commit, đủ số dòng, row_id = 0..n-1 theo thứ tự,
    nhãn trùng nhãn tập gốc (test hoặc dev) từng dòng.
    """
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{path}: sha256 khác manifest đã commit, không dùng bộ này")
    if len(noisy) != len(labels):
        raise ValueError(f"{path}: {len(noisy)} dòng, tập gốc có {len(labels)}")
    if not np.array_equal(noisy["row_id"].to_numpy(), np.arange(len(labels))):
        raise ValueError(f"{path}: row_id không phải 0..n-1 theo thứ tự")
    if not np.array_equal(noisy[LABEL_COL].to_numpy(), labels):
        raise ValueError(f"{path}: nhãn khác tập gốc")


def prediction_flip_rate(pred_reference: np.ndarray, pred_noisy: np.ndarray) -> float:
    """Tỉ lệ câu có nhãn dự đoán khác với bộ tham chiếu (không cần nhãn thật)."""
    return float(np.mean(np.asarray(pred_reference) != np.asarray(pred_noisy)))


def reference_set_name(config: dict[str, Any], set_names: list[str]) -> str:
    """Bộ để tính mức tụt: transformer so với `nfc_only` (nó không tự NFC, D-025), baseline
    so với tập gốc (baseline tự NFC trong models.py)."""
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


def committed_macro_f1(report_dir: Path, split: str) -> float | None:
    """macro-F1 đã ghi trước đó cho tập gốc: test_metrics.json ("macro_f1"), hoặc
    dev_metrics.json do train.py ghi ("dev" -> "macro_f1"). None nếu chưa có file."""
    path = report_dir / f"{split}_metrics.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return report["dev"]["macro_f1"] if split == "dev" else report["macro_f1"]


def _committed_check(
    report_dir: Path, split: str, macro_f1: float, tolerance: float = 1e-9
) -> dict[str, Any]:
    """So điểm tập gốc vừa đoán lại với file chỉ số đã có: khớp = đúng mô hình đã báo cáo."""
    committed = committed_macro_f1(report_dir, split)
    if committed is None:
        return {"committed_macro_f1": None, "matches": None}
    return {"committed_macro_f1": committed, "matches": abs(committed - macro_f1) <= tolerance}


@dataclass
class NoisyInputs:
    """Tập gốc + mọi bộ nhiễu đã kiểm tra, gióng theo row_id; khóa là tên bộ ("original" trước)."""

    manifest: dict[str, Any]
    y_true: np.ndarray
    texts: dict[str, pd.Series]
    seen: dict[str, np.ndarray]
    changed: dict[str, np.ndarray | None]


def load_noisy_inputs(
    config: dict[str, Any], split: str, noise_config_path: str | Path
) -> NoisyInputs:
    """Nạp tập gốc (dev/test) và các bộ nhiễu trong manifest; dừng nếu bộ nào sai hash/lệch hàng.

    seen/unseen tính theo CÂU GỐC (qua row_id): câu nhiễu gần như không bao giờ trùng train.
    """
    settings = load_config(noise_config_path)[NOISY_SPLITS[split][0]]
    manifest = json.loads(Path(settings["manifest"]).read_text(encoding="utf-8"))
    noisy_dir = Path(settings["out_dir"])

    base = load_vihsd(split, config["data_dir"])
    train_texts = load_vihsd("train", config["data_dir"])[TEXT_COL]
    y_true = base[LABEL_COL].to_numpy()
    seen_original = find_seen(base[TEXT_COL], train_texts).to_numpy()

    texts = {ORIGINAL_SET: base[TEXT_COL].reset_index(drop=True)}
    seen = {ORIGINAL_SET: seen_original}
    changed: dict[str, np.ndarray | None] = {ORIGINAL_SET: None}
    for set_name, info in manifest["sets"].items():
        path = noisy_dir / f"{set_name}.csv"
        noisy = load_noisy_set(path)
        check_noisy_set(noisy, y_true, path, info["sha256"])
        texts[set_name] = noisy[TEXT_COL]
        seen[set_name] = seen_original[noisy["row_id"].to_numpy()]
        changed[set_name] = noisy["changed"].to_numpy()
    return NoisyInputs(manifest, y_true, texts, seen, changed)


def evaluate_noisy(
    config_path: str | Path,
    seed: int | None = None,
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
    noise_config_path: str | Path = "configs/noise.yaml",
    split: str = "test",
) -> dict[str, Any]:
    """Đánh giá một mô hình đã đóng băng trên tập gốc + mọi bộ nhiễu trong manifest.

    split="test": bộ test nhiễu (D-026); split="dev": bộ dev nhiễu (D-027), dùng để chọn cho
    B/C/D. Ghi reports/<tên>/noisy_<split>_metrics.json và noisy_<split>_predictions.csv
    (row_id + nhãn đoán từng bộ, không có văn bản). Không ghi đè <split>_metrics.json.
    """
    decision = NOISY_SPLITS[split][1]
    config = load_config(config_path)
    if seed is not None:
        config = with_seed(config, seed)
    set_seed(config["seed"])
    name = config["name"]

    model = load_model(config, models_dir)
    inputs = load_noisy_inputs(config, split, noise_config_path)
    manifest, y_true, seen, changed = inputs.manifest, inputs.y_true, inputs.seen, inputs.changed
    predictions = {
        set_name: np.asarray(model.predict(texts)) for set_name, texts in inputs.texts.items()
    }

    if model_type(config) != "transformer" and "nfc_only" in predictions:
        # Baseline tự NFC, nên nfc_only phải cho đúng các dự đoán của tập gốc.
        if not np.array_equal(predictions["nfc_only"], predictions[ORIGINAL_SET]):
            raise ValueError("Baseline đoán khác nhau trên nfc_only và tập gốc: bộ bị lệch hàng?")

    reference = reference_set_name(config, list(predictions))
    sets = {
        set_name: noisy_set_metrics(
            y_true, pred, predictions[reference], seen[set_name], changed[set_name]
        )
        for set_name, pred in predictions.items()
    }
    check = _committed_check(Path(reports_dir) / name, split, sets[ORIGINAL_SET]["macro_f1"])
    report = {
        "config": name,
        "decision": decision,
        "rows": len(y_true),
        "reference_set": reference,
        "noise_seed": manifest["seed"],
        "noise_inputs": manifest["inputs"],
        f"original_matches_committed_{split}": check,
        "sets": sets,
    }
    out_dir = Path(reports_dir) / name
    save_metrics(report, out_dir / f"noisy_{split}_metrics.json")
    pd.DataFrame({"row_id": np.arange(len(y_true)), **predictions}).to_csv(
        out_dir / f"noisy_{split}_predictions.csv", index=False, lineterminator="\n"
    )

    if check["matches"] is False:
        logger.warning(
            f"{name}: {split} gốc ra {sets[ORIGINAL_SET]['macro_f1']:.6f}, khác "
            f"{split}_metrics.json ({check['committed_macro_f1']:.6f}): "
            "KHÔNG phải mô hình đã báo cáo"
        )
    logger.info(f"{name} | tham chiếu: {reference}")
    for set_name, m in sets.items():
        logger.info(
            f"{name} | {set_name:26s} macro-F1 {m['macro_f1']:.4f} | tụt {m['drop_macro_f1']:+.4f}"
            f" | đổi nhãn {m['flip_rate']:.2%}"
        )
    return report


def summarize_noisy_seeds(
    name: str, seeds: list[int], reports_dir: str | Path = "reports", split: str = "test"
) -> dict[str, Any]:
    """Gộp noisy_<split>_metrics.json của các seed: trung bình ± độ lệch chuẩn từng bộ.

    Mức tụt được tính trong từng seed trước rồi mới gộp (so cặp trong cùng một mô hình).
    """
    file_name = f"noisy_{split}_metrics.json"
    runs = [
        json.loads((Path(reports_dir) / f"{name}_s{seed}" / file_name).read_text("utf-8"))
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
        f"original_matches_committed_{split}": [
            run[f"original_matches_committed_{split}"]["matches"] for run in runs
        ],
        "sets": sets,
    }
    save_metrics(summary, Path(reports_dir) / name / f"noisy_{split}_summary.json")
    for set_name, s in sets.items():
        m, d = s["macro_f1"], s["drop_macro_f1"]
        logger.info(
            f"{name} | {set_name:26s} macro-F1 {m['mean']:.4f} ± {m['std']:.4f}"
            f" | tụt {d['mean']:+.4f} ± {d['std']:.4f}"
        )
    return summary


# --- Cấu hình B: normalize trước mô hình A đóng băng (D-028) ---

NORMALIZE_VARIANTS = ("rules", "full")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sha256 (CRLF đổi thành LF) ghi ở D-028 trước khi chạy: B phải đo đúng bản chuẩn hóa này.
FROZEN_FOR_CONFIG_B = {
    "src/normalize.py": "94c7b0696f3de232ddfec0d46ac084a142a31631805b781f719348b6c3533a73",
    "resources/teencode.tsv": "31026f967755ff662d1168d00f27a175b2c2a11ec853926b0b53e6e070800907",
}


def sha256_lf(path: str | Path) -> str:
    """sha256 sau khi đổi CRLF thành LF: cùng một file cho cùng hash trên Windows và Kaggle."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_frozen_files(expected: dict[str, str], root: str | Path = PROJECT_ROOT) -> dict[str, str]:
    """Dừng (ValueError) nếu một file đã đóng băng có hash khác giá trị đã ghi; trả các hash."""
    actual = {rel: sha256_lf(Path(root) / rel) for rel in expected}
    modified = [rel for rel, digest in expected.items() if actual[rel] != digest]
    if modified:
        raise ValueError(f"File đã đóng băng cho cấu hình B bị sửa (D-028): {modified}")
    return actual


def make_normalizer(variant: str) -> Callable[[str], str]:
    """`rules`: chỉ lớp luật 3a; `full`: lớp luật + từ điển teencode 3b (D-028)."""
    if variant == "rules":
        return normalize
    if variant == "full":
        lexicon = load_lexicon()
        return lambda text: normalize(text, lexicon)
    raise ValueError(f"Biến thể chuẩn hóa không hợp lệ: {variant!r}, chọn {NORMALIZE_VARIANTS}")


def normalize_stats(
    raw: pd.Series,
    normalized: pd.Series,
    normalized_original: pd.Series,
    changed: np.ndarray | None,
) -> dict[str, Any]:
    """Chuẩn hóa tác động ra sao lên một bộ, không cần mô hình.

    - `normalize_changed_rate`: % câu bị luật/từ điển đổi (không tính NFC và khoảng trắng).
    - `recovered_rate`: % câu mà normalize(câu bộ này) == normalize(câu gốc): khôi phục hoàn toàn.
    - `recovered_rate_changed_rows`: như trên nhưng chỉ trên câu bị nhiễu đổi.
    """
    plain = raw.map(lambda text: collapse_whitespace(unicodedata.normalize("NFC", text)))
    recovered = normalized.to_numpy() == normalized_original.to_numpy()
    stats = {
        "normalize_changed_rate": float(np.mean(normalized.to_numpy() != plain.to_numpy())),
        "recovered_rate": float(recovered.mean()),
    }
    if changed is not None and changed.any():
        stats["recovered_rate_changed_rows"] = float(recovered[changed].mean())
    return stats


def paired_gain(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Lợi ích B − A trên cùng một bộ, cùng mô hình, cùng lần chạy (dương = B tốt hơn)."""
    paired = {"gain_macro_f1": b["macro_f1"] - a["macro_f1"]}
    if "changed_rows" in a:
        paired["gain_changed_rows_macro_f1"] = (
            b["changed_rows"]["macro_f1"] - a["changed_rows"]["macro_f1"]
        )
    return paired


def compare_with_committed_predictions(
    path: str | Path, predictions: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Cổng "đúng mô hình" cho A: đếm số câu A lần này đoán khác file dự đoán A đã commit."""
    path = Path(path)
    if not path.exists():
        return {"available": False}
    committed = pd.read_csv(path)
    differing = {
        set_name: int(np.sum(committed[set_name].to_numpy() != pred))
        for set_name, pred in predictions.items()
        if set_name in committed.columns
    }
    missing = [set_name for set_name in predictions if set_name not in committed.columns]
    return {
        "available": True,
        "rows_differing": differing,
        "total_differing": sum(differing.values()),
        "missing_sets": missing,
    }


def config_b_file(split: str, variant: str, kind: str) -> str:
    """Tên file của B, tách hẳn file của A: noisy_dev_b_rules_metrics.json, ..."""
    return f"noisy_{split}_b_{variant}_{kind}"


def evaluate_config_b(
    config_path: str | Path,
    variant: str,
    seed: int | None = None,
    models_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
    noise_config_path: str | Path = "configs/noise.yaml",
    split: str = "dev",
) -> dict[str, Any]:
    """Cấu hình B (D-028): mỗi bộ đoán hai lần trong cùng lần chạy, thô (A) và đã chuẩn hóa (B).

    Tham chiếu của A như D-026/D-027; tham chiếu của B là tập gốc đã chuẩn hóa (normalize có NFC
    nên `original` và `nfc_only` thành cùng một đầu vào). Ghi reports/<tên>/noisy_dev_b_<biến
    thể>_metrics.json và _predictions.csv; không đụng file nào của A.
    """
    if split != "dev":
        raise ValueError("D-028: cấu hình B chỉ chạy trên dev nhiễu cho tới khi chốt biến thể")
    frozen = check_frozen_files(FROZEN_FOR_CONFIG_B)
    normalizer = make_normalizer(variant)
    config = load_config(config_path)
    if seed is not None:
        config = with_seed(config, seed)
    set_seed(config["seed"])
    name = config["name"]

    model = load_model(config, models_dir)
    inputs = load_noisy_inputs(config, split, noise_config_path)
    normalized = {set_name: texts.map(normalizer) for set_name, texts in inputs.texts.items()}
    if "nfc_only" in normalized and not np.array_equal(
        normalized["nfc_only"].to_numpy(), normalized[ORIGINAL_SET].to_numpy()
    ):
        raise ValueError("Sau chuẩn hóa, nfc_only khác tập gốc: bộ bị lệch hàng?")

    pred_a = {s: np.asarray(model.predict(texts)) for s, texts in inputs.texts.items()}
    pred_b = {s: np.asarray(model.predict(texts)) for s, texts in normalized.items()}
    reference_a = reference_set_name(config, list(pred_a))

    sets = {}
    for set_name in inputs.texts:
        args = (inputs.seen[set_name], inputs.changed[set_name])
        a = noisy_set_metrics(inputs.y_true, pred_a[set_name], pred_a[reference_a], *args)
        b = noisy_set_metrics(inputs.y_true, pred_b[set_name], pred_b[ORIGINAL_SET], *args)
        stats = normalize_stats(
            inputs.texts[set_name],
            normalized[set_name],
            normalized[ORIGINAL_SET],
            inputs.changed[set_name],
        )
        sets[set_name] = {"a": a, "b": b, "paired": paired_gain(a, b), "normalize": stats}

    out_dir = Path(reports_dir) / name
    report = {
        "config": name,
        "decision": "D-028",
        "variant": variant,
        "rows": len(inputs.y_true),
        "reference_set_a": reference_a,
        "reference_set_b": ORIGINAL_SET,
        "noise_seed": inputs.manifest["seed"],
        "noise_inputs": inputs.manifest["inputs"],
        "frozen_inputs": frozen,
        # Dương = chuẩn hóa làm hại câu sạch.
        "clean_cost_macro_f1": sets[reference_a]["a"]["macro_f1"]
        - sets[ORIGINAL_SET]["b"]["macro_f1"],
        "a_matches_committed_predictions": compare_with_committed_predictions(
            out_dir / f"noisy_{split}_predictions.csv", pred_a
        ),
        "sets": sets,
    }
    save_metrics(report, out_dir / config_b_file(split, variant, "metrics.json"))
    columns = {f"a_{s}": p for s, p in pred_a.items()} | {f"b_{s}": p for s, p in pred_b.items()}
    pd.DataFrame({"row_id": np.arange(len(inputs.y_true)), **columns}).to_csv(
        out_dir / config_b_file(split, variant, "predictions.csv"),
        index=False,
        lineterminator="\n",
    )

    gate = report["a_matches_committed_predictions"]
    if gate.get("total_differing"):
        logger.warning(f"{name}: A lần này đoán khác file đã commit {gate['rows_differing']}")
    logger.info(
        f"{name} | B-{variant} | tham chiếu A: {reference_a} | "
        f"chi phí dev sạch {report['clean_cost_macro_f1']:+.4f}"
    )
    for set_name, s in sets.items():
        logger.info(
            f"{name} | {set_name:26s} A {s['a']['macro_f1']:.4f} | B {s['b']['macro_f1']:.4f}"
            f" | B−A {s['paired']['gain_macro_f1']:+.4f}"
            f" | khôi phục {s['normalize']['recovered_rate']:.1%}"
        )
    return report


def summarize_config_b_seeds(
    name: str,
    seeds: list[int],
    variant: str,
    reports_dir: str | Path = "reports",
    split: str = "dev",
) -> dict[str, Any]:
    """Gộp báo cáo B của các seed: trung bình ± độ lệch chuẩn của lợi ích ghép cặp B − A.

    Lợi ích đã tính trong từng seed. `recovered_share` = lợi ích trung bình / mức tụt trung bình
    của A (tỉ số của hai trung bình, ổn định hơn trung bình của các tỉ số khi mức tụt nhỏ).
    """
    file_name = config_b_file(split, variant, "metrics.json")
    runs = [
        json.loads((Path(reports_dir) / f"{name}_s{seed}" / file_name).read_text("utf-8"))
        for seed in seeds
    ]
    for key in ("variant", "reference_set_a", "frozen_inputs"):
        values = {json.dumps(run[key], sort_keys=True) for run in runs}
        if len(values) != 1:
            raise ValueError(f"Các seed khác nhau ở {key!r}: {values}")

    def stat(set_name: str, getter) -> dict[str, Any]:
        return _mean_std([getter(run["sets"][set_name]) for run in runs])

    sets = {}
    for set_name, first in runs[0]["sets"].items():
        gains = [run["sets"][set_name]["paired"]["gain_macro_f1"] for run in runs]
        a_drop = stat(set_name, lambda s: s["a"]["drop_macro_f1"])
        gain = _mean_std(gains)
        entry = {
            "a_macro_f1": stat(set_name, lambda s: s["a"]["macro_f1"]),
            "b_macro_f1": stat(set_name, lambda s: s["b"]["macro_f1"]),
            "gain_macro_f1": gain,
            "positive_gain_seeds": sum(g > 0 for g in gains),
            "a_drop_macro_f1": a_drop,
            "b_drop_macro_f1": stat(set_name, lambda s: s["b"]["drop_macro_f1"]),
            "recovered_share": gain["mean"] / a_drop["mean"] if a_drop["mean"] > 0 else None,
            "b_per_class_f1": {
                label: stat(set_name, lambda s, label=label: s["b"]["per_class"][label]["f1"])
                for label in LABEL_NAMES.values()
            },
            # Chuẩn hóa là tất định: giống nhau ở mọi seed, lấy từ seed đầu.
            "normalize": first["normalize"],
        }
        if "gain_changed_rows_macro_f1" in first["paired"]:
            entry["gain_changed_rows_macro_f1"] = stat(
                set_name, lambda s: s["paired"]["gain_changed_rows_macro_f1"]
            )
        sets[set_name] = entry
    summary = {
        "config": name,
        "decision": "D-028",
        "variant": variant,
        "seeds": list(seeds),
        "reference_set_a": runs[0]["reference_set_a"],
        "frozen_inputs": runs[0]["frozen_inputs"],
        "clean_cost_macro_f1": _mean_std([run["clean_cost_macro_f1"] for run in runs]),
        "a_total_differing_from_committed": [
            run["a_matches_committed_predictions"].get("total_differing") for run in runs
        ],
        "sets": sets,
    }
    save_metrics(summary, Path(reports_dir) / name / config_b_file(split, variant, "summary.json"))
    cost = summary["clean_cost_macro_f1"]
    logger.info(f"{name} | B-{variant} | chi phí dev sạch {cost['mean']:+.4f} ± {cost['std']:.4f}")
    for set_name, s in sets.items():
        g = s["gain_macro_f1"]
        positive = f"{s['positive_gain_seeds']}/{len(seeds)} seed dương"
        logger.info(
            f"{name} | {set_name:26s} B {s['b_macro_f1']['mean']:.4f}"
            f" | B−A {g['mean']:+.4f} ± {g['std']:.4f} ({positive})"
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
    parser.add_argument(
        "--noisy-dev",
        action="store_true",
        help="đánh giá trên dev gốc + bộ dev nhiễu (D-027), dùng để chọn cho B/C/D",
    )
    parser.add_argument(
        "--normalize",
        choices=NORMALIZE_VARIANTS,
        help="cấu hình B (D-028): chuẩn hóa trước mô hình A đóng băng; chỉ dùng với --noisy-dev",
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
    if args.noisy_test + args.noisy_dev + bool(args.split) != 1:
        parser.error("chọn đúng một trong --split, --noisy-test hoặc --noisy-dev")
    if args.normalize and not args.noisy_dev:
        parser.error("--normalize chỉ dùng với --noisy-dev (D-028: chưa chạy B trên test)")
    noisy_split = "test" if args.noisy_test else "dev" if args.noisy_dev else None
    if args.normalize:
        if args.summarize_seeds:
            name = load_config(args.config)["name"]
            summarize_config_b_seeds(name, args.summarize_seeds, args.normalize)
            return
        evaluate_config_b(args.config, args.normalize, seed=args.seed)
        return
    if noisy_split:
        if args.summarize_seeds:
            name = load_config(args.config)["name"]
            summarize_noisy_seeds(name, args.summarize_seeds, split=noisy_split)
            return
        if noisy_split == "test":
            logger.warning(
                "Bộ test nhiễu suy ra từ TEST: mô hình đóng băng, không chỉnh gì theo kết quả "
                "(D-026)."
            )
        evaluate_noisy(args.config, seed=args.seed, split=noisy_split)
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
