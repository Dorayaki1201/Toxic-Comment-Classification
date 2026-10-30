"""Fine-tune mô hình transformer (ViSoBERT, PhoBERT...) để phân loại 3 nhãn.

Tách khỏi models.py để baseline TF-IDF không phải import torch.
Cần transformers 4.x: bản 5.x tách token ViSoBERT sai (xem docs/decisions.md, D-016).
Mô hình sau khi train được bọc trong TransformerClassifier, có predict/predict_proba
giống Pipeline của scikit-learn, nên evaluate.py và moderation.py dùng chung được.
"""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.data import LABEL_COL, LABEL_NAMES, TEXT_COL
from src.evaluate import compute_metrics

CLASSIFIER_CONFIG = "classifier_config.json"


def compute_class_weights(labels: Sequence[int], n_classes: int = len(LABEL_NAMES)) -> list[float]:
    """Trọng số "balanced" giống scikit-learn: n_mẫu / (n_lớp * số mẫu của lớp)."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes)
    if (counts == 0).any():
        raise ValueError(f"Có lớp không xuất hiện trong dữ liệu train: {counts.tolist()}")
    return (len(labels) / (n_classes * counts)).tolist()


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor | None,
    num_items_in_batch: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy có trọng số lớp, chia cho số mẫu trong batch.

    weight=None nghĩa là cross-entropy thường.
    """
    if weight is not None:
        weight = weight.to(logits.device)
    total = F.cross_entropy(logits, labels, weight=weight, reduction="sum")
    n_items = labels.numel() if num_items_in_batch is None else num_items_in_batch
    return total / n_items


class EncodedDataset(torch.utils.data.Dataset):
    """Dữ liệu đã tách token, trả từng mẫu dạng dict cho Trainer (chưa padding)."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def token_lengths(texts: Sequence[str], tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Số token của từng câu (kể cả token đặc biệt), dùng để chọn max_length trên train."""
    return [len(ids) for ids in tokenizer(list(texts))["input_ids"]]


class TransformerClassifier:
    """Bọc mô hình + tokenizer, có predict/predict_proba như scikit-learn."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Xác suất của 3 nhãn cho từng câu, mảng (số câu, 3); cột i là nhãn id i."""
        texts = list(texts)
        device = next(self.model.parameters()).device
        self.model.eval()
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                inputs = self.tokenizer(
                    texts[start : start + self.batch_size],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                logits = self.model(**inputs).logits
                batches.append(torch.softmax(logits, dim=-1).cpu().numpy())
        if not batches:
            return np.empty((0, len(LABEL_NAMES)))
        return np.concatenate(batches)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Nhãn id có xác suất cao nhất cho từng câu."""
        return self.predict_proba(texts).argmax(axis=1)

    def save(self, path: str | Path) -> None:
        """Lưu mô hình, tokenizer và max_length vào một thư mục."""
        path = Path(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        settings = {"max_length": self.max_length, "batch_size": self.batch_size}
        (path / CLASSIFIER_CONFIG).write_text(json.dumps(settings), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TransformerClassifier":
        """Nạp lại từ thư mục đã save(); tự dùng GPU nếu có."""
        path = Path(path)
        settings = json.loads((path / CLASSIFIER_CONFIG).read_text(encoding="utf-8"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path)
        return cls(model, tokenizer, **settings)


def _trainer_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Chỉ số Trainer đo trên dev mỗi epoch; macro_f1 dùng để giữ epoch tốt nhất."""
    predictions = np.asarray(eval_pred.predictions).argmax(axis=1)
    metrics = compute_metrics(np.asarray(eval_pred.label_ids).tolist(), predictions.tolist())
    return {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]}


def train_transformer(
    config: dict[str, Any],
    train: pd.DataFrame,
    dev: pd.DataFrame,
    checkpoints_dir: str | Path,
) -> TransformerClassifier:
    """Fine-tune trên train, đo dev mỗi epoch, trả về mô hình ở epoch có macro-F1 dev cao nhất.

    config["model"]: pretrained, max_length, class_weight (null | balanced).
    config["training"]: learning_rate, batch_size, epochs, warmup_ratio, weight_decay,
    fp16, max_train_samples (null = dùng hết; đặt số nhỏ để chạy thử).
    """
    model_cfg, train_cfg, seed = config["model"], config["training"], config["seed"]
    # Nhiều GPU: Trainer tự chia batch cho từng GPU, batch thật = batch_size x số GPU (D-018).
    if torch.cuda.device_count() > 1:
        raise RuntimeError(
            f"Thấy {torch.cuda.device_count()} GPU; batch size thật sẽ bị nhân lên. "
            "Chỉ dùng 1 GPU: đặt CUDA_VISIBLE_DEVICES=0 trước lệnh train."
        )
    if not 0 <= train_cfg["warmup_ratio"] < 1:
        raise ValueError(f"warmup_ratio phải trong [0, 1): {train_cfg['warmup_ratio']}")

    if train_cfg.get("max_train_samples"):
        train = train.sample(n=train_cfg["max_train_samples"], random_state=seed)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["pretrained"],
        num_labels=len(LABEL_NAMES),
        id2label=LABEL_NAMES,
        label2id={name: label for label, name in LABEL_NAMES.items()},
    )

    max_length = model_cfg["max_length"]
    train_set = EncodedDataset(train[TEXT_COL], train[LABEL_COL], tokenizer, max_length)
    dev_set = EncodedDataset(dev[TEXT_COL], dev[LABEL_COL], tokenizer, max_length)

    weight = None
    if model_cfg.get("class_weight") == "balanced":
        weight = torch.tensor(compute_class_weights(train[LABEL_COL].tolist()))
    elif model_cfg.get("class_weight") is not None:
        raise ValueError(f"class_weight chỉ nhận null hoặc balanced: {model_cfg['class_weight']}")

    def loss_func(outputs: Any, labels: torch.Tensor, num_items_in_batch: Any = None) -> Any:
        return weighted_cross_entropy(outputs.logits, labels, weight, num_items_in_batch)

    args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"] * 2,
        num_train_epochs=train_cfg["epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        save_only_model=True,
        fp16=bool(train_cfg.get("fp16")) and torch.cuda.is_available(),
        seed=seed,
        data_seed=seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=train_set,
        eval_dataset=dev_set,
        processing_class=tokenizer,
        compute_loss_func=loss_func,
        compute_metrics=_trainer_metrics,
    )
    trainer.train()
    # Mô hình epoch tốt nhất đã nằm trong trainer.model và sẽ được lưu riêng; xóa checkpoint.
    shutil.rmtree(checkpoints_dir, ignore_errors=True)
    return TransformerClassifier(trainer.model, tokenizer, max_length)


def main() -> None:
    """In phân bố số token trên TRAIN để chọn max_length (không đọc dev/test)."""
    import argparse

    from src.data import load_vihsd
    from src.utils import load_config

    parser = argparse.ArgumentParser(description="Đo độ dài token trên train để chọn max_length.")
    parser.add_argument("--config", required=True)
    config = load_config(parser.parse_args().config)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained"])
    lengths = pd.Series(token_lengths(load_vihsd("train", config["data_dir"])[TEXT_COL], tokenizer))
    print(lengths.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(1).to_string())
    for limit in (64, 96, 128, 256):
        print(f"Câu dài hơn {limit} token (bị cắt): {(lengths > limit).mean():.2%}")


if __name__ == "__main__":
    main()
