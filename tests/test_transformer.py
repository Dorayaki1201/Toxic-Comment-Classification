import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.utils.class_weight import compute_class_weight
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from src.data import LABEL_COL, TEXT_COL
from src.evaluate import evaluate_saved_model
from src.models import load_model, model_type
from src.train import run
from src.transformer import (
    TransformerClassifier,
    compute_class_weights,
    token_lengths,
    weighted_cross_entropy,
)

# Dữ liệu đồ chơi trung tính: nhãn chỉ để kiểm tra luồng code, không mang nghĩa thật.
TEXTS = ["mèo đáng yêu", "mèo ngủ ngon", "chó sủa to", "chó chạy nhanh", "cá bơi lội", "cá bơi xa"]
LABELS = [0, 0, 1, 1, 2, 2]
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


@pytest.fixture
def tiny_pretrained(tmp_path):
    """Mô hình BERT tí hon + tokenizer từ vựng nhỏ, lưu ra thư mục: không cần tải gì từ mạng."""
    words = sorted({word for text in TEXTS + ["rùa bò chậm"] for word in text.split()})
    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS + words)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=3,
    )
    path = tmp_path / "tiny"
    BertForSequenceClassification(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def classifier_from(pretrained):
    model = AutoModelForSequenceClassification.from_pretrained(pretrained)
    return TransformerClassifier(model, AutoTokenizer.from_pretrained(pretrained), max_length=16)


def transformer_config(pretrained, data_dir, class_weight=None):
    return {
        "name": "toy_bert",
        "type": "transformer",
        "seed": 0,
        "data_dir": str(data_dir),
        "model": {"pretrained": str(pretrained), "max_length": 16, "class_weight": class_weight},
        "training": {
            "learning_rate": 1.0e-3,
            "batch_size": 4,
            "epochs": 2,
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "fp16": False,
            "max_train_samples": None,
        },
    }


def write_toy_data(data_dir):
    data_dir.mkdir()
    for split in ["train", "dev", "test"]:
        pd.DataFrame({TEXT_COL: TEXTS + ["#ERROR!"], LABEL_COL: LABELS + [0]}).to_csv(
            data_dir / f"{split}.csv", index=False
        )


def write_config(tmp_path, config):
    path = tmp_path / f"{config['name']}.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def test_compute_class_weights_matches_scikit_learn():
    labels = [0] * 8 + [1] * 1 + [2] * 3
    expected = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.array(labels))
    assert compute_class_weights(labels) == pytest.approx(expected.tolist())


def test_compute_class_weights_rejects_missing_class():
    with pytest.raises(ValueError, match="không xuất hiện"):
        compute_class_weights([0, 0, 2])


def test_weighted_cross_entropy_without_weight_equals_plain_mean():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    labels = torch.tensor([0, 2, 2])
    expected = torch.nn.functional.cross_entropy(logits, labels)
    assert weighted_cross_entropy(logits, labels, None).item() == pytest.approx(expected.item())


def test_weighted_cross_entropy_penalizes_mistakes_on_heavy_class_more():
    logits = torch.tensor([[0.0, 0.0, 2.0]])  # đoán HATE
    labels = torch.tensor([1])  # nhãn thật OFFENSIVE
    plain = weighted_cross_entropy(logits, labels, None)
    heavy = weighted_cross_entropy(logits, labels, torch.tensor([1.0, 5.0, 1.0]))
    assert heavy.item() == pytest.approx(5 * plain.item())


def test_model_type_defaults_to_tfidf_and_rejects_unknown():
    assert model_type({"name": "x"}) == "tfidf_lr"
    with pytest.raises(ValueError, match="type"):
        model_type({"type": "svm"})


def test_token_lengths_counts_special_tokens(tiny_pretrained):
    classifier = classifier_from(tiny_pretrained)
    lengths = token_lengths(["mèo đáng yêu", "cá"], classifier.tokenizer)
    assert len(lengths) == 2
    assert lengths[0] > lengths[1]


def test_train_save_load_and_evaluate_end_to_end(tmp_path, tiny_pretrained):
    data_dir = tmp_path / "data"
    write_toy_data(data_dir)
    config = transformer_config(tiny_pretrained, data_dir, class_weight="balanced")
    config_path = write_config(tmp_path, config)
    models_dir, reports_dir = tmp_path / "models", tmp_path / "reports"

    report = run(config_path, models_dir=models_dir, reports_dir=reports_dir)

    assert (models_dir / "toy_bert" / "classifier_config.json").exists()
    assert {"torch_version", "transformers_version"} <= set(report)
    assert sum(map(sum, report["dev"]["confusion_matrix"])) == 7

    model = load_model(config, models_dir)
    probabilities = model.predict_proba(TEXTS)
    assert probabilities.shape == (6, 3)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(6), abs=1e-5)
    assert model.predict(TEXTS).tolist() == probabilities.argmax(axis=1).tolist()

    metrics = evaluate_saved_model(config_path, "test", models_dir, reports_dir)
    saved = json.loads((reports_dir / "toy_bert" / "test_metrics.json").read_text("utf-8"))
    assert saved["macro_f1"] == metrics["macro_f1"]
    assert metrics["seen_in_train"]["rows"] == 6


def test_saved_model_gives_same_probabilities_after_reload(tmp_path, tiny_pretrained):
    classifier = classifier_from(tiny_pretrained)
    before = classifier.predict_proba(TEXTS)
    classifier.save(tmp_path / "saved")
    after = TransformerClassifier.load(tmp_path / "saved").predict_proba(TEXTS)
    assert after == pytest.approx(before, abs=1e-6)


def test_predict_proba_on_empty_input_returns_empty_array(tiny_pretrained):
    classifier = classifier_from(tiny_pretrained)
    assert classifier.predict_proba([]).shape == (0, 3)


def test_training_refuses_multiple_gpus(tmp_path, tiny_pretrained, monkeypatch):
    # Nhiều GPU làm batch thật = batch_size x số GPU mà không báo (D-018).
    data_dir = tmp_path / "data"
    write_toy_data(data_dir)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    config_path = write_config(tmp_path, transformer_config(tiny_pretrained, data_dir))
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=0"):
        run(config_path, tmp_path / "m", tmp_path / "r")


def test_seed_override_names_outputs_and_removes_checkpoints(tmp_path, tiny_pretrained):
    data_dir = tmp_path / "data"
    write_toy_data(data_dir)
    config_path = write_config(tmp_path, transformer_config(tiny_pretrained, data_dir))
    models_dir, reports_dir = tmp_path / "models", tmp_path / "reports"

    report = run(config_path, models_dir, reports_dir, seed=7)

    assert report["config"] == "toy_bert_s7"
    assert report["config_values"]["seed"] == 7
    assert (models_dir / "toy_bert_s7" / "classifier_config.json").exists()
    assert not (models_dir / "toy_bert_s7_checkpoints").exists()
    metrics = evaluate_saved_model(config_path, "test", models_dir, reports_dir, seed=7)
    assert metrics["config"] == "toy_bert_s7"
    assert (reports_dir / "toy_bert_s7" / "test_metrics.json").exists()


def test_report_records_device(tmp_path, tiny_pretrained):
    data_dir = tmp_path / "data"
    write_toy_data(data_dir)
    config_path = write_config(tmp_path, transformer_config(tiny_pretrained, data_dir))
    report = run(config_path, tmp_path / "m", tmp_path / "r")
    assert report["device"] == "cpu"


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1", reason="cần mạng; đặt RUN_NETWORK_TESTS=1"
)
def test_visobert_tokenizer_keeps_whole_vietnamese_syllables():
    # transformers 5.x tách ViSoBERT sai thành mảnh nhỏ (D-016); test này chặn lỗi đó quay lại.
    tokenizer = AutoTokenizer.from_pretrained("uitnlp/visobert")
    assert tokenizer.tokenize("mèo đáng yêu") == ["▁mèo", "▁đáng", "▁yêu"]
    assert tokenizer("không biết")["input_ids"] == [0, 77, 225, 2]


def test_invalid_class_weight_and_warmup_are_rejected(tmp_path, tiny_pretrained):
    data_dir = tmp_path / "data"
    write_toy_data(data_dir)
    bad_weight = transformer_config(tiny_pretrained, data_dir, class_weight="auto")
    with pytest.raises(ValueError, match="class_weight"):
        run(write_config(tmp_path, bad_weight), tmp_path / "m", tmp_path / "r")
    bad_warmup = transformer_config(tiny_pretrained, data_dir)
    bad_warmup["training"]["warmup_ratio"] = 1
    with pytest.raises(ValueError, match="warmup_ratio"):
        run(write_config(tmp_path, bad_warmup), tmp_path / "m", tmp_path / "r")
