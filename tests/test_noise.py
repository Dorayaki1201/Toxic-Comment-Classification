import hashlib
import random
import re
import unicodedata
from collections import Counter

import pandas as pd
import pytest

from src.noise import (
    DEFAULT_ABBREV_PATH,
    INSERT_CHARS,
    NFC_ONLY,
    TRANSFORMS,
    NoisySpec,
    abbreviate_word,
    build_abbreviations,
    build_noisy_set,
    compare_with_manifest,
    count_abbreviations,
    count_changed_words,
    count_non_nfc,
    csv_bytes,
    digit_word,
    elongate_word,
    explained_by_char_transforms,
    extract_word_pairs,
    insert_word,
    load_abbreviations,
    make_noisy_test_sets,
    noisy_test_specs,
    perturb,
    perturb_texts,
    rng_for,
    save_abbreviations,
    strip_diacritics,
    teencode_word,
)

# Ví dụ dùng từ nhẹ thay cho từ thô tục (CLAUDE.md mục 10.7).
SENTENCE = "Thằng này nói gì vậy, phải đi học chứ con ơi"
ALL = list(TRANSFORMS)


def rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


@pytest.mark.parametrize("name", ALL)
def test_rate_zero_changes_nothing(name):
    assert perturb(SENTENCE, [name], 0.0, rng()) == SENTENCE


@pytest.mark.parametrize("name", ALL)
def test_rate_one_changes_sentence(name):
    assert perturb(SENTENCE, [name], 1.0, rng()) != SENTENCE


@pytest.mark.parametrize("names", [[name] for name in ALL] + [ALL])
@pytest.mark.parametrize(
    "text", ["https://www.youtube.com/watch?v=abcdef", "50k 2020 5h", "😂😂 :))) !!!", ""]
)
def test_leaves_links_numbers_and_emoji_alone(names, text):
    assert perturb(text, names, 1.0, rng()) == text


def test_link_inside_sentence_is_kept():
    text = "xem đi nè https://vn.example.com/phai-di rồi nói"
    noisy = perturb(text, ALL, 1.0, rng())
    assert "https://vn.example.com/phai-di" in noisy


@pytest.mark.parametrize(
    ("word", "expected"),
    [("Đường", "Duong"), ("thằng", "thang"), ("NGƯỜI", "NGUOI"), ("abc", "abc")],
)
def test_strip_diacritics(word, expected):
    assert strip_diacritics(word) == expected


def test_remove_diacritics_only_strips_marks():
    noisy = perturb(SENTENCE, ["remove_diacritics"], 0.5, rng())
    assert noisy != SENTENCE
    assert strip_diacritics(noisy) == strip_diacritics(SENTENCE)


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("phở", "fở"),
        ("Phở", "Fở"),
        ("PHỞ", "FỞ"),
        ("quá", "wá"),
        ("giờ", "jờ"),
        ("gì", "j"),
        ("con", "kon"),
        ("vậy", "zậy"),
        ("ơi", "ơj"),
        ("người", "ngườj"),  # dấu nằm ở "ờ", chữ "i" cuối không dấu
    ],
)
def test_teencode_rules(word, expected):
    # Các từ này chỉ khớp đúng một luật, nên kết quả không phụ thuộc seed.
    assert teencode_word(word, rng()) == expected


@pytest.mark.parametrize("word", ["chứ", "Hà", "mày", "bì"])
def test_teencode_leaves_words_without_matching_rule(word):
    # "ch" không thành "kh"; "ì" có dấu ở cuối không thành "j".
    assert teencode_word(word, rng()) == word


@pytest.mark.parametrize("seed", range(20))
def test_elongate_repeats_one_vowel_2_to_5_times(seed):
    noisy = elongate_word("điên", rng(seed))
    assert re.fullmatch(r"đi{2,5}ên|đie{2,5}n|điê{2,5}n", noisy)


def test_elongate_skips_words_without_vowel():
    assert elongate_word("đm", rng()) == "đm"


@pytest.mark.parametrize("seed", range(20))
def test_digit_replaces_exactly_one_letter(seed):
    word = "chết"
    noisy = digit_word(word, rng(seed))
    diffs = [(a, b) for a, b in zip(word, noisy, strict=True) if a != b]
    assert len(diffs) == 1
    assert diffs[0][1] in "0134$"


def test_digit_can_hit_end_of_word():
    # Dạng mà lớp luật 3a cố ý bỏ qua (D-021): nhiễu vẫn phải sinh ra được.
    assert {digit_word("ngo", rng(seed)) for seed in range(20)} == {"ng0"}


@pytest.mark.parametrize("seed", range(20))
def test_insert_only_adds_one_kind_of_separator(seed):
    noisy = insert_word("ngu", rng(seed))
    added = {char for char in noisy if char in INSERT_CHARS}
    assert len(added) == 1
    assert noisy.replace(added.pop(), "") == "ngu"


def test_insert_skips_single_letters():
    assert insert_word("k", rng()) == "k"


def test_same_seed_gives_same_output():
    texts = [SENTENCE] * 5
    first = perturb_texts(texts, ALL, 0.5, seed=7)
    assert first == perturb_texts(texts, ALL, 0.5, seed=7)
    assert first != perturb_texts(texts, ALL, 0.5, seed=8)


def test_each_sentence_has_its_own_rng():
    # Câu giống nhau ở vị trí khác nhau thì nhiễu khác nhau (không phải chép cùng một mẫu).
    a, b = perturb_texts([SENTENCE, SENTENCE], ALL, 0.5, seed=7)
    assert a != b
    # Đổi câu 0 không làm đổi nhiễu của câu 1.
    other = perturb_texts(["câu khác hẳn", SENTENCE], ALL, 0.5, seed=7)
    assert other[1] == b


def test_rng_for_is_stable_across_processes():
    # Giá trị cố định: nếu Python đổi cách băm seed chuỗi, bộ test nhiễu sẽ không tái lập được.
    assert rng_for(42, 0).random() == 0.7978966933582126
    assert rng_for(42, 0).random() != rng_for(42, 1).random()


def test_order_of_transforms_argument_does_not_matter():
    assert perturb(SENTENCE, ALL, 0.5, rng(3)) == perturb(SENTENCE, ALL[::-1], 0.5, rng(3))


@pytest.mark.parametrize("names", [["remove_diacritics"], ALL])
def test_rate_is_fraction_of_words_changed(names):
    # Ghép nhiều biến đổi KHÔNG được làm tăng tỉ lệ từ bị đổi (lỗi đã gặp: 0.3 thành ~0.83).
    # "đường" áp được mọi biến đổi; mỗi từ cách nhau dấu cách, ký tự chèn không chứa cách.
    words = ["đường"] * 2000
    noisy = perturb(" ".join(words), names, 0.3, rng(1)).split()
    assert len(noisy) == len(words)
    changed = sum(word != "đường" for word in noisy) / len(words)
    assert 0.27 < changed < 0.33


@pytest.mark.parametrize("seed", range(30))
def test_each_word_gets_only_one_transform(seed):
    # Một từ nhận nhiều biến đổi chồng nhau sẽ có cả ký tự chèn lẫn số, hoặc chữ lặp và mất dấu.
    noisy = perturb("đường", ALL, 1.0, rng(seed))
    kinds = [
        any(char in INSERT_CHARS for char in noisy),
        any(char in "0134$" for char in noisy),
        "ư" not in noisy and "ờ" not in noisy and "đ" not in noisy,  # mất dấu
        bool(re.search(r"(\w)\1", noisy)),  # kéo dài
    ]
    assert sum(kinds) == 1, noisy


def test_input_is_normalized_to_nfc():
    nfd = unicodedata.normalize("NFD", "thằng")
    assert perturb(nfd, [], 1.0, rng()) == "thằng"


@pytest.mark.parametrize(
    ("names", "rate"), [(["no_such"], 0.5), (["elongate"], 1.5), (["abbreviate"], 0.5)]
)
def test_invalid_arguments_raise(names, rate):
    # "abbreviate" không có bảng viết tắt cũng phải báo lỗi, không lặng lẽ bỏ qua.
    with pytest.raises(ValueError):
        perturb(SENTENCE, names, rate, rng())


# --- Bảng viết tắt (D-024). Bảng nhỏ viết trong code, không đọc ViLexNorm. ---

ABBREV = {"không": {"k": 3, "ko": 1}, "tôi": {"t": 1}}


@pytest.mark.parametrize(
    ("original", "normalized", "expected"),
    [
        ("t k biết", "tôi không biết", [("t", "tôi"), ("k", "không")]),
        ("Ê k khóc được =))?", "Ê không khóc được =))?", [("k", "không")]),
        ("thế k?", "thế không?", [("k", "không")]),  # dấu câu dính vào từ
        ("ng0 quá", "ngo quá", []),  # có số: không phải chữ cái, bỏ
        ("đ.ịt", "địt", []),
        ("ntn", "như thế nào", []),  # 1 -> 3 từ: bỏ
        ("y hệt", "y hệt", []),
    ],
)
def test_extract_word_pairs(original, normalized, expected):
    assert extract_word_pairs(original, normalized) == expected


@pytest.mark.parametrize(
    ("noisy", "standard"),
    [("nóiii", "nói"), ("khong", "không"), ("fải", "phải"), ("j", "gì"), ("Khong", "không")],
)
def test_explained_by_char_transforms(noisy, standard):
    assert explained_by_char_transforms(noisy, standard)


@pytest.mark.parametrize(
    ("noisy", "standard"), [("ko", "không"), ("thoai", "thôi"), ("đc", "được"), ("t", "tôi")]
)
def test_not_explained_by_char_transforms(noisy, standard):
    assert not explained_by_char_transforms(noisy, standard)


def test_count_abbreviations_skips_char_noise_and_merges_elongation():
    pairs = [("ko nóiii", "không nói"), ("kooo khong", "không không"), ("lunnn", "luôn")]
    assert count_abbreviations(pairs) == Counter({("không", "ko"): 2, ("luôn", "lun"): 1})


def test_build_abbreviations_applies_min_count_and_sorts():
    counts = Counter({("không", "ko"): 2, ("không", "k"): 5, ("tôi", "t"): 1})
    table = build_abbreviations(counts, min_count=2)
    assert table == {"không": {"k": 5, "ko": 2}}
    assert list(table["không"]) == ["k", "ko"]


def test_save_and_load_abbreviations_roundtrip(tmp_path):
    path = tmp_path / "abbrev.tsv"
    save_abbreviations(ABBREV, path)
    assert load_abbreviations(path) == ABBREV


def test_load_abbreviations_rejects_duplicates(tmp_path):
    path = tmp_path / "abbrev.tsv"
    path.write_text("normalized\tvariant\tcount\nkhông\tk\t3\nkhông\tk\t2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_abbreviations(path)


def test_abbreviate_follows_frequency_and_keeps_case():
    results = Counter(abbreviate_word("không", rng(seed), ABBREV) for seed in range(2000))
    assert set(results) == {"k", "ko"}
    assert 0.70 < results["k"] / 2000 < 0.80  # trọng số 3 : 1
    assert abbreviate_word("Không", rng(), {"không": {"ko": 1}}) == "Ko"
    assert abbreviate_word("bạn", rng(), ABBREV) == "bạn"


def test_perturb_with_abbreviations():
    noisy = perturb("Tôi không biết", ["abbreviate"], 1.0, rng(), ABBREV)
    assert noisy in {"T k biết", "T ko biết"}


def test_committed_abbreviation_table_loads():
    # File thật trong resources/ phải đọc được và không có dạng kéo dài (đã gộp khi đếm).
    table = load_abbreviations(DEFAULT_ABBREV_PATH)
    assert table["không"]
    assert not any(re.search(r"(.)\1\1", v) for variants in table.values() for v in variants)


# --- Bộ test nhiễu (D-025). DataFrame nhỏ viết trong code, không đọc ViHSD. ---

CONFIG = {"levels": {"light": 0.1, "medium": 0.3, "heavy": 0.5}, "transforms": ["abbreviate", *ALL]}
NFD = unicodedata.normalize("NFD", "thằng này")


def small_df() -> pd.DataFrame:
    texts = [SENTENCE, "Tôi không biết", "", "50k 2020", SENTENCE.upper(), "phải đi học"]
    return pd.DataFrame({"free_text": texts, "label_id": [0, 1, 2, 0, 1, 2]})


def test_noisy_test_specs_are_six_single_plus_three_combined():
    specs = noisy_test_specs(CONFIG)
    assert [s.name for s in specs] == [f"{n}_medium" for n in CONFIG["transforms"]] + [
        "all_light",
        "all_medium",
        "all_heavy",
    ]
    assert all(s.rate == 0.3 and len(s.transforms) == 1 for s in specs[:6])
    assert [s.rate for s in specs[6:]] == [0.1, 0.3, 0.5]
    assert all(set(s.transforms) == set(CONFIG["transforms"]) for s in specs[6:])


def test_build_noisy_set_keeps_rows_order_and_labels():
    df = small_df()
    noisy = build_noisy_set(df, NoisySpec("all_heavy", tuple(ALL), 1.0), 1105, ABBREV)
    assert list(noisy.columns) == ["row_id", "free_text", "label_id", "changed"]
    assert noisy["row_id"].tolist() == list(range(len(df)))
    assert noisy["label_id"].tolist() == df["label_id"].tolist()
    # Câu rỗng (dòng lỗi, D-004) và câu không có chữ thì giữ nguyên, changed = False.
    assert noisy["free_text"][2] == "" and not noisy["changed"][2]
    assert noisy["free_text"][3] == "50k 2020" and not noisy["changed"][3]
    assert noisy["changed"][0]
    assert noisy["changed"].tolist() == (noisy["free_text"] != df["free_text"]).tolist()


def test_build_noisy_set_is_deterministic_and_streams_are_independent():
    df = pd.DataFrame({"free_text": [SENTENCE] * 20, "label_id": [0] * 20})
    first = build_noisy_set(df, NoisySpec("a", tuple(ALL), 0.5), 1105)
    assert first.equals(build_noisy_set(df, NoisySpec("a", tuple(ALL), 0.5), 1105))
    # Cùng biến đổi, cùng rate, cùng seed, khác tên bộ: phải khác (luồng riêng).
    other = build_noisy_set(df, NoisySpec("b", tuple(ALL), 0.5), 1105)
    assert (first["free_text"] != other["free_text"]).sum() > 10


def test_stream_none_keeps_old_behaviour():
    # Bản xem thử trên train và test cũ phải ra đúng như trước khi thêm `stream`.
    texts = [SENTENCE] * 3
    old = [perturb(t, ALL, 0.5, rng_for(7, i)) for i, t in enumerate(texts)]
    assert perturb_texts(texts, ALL, 0.5, seed=7) == old
    assert perturb_texts(texts, ALL, 0.5, seed=7, stream="x") != old


def test_nfc_only_set_only_normalizes():
    df = pd.DataFrame({"free_text": [NFD, SENTENCE], "label_id": [0, 1]})
    noisy = build_noisy_set(df, NFC_ONLY, 1105)
    assert noisy["free_text"].tolist() == ["thằng này", SENTENCE]
    assert noisy["changed"].tolist() == [False, False]  # so với bản gốc đã NFC


def test_nfc_control_is_added_only_when_needed():
    assert count_non_nfc(small_df()["free_text"]) == 0
    assert "nfc_only" not in make_noisy_test_sets(small_df(), CONFIG, 1105, ABBREV)
    df = pd.concat([small_df(), pd.DataFrame({"free_text": [NFD], "label_id": [0]})])
    sets = make_noisy_test_sets(df.reset_index(drop=True), CONFIG, 1105, ABBREV)
    assert len(sets) == 10 and "nfc_only" in sets


def test_compare_with_manifest_detects_changes():
    sets = make_noisy_test_sets(small_df(), CONFIG, 1105, ABBREV)
    manifest = {
        "sets": {n: {"sha256": hashlib.sha256(csv_bytes(d)).hexdigest()} for n, d in sets.items()}
    }
    assert compare_with_manifest(sets, manifest) == []
    # Seed khác -> hash khác: manifest bắt được.
    other = make_noisy_test_sets(small_df(), CONFIG, 1106, ABBREV)
    assert compare_with_manifest(other, manifest)
    # Thiếu một bộ cũng phải báo.
    del sets["all_light"]
    assert compare_with_manifest(sets, manifest) == ["all_light"]


def test_csv_bytes_use_lf_and_utf8():
    data = csv_bytes(pd.DataFrame({"free_text": ["thằng"], "label_id": [0]}))
    assert b"\r\n" not in data and "thằng".encode() in data


def test_count_changed_words():
    assert count_changed_words("tôi không biết", "tôi ko biết") == 1
    assert count_changed_words("tôi không biết", "tôi không biết") == 0
