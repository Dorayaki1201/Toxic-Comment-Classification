import random
import re
import unicodedata

import pytest

from src.noise import (
    INSERT_CHARS,
    TRANSFORMS,
    digit_word,
    elongate_word,
    insert_word,
    perturb,
    perturb_texts,
    rng_for,
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


@pytest.mark.parametrize(("names", "rate"), [(["no_such"], 0.5), (["elongate"], 1.5)])
def test_invalid_arguments_raise(names, rate):
    with pytest.raises(ValueError):
        perturb(SENTENCE, names, rate, rng())
