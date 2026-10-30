import unicodedata

import pytest

from src.normalize import (
    collapse_whitespace,
    normalize,
    reduce_elongation,
    remove_inserted_chars,
    replace_digit_letters,
)

# Mỗi luật có hai nhóm test: "phải sửa" (cố ý lách) và "không được đụng" (câu bình thường).
# Ví dụ dùng từ nhẹ thay cho từ thô tục (CLAUDE.md mục 10.7).


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("điiiiên", "điên"),
        ("nguuuu", "ngu"),
        ("NGUUUU", "NGU"),
        ("quáaaa đi", "quáa đi"),  # hạn chế đã biết: "á" và "a" là hai ký tự khác nhau
        ("ngUuU", "ngU"),  # lặp không phân biệt hoa/thường
    ],
)
def test_reduce_elongation_fixes_repeated_letters(text, expected):
    assert reduce_elongation(text) == expected


@pytest.mark.parametrize("text", ["xoong nồi", "boong tàu", "1000 đồng", "đii", "!!!", "😂😂😂"])
def test_reduce_elongation_leaves_normal_text(text):
    assert reduce_elongation(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [("ch3t", "chet"), ("b4n", "ban"), ("ng0c", "ngoc"), ("d1t", "dit"), ("cu$t", "cust")],
)
def test_replace_digit_letters_fixes_digits_between_letters(text, expected):
    assert replace_digit_letters(text) == expected


@pytest.mark.parametrize(
    "text", ["5h chiều", "50k", "năm 2020", "covid19", "mp3", "ab@gmail.com", "3.5 điểm"]
)
def test_replace_digit_letters_leaves_numbers_alone(text):
    assert replace_digit_letters(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("n.g.u", "ngu"),
        ("c.h.ế.t", "chết"),
        ("đ.iên", "điên"),
        ("n-g-u", "ngu"),
        ("n_g_u", "ngu"),
        ("n*g*u", "ngu"),
        ("d!t", "dit"),
        ("đ.c.m", "đcm"),
        ("thằng n.g.u quá", "thằng ngu quá"),
    ],
)
def test_remove_inserted_chars_fixes_obfuscated_words(text, expected):
    assert remove_inserted_chars(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "ừ.thế thôi",  # thiếu khoảng trắng sau dấu chấm, hai bên là âm tiết thật
        "hay!quá",
        "đm.mày",  # "đm" có 2 chữ, không phải phụ âm đơn lẻ
        "e-mail",
        "3.5",
        "anh_em",
        "Hà Nội.",
    ],
)
def test_remove_inserted_chars_leaves_missing_space_and_normal_text(text):
    assert remove_inserted_chars(text) == text


def test_collapse_whitespace():
    assert collapse_whitespace("  mèo \t đẹp\n\nquá  ") == "mèo đẹp quá"


def test_normalize_combines_rules_in_order():
    assert normalize("  thằng   n.g.uuuu  ch3t  ") == "thằng ngu chet"


def test_normalize_keeps_urls_untouched():
    text = "xem www.example.com/aaaa và https://a.b-c.vn/x_y nhé điiiiên"
    assert normalize(text) == "xem www.example.com/aaaa và https://a.b-c.vn/x_y nhé điên"


def test_normalize_applies_nfc_first():
    decomposed = unicodedata.normalize("NFD", "điiiiên")
    assert normalize(decomposed) == "điên"


@pytest.mark.parametrize(
    "text",
    ["", "mèo đáng yêu quá 😍", "ko bít nữa", "5h chiều gặp nhé", "ừ.thế thôi", "xoong nồi"],
)
def test_normalize_does_not_change_clean_text(text):
    assert normalize(text) == unicodedata.normalize("NFC", text)


@pytest.mark.parametrize(
    "text", ["n.g.uuuu", "c.h.3.t", "đ.iiiiên quá!!!", "ch3t   đi", "www.aaa.com n.g.u"]
)
def test_normalize_is_idempotent(text):
    # Chạy hai lần phải ra cùng kết quả: nếu không, luật đang tạo ra chỗ cần sửa mới.
    once = normalize(text)
    assert normalize(once) == once
