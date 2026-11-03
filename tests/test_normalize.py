import unicodedata

import pytest

from src.normalize import (
    apply_outside_urls,
    collapse_whitespace,
    load_lexicon,
    normalize,
    reduce_elongation,
    remove_inserted_chars,
    replace_digit_letters,
    replace_teencode,
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


@pytest.mark.parametrize(
    "text",
    ["xoong nồi", "boong tàu", "1000 đồng", "đii", "!!!", "😂😂😂", "kkkk", "Hhhhh", "kkkk:)))"],
)
def test_reduce_elongation_leaves_normal_text(text):
    # "kkkk" là tiếng cười; rút về "k" sẽ thành teencode của "không".
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


@pytest.mark.parametrize("text", ["(j4f)", "kiss0fangel", "fSypgwW1L", "z1z", "W0W"])
def test_replace_digit_letters_skips_tokens_with_non_vietnamese_letters(text):
    # f/j/w/z không có trong bảng chữ cái tiếng Việt: viết tắt, tiếng Anh hoặc mã (D-021).
    assert replace_digit_letters(text) == text


@pytest.mark.parametrize("text", ["M4A4", "A4B4"])
def test_replace_digit_letters_skips_codes_with_several_digits(text):
    assert replace_digit_letters(text) == text


def test_replace_digit_letters_skips_tokens_longer_than_a_syllable():
    # Câu viết liền: "1" ở đây là số thật ("1 cái"), không phải chữ "i".
    assert replace_digit_letters("toimua1caiao") == "toimua1caiao"
    assert replace_digit_letters("ngh13ng") == "ngh13ng"  # 2 chữ số
    assert replace_digit_letters("nghi3ng") == "nghieng"  # đúng 7 ký tự: vẫn sửa


def test_replace_digit_letters_checks_each_token_separately():
    assert replace_digit_letters("j4f ch3t") == "j4f chet"


def test_apply_outside_urls_protects_links_for_a_single_rule():
    text = "b4n xem https://youtu.be/aB1cd nhé"
    assert apply_outside_urls(text, replace_digit_letters) == "ban xem https://youtu.be/aB1cd nhé"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("n.g.u", "ngu"),
        ("c.h.ế.t", "chết"),
        ("đ.iên", "điên"),
        ("n-g-u", "ngu"),
        ("n_g_u", "ngu"),
        ("n*g*u", "ngu"),  # "*" từ 2 lần: tách từng chữ, là ký tự chèn
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
        "K-pop",  # một gạch ngang: thường là tên riêng
        "G-Dragon",
        "K-icm ca sĩ",
        "n-gu",
        "x*u",  # "*" che một chữ ("xấu"): bỏ đi sẽ thành từ khác ("xu")
        "ng*c",
        "x*u n*g",  # mỗi chuỗi chỉ có một "*", dù cả câu có hai
        "được k.tết rồi",  # thiếu dấu cách; "k" = "không" trong teencode
        "hay k M.n",  # "M.n" = "mọi người"
        "bậy.T like",
        "c.bi",
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


# Từ điển nhỏ viết trong test (D-006): test không phụ thuộc file resources/teencode.tsv.
LEXICON = {"ko": "không", "vn": "Việt Nam", "v": "vậy", "z": "vậy", "h": "giờ", "ae": "anh em"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ko biết", "không biết"),
        ("Ko biết", "Không biết"),  # giữ chữ hoa đầu
        ("KO", "Không"),
        ("vn mình", "Việt Nam mình"),
        ("thế ko.", "thế không."),
        ("(ko)", "(không)"),
        ("ko,ae ơi", "không,anh em ơi"),
        ("đi z?", "đi vậy?"),
    ],
)
def test_replace_teencode_fixes_whole_tokens(text, expected):
    assert replace_teencode(text, LEXICON) == expected


@pytest.mark.parametrize(
    "text",
    [
        "kong",  # chỉ thay nguyên token, không thay một phần
        "vnexpress",
        "5h chiều",  # dính số
        "h5",
        "hay :v",  # mặt cười
        ":vv",
        "=]]z",
        ":))z",
        "ok ;v",
        "abc.vn",  # tên miền
        "@ko",  # tag
        "a/ko",
        "xem https://ko.vn/ko nhé",  # trong link (chữ "nhé" không có trong từ điển)
        "https://youtu.be/xem-ko-h",  # "-" không chặn: chỉ việc bảo vệ link giữ được
        "được k.tết",  # "k" không có trong LEXICON; "." bên trái thì cũng không thay
    ],
)
def test_replace_teencode_leaves_other_text(text):
    assert replace_teencode(text, LEXICON) == text


def test_replace_teencode_skips_token_after_dot_even_if_in_lexicon():
    # Đánh đổi đã chấp nhận: "x.ko" có thể là thiếu dấu cách, nhưng cũng giống tên miền.
    assert replace_teencode("thế.ko", LEXICON) == "thế.ko"


def test_normalize_applies_lexicon_after_rules():
    # "kooo" được lớp luật rút về "ko" trước, rồi mới tra từ điển.
    assert normalize("  kooo   biết  ", LEXICON) == "không biết"
    assert normalize("k.o biết", LEXICON) == "không biết"


def test_normalize_without_lexicon_keeps_teencode():
    # Mặc định chỉ chạy lớp luật: công cụ tìm ứng viên (keywords.py) cần thấy "ko".
    assert normalize("ko biết") == "ko biết"
    assert normalize("ko biết", {}) == "ko biết"


@pytest.mark.parametrize("text", ["ko biết z", "Ko ae ơi :v", "vn vn vn", "kooo h"])
def test_normalize_with_lexicon_is_idempotent(text):
    once = normalize(text, LEXICON)
    assert normalize(once, LEXICON) == once


def _write_tsv(path, rows):
    header = "token\tdecision\treplacement\n"
    path.write_text(header + "".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_load_lexicon_keeps_only_keep_rows(tmp_path):
    path = _write_tsv(tmp_path / "lex.tsv", [("ko", "keep", "không"), ("m", "reject", "")])
    assert load_lexicon(path) == {"ko": "không"}


@pytest.mark.parametrize(
    "rows",
    [
        [("ko", "keep", "")],  # keep thiếu replacement
        [("ko", "keep", "không"), ("ko", "reject", "")],  # token trùng
        [("ae", "keep", "anh em"), ("em", "keep", "em ấy")],  # thay dây chuyền
    ],
)
def test_load_lexicon_rejects_broken_file(tmp_path, rows):
    with pytest.raises(ValueError):
        load_lexicon(_write_tsv(tmp_path / "lex.tsv", rows))


def test_load_lexicon_reads_committed_file():
    # Số mục keep sau khi chấm (01/11/2026); sửa số này khi duyệt lại từ điển.
    lexicon = load_lexicon()
    assert len(lexicon) == 83
    assert lexicon["ko"] == "không"
    assert all(token == token.lower() for token in lexicon)
