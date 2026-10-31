"""Chuẩn hóa văn bản bằng luật (cấu hình B), lớp không cần từ điển.

Nguyên tắc: thà bỏ sót còn hơn sửa sai. Một luật sửa nhầm câu bình thường sẽ làm hỏng
cả những câu sạch, nên mỗi luật chỉ chạy ở chỗ chắc chắn là cố ý lách.
Không viết thường, không đổi emoji, không thay từ (từ điển teencode là phần riêng).
"""

import re
import unicodedata
from collections.abc import Callable

# Chữ cái (kể cả chữ có dấu tiếng Việt), không gồm số và dấu gạch dưới.
LETTER = r"[^\W\d_]"

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

# Số/ký hiệu hay dùng thay chữ. Không có "@" để không phá địa chỉ email.
DIGIT_LETTERS = {"0": "o", "1": "i", "3": "e", "4": "a", "$": "s"}
DIGIT_BETWEEN_LETTERS = re.compile(rf"(?<={LETTER})[0134$](?={LETTER})")
# Chữ không có trong bảng chữ cái tiếng Việt: token chứa chúng là viết tắt/tiếng Anh/mã
# ("j4f", "fSypgwW1L"), không phải từ tiếng Việt bị thay số, nên không đụng (D-021).
FOREIGN_LETTERS = set("fjwz")
# Âm tiết tiếng Việt dài nhất có 7 chữ ("nghiêng"): token dài hơn là câu viết liền,
# tiếng Anh hoặc tên, không phải một chữ bị thay số.
MAX_SYLLABLE_LENGTH = 7
TOKEN = re.compile(r"\S+")

# Ký tự chèn giữa các chữ: "!" được hiểu là chữ "i", các ký tự khác bị bỏ.
INSERTED_CHARS = ".-_*!"
INSERTED_RUN = re.compile(rf"{LETTER}+(?:[{re.escape(INSERTED_CHARS)}]{LETTER}+)+")

ELONGATED = re.compile(rf"({LETTER})\1{{2,}}", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")

VOWELS = set("aeiouy")


def _has_vowel(piece: str) -> bool:
    """Có nguyên âm không (bỏ dấu trước khi xét: "ế" -> "e")."""
    base = unicodedata.normalize("NFD", piece.lower())
    return any(char in VOWELS for char in base)


def _is_single_consonant(piece: str) -> bool:
    """Một phụ âm đứng một mình, như "đ", "n": không thể là một âm tiết tiếng Việt."""
    return len(piece) == 1 and not _has_vowel(piece)


def replace_digit_letters(text: str) -> str:
    """Thay số/ký hiệu nằm giữa hai chữ cái bằng chữ: "ch3t" -> "chet", "b4n" -> "ban".

    Số đứng đầu/cuối hoặc cạnh số khác không bị đụng ("5h", "50k", "2020"). Bỏ qua cả
    token có chữ f/j/w/z ("j4f"), có từ 2 chữ số ("M4A4"), hoặc dài hơn một âm tiết.
    """

    def fix_token(match: re.Match[str]) -> str:
        token = match.group()
        if (
            FOREIGN_LETTERS & set(token.lower())
            or sum(char.isdigit() for char in token) >= 2
            or sum(char.isalnum() for char in token) > MAX_SYLLABLE_LENGTH
        ):
            return token
        return DIGIT_BETWEEN_LETTERS.sub(lambda digit: DIGIT_LETTERS[digit.group()], token)

    return TOKEN.sub(fix_token, text)


def _merge_inserted(match: re.Match[str]) -> str:
    parts = re.split(rf"([{re.escape(INSERTED_CHARS)}])", match.group())
    pieces, separators = parts[0::2], parts[1::2]
    # Một gạch ngang thường là tên riêng ("K-pop", "G-Dragon"); lách luật thì tách từng chữ.
    lone_hyphen = separators.count("-") < 2
    result = pieces[0]
    for i, separator in enumerate(separators):
        left, right = pieces[i], pieces[i + 1]
        if separator == "-" and lone_hyphen:
            result += separator + right
        elif _is_single_consonant(left) or _is_single_consonant(right):
            result += ("i" if separator == "!" else "") + right
        else:
            result += separator + right
    return result


def remove_inserted_chars(text: str) -> str:
    """Bỏ ký tự chèn giữa chữ khi một bên là phụ âm đơn lẻ: "đ.ịt" -> "địt", "n-g-u" -> "ngu".

    Giữ nguyên câu thiếu khoảng trắng sau dấu câu ("ừ.thế", "hay!quá") vì hai bên đều là
    âm tiết thật, và gạch ngang đứng một mình ("K-pop") vì thường là tên riêng.
    """
    return INSERTED_RUN.sub(_merge_inserted, text)


def reduce_elongation(text: str) -> str:
    """Rút chữ cái lặp từ 3 lần trở lên về 1: "điiiiên" -> "điên".

    Lặp 2 lần được giữ vì có từ thật như "xoong", "boong" (D-021). Token chỉ gồm một chữ
    lặp lại ("kkkk", "hhhh") là tiếng cười, giữ nguyên: rút về "k" sẽ thành "không".
    """

    def fix_token(match: re.Match[str]) -> str:
        token = match.group()
        if len({char.lower() for char in token if char.isalpha()}) == 1:
            return token
        return ELONGATED.sub(r"\1", token)

    return TOKEN.sub(fix_token, text)


def collapse_whitespace(text: str) -> str:
    """Gộp khoảng trắng liên tiếp, bỏ khoảng trắng đầu/cuối."""
    return WHITESPACE.sub(" ", text).strip()


def _normalize_segment(text: str) -> str:
    text = replace_digit_letters(text)
    text = remove_inserted_chars(text)
    return reduce_elongation(text)


def apply_outside_urls(text: str, rule: Callable[[str], str]) -> str:
    """Áp `rule` lên phần văn bản nằm ngoài đường link; link giữ nguyên."""
    segments = URL_PATTERN.split(text)
    # re.split có nhóm bắt: phần tử ở vị trí lẻ là link, giữ nguyên.
    return "".join(segment if i % 2 else rule(segment) for i, segment in enumerate(segments))


def normalize(text: str) -> str:
    """Chuẩn hóa một bình luận theo thứ tự cố định; đường link được giữ nguyên.

    Thứ tự: NFC -> số thay chữ -> ký tự chèn -> ký tự kéo dài -> khoảng trắng.
    """
    text = unicodedata.normalize("NFC", text)
    return collapse_whitespace(apply_outside_urls(text, _normalize_segment))


RULES = {
    "replace_digit_letters": replace_digit_letters,
    "remove_inserted_chars": remove_inserted_chars,
    "reduce_elongation": reduce_elongation,
}


def main() -> None:
    """Đo trên TRAIN: mỗi luật đổi bao nhiêu câu, kèm mẫu ngẫu nhiên để đọc kiểm tra."""
    import argparse

    from src.data import TEXT_COL, load_vihsd

    parser = argparse.ArgumentParser(description="Thống kê tác động của chuẩn hóa trên train.")
    parser.add_argument("--data-dir", default="data/raw/vihsd")
    parser.add_argument("--samples", type=int, default=10, help="số ví dụ in ra cho mỗi luật")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Gộp khoảng trắng sẵn để mẫu in ra dễ đọc; normalize cũng làm bước này ở cuối.
    texts = load_vihsd("train", args.data_dir)[TEXT_COL].map(
        lambda t: collapse_whitespace(unicodedata.normalize("NFC", t))
    )
    print(f"Số câu train: {len(texts)}")
    for name, rule in RULES.items():
        # Áp luật giống hệt trong normalize (bỏ qua link), để số đếm khớp hành vi thật.
        after = texts.map(lambda t, rule=rule: apply_outside_urls(t, rule))
        changed = texts[after != texts]
        print(f"\n== {name}: đổi {len(changed)} câu ({len(changed) / len(texts):.2%}) ==")
        sample = changed.sample(min(args.samples, len(changed)), random_state=args.seed)
        for i, text in sample.items():
            print(f"  {text!r}\n  -> {after[i]!r}")
    total = (texts.map(normalize) != texts).sum()
    print(f"\nTổng: normalize đổi {total} câu ({total / len(texts):.2%}), không tính khoảng trắng")


if __name__ == "__main__":
    main()
