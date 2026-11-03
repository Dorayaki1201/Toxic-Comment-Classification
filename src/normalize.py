"""Chuẩn hóa văn bản bằng luật (cấu hình B): lớp luật (3a) và lớp từ điển teencode (3b).

Nguyên tắc: thà bỏ sót còn hơn sửa sai. Một luật sửa nhầm câu bình thường sẽ làm hỏng
cả những câu sạch, nên mỗi luật chỉ chạy ở chỗ chắc chắn là cố ý lách.
Không viết thường, không đổi emoji. Từ điển chỉ chạy khi được truyền vào `normalize`.
"""

import csv
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path

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
# Ký tự có hai nghĩa, chỉ coi là ký tự chèn khi xuất hiện từ 2 lần trong chuỗi ("n-g-u",
# "n*g*u"). Một lần thì "-" thường là tên riêng ("K-pop"), "*" là ký tự che một chữ
# ("x*u" = "xấu", bỏ đi sẽ thành "xu"); khôi phục chữ bị che là việc của từ điển (3b).
NEEDS_REPEAT_CHARS = "-*"
INSERTED_RUN = re.compile(rf"{LETTER}+(?:[{re.escape(INSERTED_CHARS)}]{LETTER}+)+")

ELONGATED = re.compile(rf"({LETTER})\1{{2,}}", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")

VOWELS = set("aeiouy")

# Từ điển teencode xây từ train, đã chấm tay (resources/README.md, D-022).
DEFAULT_LEXICON_PATH = Path(__file__).resolve().parents[1] / "resources" / "teencode.tsv"
# Token đứng ngay sau các ký tự này không được thay: mặt cười (":v", ":D", "=]]z", ":))z"),
# tên miền ("abc.vn"), tag ("@hp"), đường dẫn ("a/b").
NOT_AFTER = ":;=)]./@"
# Một chuỗi chữ cái đứng riêng: không dính chữ/số hai bên ("5h", "kong" không khớp "h", "ko").
LEXICON_TOKEN = re.compile(rf"(?<![\w{re.escape(NOT_AFTER)}]){LETTER}+(?!\w)")


def has_vowel(piece: str) -> bool:
    """Có nguyên âm không (bỏ dấu trước khi xét: "ế" -> "e")."""
    base = unicodedata.normalize("NFD", piece.lower())
    return any(char in VOWELS for char in base)


def _is_single_consonant(piece: str) -> bool:
    """Một phụ âm đứng một mình, như "đ", "n": không thể là một âm tiết tiếng Việt."""
    return len(piece) == 1 and not has_vowel(piece)


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


def _starts_with_vowel(piece: str) -> bool:
    return has_vowel(piece[:1])


def _should_merge(separator: str, separators: list[str], left: str, right: str) -> bool:
    if separator == "!" or separators.count(separator) >= 2:
        # Tách từng chữ ("n.g.u") hoặc "!" thay chữ "i" ("d!t").
        return _is_single_consonant(left) or _is_single_consonant(right)
    if separator in NEEDS_REPEAT_CHARS:
        return False
    # Một dấu "." / "_": chỉ gộp khi nó tách phụ âm đầu khỏi phần vần ("đ.iên"). Phụ âm
    # đơn trong teencode cũng là một chữ ("k" = không), nên "k.tết", "M.n" là thiếu dấu cách.
    return _is_single_consonant(left) and _starts_with_vowel(right)


def _merge_inserted(match: re.Match[str]) -> str:
    parts = re.split(rf"([{re.escape(INSERTED_CHARS)}])", match.group())
    pieces, separators = parts[0::2], parts[1::2]
    result = pieces[0]
    for i, separator in enumerate(separators):
        left, right = pieces[i], pieces[i + 1]
        if _should_merge(separator, separators, left, right):
            result += ("i" if separator == "!" else "") + right
        else:
            result += separator + right
    return result


def remove_inserted_chars(text: str) -> str:
    """Bỏ ký tự chèn giữa chữ khi một bên là phụ âm đơn lẻ: "đ.ịt" -> "địt", "n-g-u" -> "ngu".

    Giữ nguyên câu thiếu khoảng trắng sau dấu câu ("ừ.thế", "hay!quá", "k.tết") vì bên phải
    là một chữ mới; "-" và "*" chỉ xuất hiện một lần ("K-pop", "x*u") cũng giữ nguyên.
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


def load_lexicon(path: str | Path = DEFAULT_LEXICON_PATH) -> dict[str, str]:
    """Đọc các mục `keep` của từ điển teencode: {"ko": "không", ...}.

    Báo lỗi nếu file sai: mục keep thiếu dạng đầy đủ, token trùng, hoặc dạng đầy đủ chứa
    một token khác của từ điển (sẽ thay dây chuyền, normalize hết idempotent).
    """
    lexicon: dict[str, str] = {}
    seen: set[str] = set()
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            token = row["token"].lower()
            if token in seen:
                raise ValueError(f"Token trùng trong từ điển: {token!r}")
            seen.add(token)
            if row["decision"] != "keep":
                continue
            if not row["replacement"]:
                raise ValueError(f"Mục keep thiếu replacement: {token!r}")
            lexicon[token] = row["replacement"]
    for token, replacement in lexicon.items():
        chained = set(replacement.lower().split()) & set(lexicon)
        if chained:
            raise ValueError(f"{token!r} -> {replacement!r} chứa token của từ điển: {chained}")
    return lexicon


def replace_teencode(text: str, lexicon: dict[str, str]) -> str:
    """Thay token đứng riêng có trong từ điển bằng dạng đầy đủ: "Ko biết" -> "Không biết".

    Tra không phân biệt hoa/thường; token viết hoa chữ đầu thì kết quả cũng viết hoa chữ đầu.
    Bỏ qua token trong link, token dính số ("5h") và token sau ký tự trong NOT_AFTER (":v").
    """

    def fix_token(match: re.Match[str]) -> str:
        token = match.group()
        replacement = lexicon.get(token.lower())
        if replacement is None:
            return token
        if token[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    return apply_outside_urls(text, lambda segment: LEXICON_TOKEN.sub(fix_token, segment))


def normalize(text: str, lexicon: dict[str, str] | None = None) -> str:
    """Chuẩn hóa một bình luận theo thứ tự cố định; đường link được giữ nguyên.

    Thứ tự: NFC -> số thay chữ -> ký tự chèn -> ký tự kéo dài -> từ điển (nếu có) ->
    khoảng trắng. Không truyền `lexicon` thì chỉ chạy lớp luật (dùng khi tìm ứng viên).
    """
    text = unicodedata.normalize("NFC", text)
    text = apply_outside_urls(text, _normalize_segment)
    if lexicon:
        text = replace_teencode(text, lexicon)
    return collapse_whitespace(text)


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
