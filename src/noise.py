"""Sinh nhiễu kiểu lách kiểm duyệt cho bộ test nhiễu và tăng cường dữ liệu (D-023).

Nhiễu mô phỏng cách người dùng thật viết, KHÔNG đảo ngược các luật của `normalize.py`
và không dùng `resources/teencode.tsv` (D-022): nếu nhiễu chỉ sinh đúng dạng mà chuẩn hóa
biết gỡ, điểm của cấu hình B trên test nhiễu sẽ là đánh giá vòng tròn. Vì vậy có cả những
dạng mà lớp luật cố ý bỏ qua: kéo dài 2 lần ("điien"), số ở cuối từ ("ng0"), một dấu "-".

Mỗi từ bị chọn với xác suất `rate` và chịu đúng MỘT biến đổi (chọn ngẫu nhiên trong các
biến đổi áp được), nên `rate` luôn là tỉ lệ từ bị đổi, dù chạy một hay nhiều biến đổi.
Module không đọc dữ liệu; bên gọi quyết định đưa tập nào vào (chỉ train cho cấu hình D).
"""

import random
import re
import unicodedata
from collections.abc import Callable, Sequence

from src.normalize import LETTER, apply_outside_urls, has_vowel

WORD = re.compile(rf"{LETTER}+")

# Teencode theo ký tự: (mẫu trên chữ thường, thay bằng). Mỗi từ chỉ chịu một luật, chọn
# ngẫu nhiên trong các luật khớp. Không dùng bảng từ nào, nên độc lập với từ điển 3b.
TEENCODE_RULES = [
    (re.compile(r"^ph"), "f"),  # "phở" -> "fở"
    (re.compile(r"^qu"), "w"),  # "quá" -> "wá"
    (re.compile(r"^gi"), "j"),  # "giờ" -> "jờ"
    (re.compile(r"^gì$"), "j"),  # "gì" -> "j"
    (re.compile(r"^c(?!h)"), "k"),  # "con" -> "kon"; "ch" giữ nguyên
    (re.compile(r"^v"), "z"),  # "vậy" -> "zậy"
    (re.compile(r"(?<=.)i$"), "j"),  # "đi" -> "đj"; chỉ "i" không dấu ở cuối
]

# Chữ (sau khi bỏ dấu) -> số/ký hiệu trông giống. Có thể thay ở bất kỳ vị trí nào trong từ.
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "$"}

INSERT_CHARS = ".-_*"
# Xác suất chèn vào giữa MỌI chữ ("n.g.u"); còn lại chèn đúng một chỗ ("ng.u").
INSERT_EVERY_GAP_PROB = 0.5
ELONGATE_MIN, ELONGATE_MAX = 2, 5  # tổng số lần chữ xuất hiện sau khi kéo dài


def rng_for(seed: int, key: int | str) -> random.Random:
    """Bộ sinh ngẫu nhiên riêng cho một câu, xác định bởi (seed, key).

    Seed dạng chuỗi được Python băm bằng SHA-512, nên ổn định giữa các lần chạy và không
    phụ thuộc PYTHONHASHSEED. Nhiễu của câu này không phụ thuộc các câu khác.
    """
    return random.Random(f"{seed}-{key}")


def _match_case(original: str, new: str) -> str:
    """Giữ kiểu chữ của từ gốc: "PHỞ" -> "FỞ", "Phở" -> "Fở"."""
    if len(original) > 1 and original.isupper():
        return new.upper()
    if original[0].isupper():
        return new[:1].upper() + new[1:]
    return new


def _base_letter(char: str) -> str:
    """Chữ gốc không dấu, viết thường: "ế" -> "e", "ơ" -> "o" ("đ" giữ nguyên)."""
    return unicodedata.normalize("NFD", char)[0].lower()


def strip_diacritics(word: str) -> str:
    """Bỏ toàn bộ dấu của một chuỗi: "Đường" -> "Duong"."""
    decomposed = unicodedata.normalize("NFD", word.replace("đ", "d").replace("Đ", "D"))
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


# Mỗi biến đổi nhận một từ (chuỗi chữ cái) và trả lại chính từ đó nếu không áp được.


def teencode_word(word: str, rng: random.Random) -> str:
    """Thay ký tự kiểu teencode: "phở" -> "fở", "quá" -> "wá", "đi" -> "đj"."""
    lower = word.lower()
    matching = [(pattern, repl) for pattern, repl in TEENCODE_RULES if pattern.search(lower)]
    if not matching:
        return word
    pattern, repl = rng.choice(matching)
    return _match_case(word, pattern.sub(repl, lower, count=1))


def remove_diacritics_word(word: str, rng: random.Random) -> str:
    """Bỏ dấu cả từ: "thằng" -> "thang". Áp từng từ nên câu có thể bỏ dấu một phần."""
    return strip_diacritics(word)


def elongate_word(word: str, rng: random.Random) -> str:
    """Kéo dài một nguyên âm thành 2-5 lần: "điên" -> "điiiên" hoặc "điien"."""
    vowels = [i for i, char in enumerate(word) if has_vowel(char)]
    if not vowels:
        return word
    i = rng.choice(vowels)
    return word[:i] + word[i] * rng.randint(ELONGATE_MIN, ELONGATE_MAX) + word[i + 1 :]


def digit_word(word: str, rng: random.Random) -> str:
    """Thay một chữ bằng số trông giống, ở bất kỳ vị trí nào: "chết" -> "ch3t", "ngo" -> "ng0"."""
    positions = [i for i, char in enumerate(word) if _base_letter(char) in LEET]
    if not positions:
        return word
    i = rng.choice(positions)
    return word[:i] + LEET[_base_letter(word[i])] + word[i + 1 :]


def insert_word(word: str, rng: random.Random) -> str:
    """Chèn một loại ký tự `. - _ *` vào mọi khe ("n.g.u") hoặc một khe ("ng-u")."""
    if len(word) < 2:
        return word
    separator = rng.choice(INSERT_CHARS)
    if rng.random() < INSERT_EVERY_GAP_PROB:
        return separator.join(word)
    i = rng.randint(1, len(word) - 1)
    return word[:i] + separator + word[i:]


TRANSFORMS: dict[str, Callable[[str, random.Random], str]] = {
    "teencode_chars": teencode_word,
    "remove_diacritics": remove_diacritics_word,
    "elongate": elongate_word,
    "substitute_digits": digit_word,
    "insert_chars": insert_word,
}


def _change_one(word: str, names: list[str], rng: random.Random) -> str:
    """Thử các biến đổi theo thứ tự ngẫu nhiên, dùng biến đổi đầu tiên làm đổi được từ."""
    for name in rng.sample(names, len(names)):
        new = TRANSFORMS[name](word, rng)
        if new != word:
            return new
    return word


def perturb(text: str, transforms: Sequence[str], rate: float, rng: random.Random) -> str:
    """Mỗi từ (ngoài link) bị chọn với xác suất `rate` và chịu đúng một biến đổi trong `transforms`.

    Luôn rút một số ngẫu nhiên cho mỗi từ, kể cả từ không đổi được, để nhiễu chỉ phụ thuộc
    vào câu và seed. Thứ tự các tên trong `transforms` không ảnh hưởng kết quả.
    """
    unknown = set(transforms) - set(TRANSFORMS)
    if unknown:
        raise ValueError(f"Biến đổi không tồn tại: {sorted(unknown)}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate phải nằm trong [0, 1], nhận được {rate}")
    names = [name for name in TRANSFORMS if name in transforms]

    def fix_word(match: re.Match[str]) -> str:
        word = match.group()
        return _change_one(word, names, rng) if rng.random() < rate else word

    text = unicodedata.normalize("NFC", text)
    return apply_outside_urls(text, lambda segment: WORD.sub(fix_word, segment))


def perturb_texts(
    texts: Sequence[str], transforms: Sequence[str], rate: float, seed: int
) -> list[str]:
    """Sinh nhiễu cho cả danh sách; câu thứ i dùng `rng_for(seed, i)`."""
    return [perturb(text, transforms, rate, rng_for(seed, i)) for i, text in enumerate(texts)]


def main() -> None:
    """Xem thử nhiễu trên mẫu TRAIN: mỗi biến đổi đổi bao nhiêu câu, kèm vài ví dụ để đọc."""
    import argparse

    from src.data import TEXT_COL, load_vihsd
    from src.utils import load_config, set_seed

    parser = argparse.ArgumentParser(description="Xem thử nhiễu trên mẫu câu train.")
    parser.add_argument("--config", default="configs/noise.yaml")
    parser.add_argument("--data-dir", default="data/raw/vihsd")
    parser.add_argument("--sentences", type=int, default=2000, help="số câu train lấy mẫu")
    parser.add_argument("--samples", type=int, default=5, help="số ví dụ in ra mỗi biến đổi")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    levels, transforms = config["levels"], config["transforms"]
    train = load_vihsd("train", args.data_dir)
    texts = train[TEXT_COL].sample(min(args.sentences, len(train)), random_state=args.seed)
    texts = [unicodedata.normalize("NFC", t) for t in texts]
    total_words = sum(len(WORD.findall(t)) for t in texts)

    runs = [([name], "medium") for name in transforms]
    runs += [(transforms, level) for level in levels]
    for names, level in runs:
        noisy = perturb_texts(texts, names, levels[level], args.seed)
        changed = [(a, b) for a, b in zip(texts, noisy, strict=True) if a != b]
        # Đếm từ bị đổi = số từ của câu gốc không còn nguyên trong câu nhiễu (xấp xỉ).
        words_changed = sum(
            sum(word not in set(WORD.findall(b)) for word in WORD.findall(a)) for a, b in changed
        )
        label = names[0] if len(names) == 1 else "all"
        print(f"\n== {label} @ {level} (rate {levels[level]}): đổi {len(changed)}", end="")
        print(f"/{len(texts)} câu ({len(changed) / len(texts):.2%}),", end=" ")
        print(f"~{words_changed / total_words:.2%} từ ==")
        for before, after in changed[: args.samples]:
            print(f"  {before!r}\n  -> {after!r}")


if __name__ == "__main__":
    main()
