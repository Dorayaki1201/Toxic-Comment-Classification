"""Sinh nhiễu kiểu lách kiểm duyệt cho bộ test nhiễu và tăng cường dữ liệu (D-023).

Nhiễu mô phỏng cách người dùng thật viết, KHÔNG đảo ngược các luật của `normalize.py`
và không dùng `resources/teencode.tsv` (D-022): nếu nhiễu chỉ sinh đúng dạng mà chuẩn hóa
biết gỡ, điểm của cấu hình B trên test nhiễu sẽ là đánh giá vòng tròn. Vì vậy có cả những
dạng mà lớp luật cố ý bỏ qua: kéo dài 2 lần ("điien"), số ở cuối từ ("ng0"), một dấu "-".

Mỗi từ bị chọn với xác suất `rate` và chịu đúng MỘT biến đổi (chọn ngẫu nhiên trong các
biến đổi áp được), nên `rate` luôn là tỉ lệ từ bị đổi, dù chạy một hay nhiều biến đổi.
Module không đọc dữ liệu; bên gọi quyết định đưa tập nào vào (chỉ train cho cấu hình D).
"""

import csv
import difflib
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path

from src.normalize import LETTER, apply_outside_urls, has_vowel

WORD = re.compile(rf"{LETTER}+")

# Bảng viết tắt rút từ ViLexNorm train (D-024), độc lập với resources/teencode.tsv.
DEFAULT_ABBREV_PATH = Path(__file__).resolve().parents[1] / "resources" / "noise_abbrev.tsv"
# {từ chuẩn: {dạng viết tắt: số lần gặp}}, tất cả viết thường.
Abbreviations = dict[str, dict[str, int]]

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


def abbreviate_word(word: str, rng: random.Random, abbreviations: Abbreviations) -> str:
    """Thay cả từ bằng một dạng viết tắt trong bảng, chọn theo tần suất: "không" -> "k"/"ko"."""
    variants = abbreviations.get(word.lower())
    if not variants:
        return word
    variant = rng.choices(list(variants), weights=list(variants.values()))[0]
    return _match_case(word, variant)


TRANSFORMS: dict[str, Callable[[str, random.Random], str]] = {
    "teencode_chars": teencode_word,
    "remove_diacritics": remove_diacritics_word,
    "elongate": elongate_word,
    "substitute_digits": digit_word,
    "insert_chars": insert_word,
}
# "abbreviate" cần bảng nên không nằm trong TRANSFORMS; perturb gắn bảng vào khi được truyền.
TRANSFORM_NAMES = ["abbreviate", *TRANSFORMS]


def _change_one(
    word: str, names: list[str], functions: dict[str, Callable], rng: random.Random
) -> str:
    """Thử các biến đổi theo thứ tự ngẫu nhiên, dùng biến đổi đầu tiên làm đổi được từ."""
    for name in rng.sample(names, len(names)):
        new = functions[name](word, rng)
        if new != word:
            return new
    return word


def perturb(
    text: str,
    transforms: Sequence[str],
    rate: float,
    rng: random.Random,
    abbreviations: Abbreviations | None = None,
) -> str:
    """Mỗi từ (ngoài link) bị chọn với xác suất `rate` và chịu đúng một biến đổi trong `transforms`.

    Luôn rút một số ngẫu nhiên cho mỗi từ, kể cả từ không đổi được, để nhiễu chỉ phụ thuộc
    vào câu và seed. Thứ tự các tên trong `transforms` không ảnh hưởng kết quả.
    "abbreviate" cần truyền `abbreviations` (từ `load_abbreviations`).
    """
    unknown = set(transforms) - set(TRANSFORM_NAMES)
    if unknown:
        raise ValueError(f"Biến đổi không tồn tại: {sorted(unknown)}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate phải nằm trong [0, 1], nhận được {rate}")
    if "abbreviate" in transforms and not abbreviations:
        raise ValueError("Biến đổi 'abbreviate' cần bảng viết tắt (load_abbreviations)")
    functions = dict(TRANSFORMS)
    if abbreviations:
        functions["abbreviate"] = partial(abbreviate_word, abbreviations=abbreviations)
    names = [name for name in TRANSFORM_NAMES if name in transforms]

    def fix_word(match: re.Match[str]) -> str:
        word = match.group()
        return _change_one(word, names, functions, rng) if rng.random() < rate else word

    text = unicodedata.normalize("NFC", text)
    return apply_outside_urls(text, lambda segment: WORD.sub(fix_word, segment))


def perturb_texts(
    texts: Sequence[str],
    transforms: Sequence[str],
    rate: float,
    seed: int,
    abbreviations: Abbreviations | None = None,
) -> list[str]:
    """Sinh nhiễu cho cả danh sách; câu thứ i dùng `rng_for(seed, i)`."""
    return [
        perturb(text, transforms, rate, rng_for(seed, i), abbreviations)
        for i, text in enumerate(texts)
    ]


# --- Bảng viết tắt từ ViLexNorm (D-024) ---

UNLEET = {digit: letter for letter, digit in LEET.items()}


def _canonical(word: str) -> str:
    """Dạng so sánh: viết thường, số về chữ, bỏ dấu, gộp chữ lặp liền nhau."""
    word = "".join(UNLEET.get(char, char) for char in word.lower())
    return re.sub(r"(.)\1+", r"\1", strip_diacritics(word))


def explained_by_char_transforms(noisy: str, standard: str) -> bool:
    """True nếu `noisy` có thể sinh từ `standard` bằng 5 biến đổi theo ký tự.

    Ví dụ "nóiii" (kéo dài), "khong" (bỏ dấu), "ng0" (số), "fải" (teencode). Các cặp này
    không vào bảng viết tắt để mỗi loại nhiễu chỉ nằm ở một biến đổi.
    """
    target = _canonical(noisy)
    if target == _canonical(standard):
        return True
    lower = standard.lower()
    return any(
        _canonical(pattern.sub(repl, lower, count=1)) == target
        for pattern, repl in TEENCODE_RULES
        if pattern.search(lower)
    )


EDGE_PUNCTUATION = re.compile(r"^\W+|\W+$")


def _split_tokens(text: str) -> list[str]:
    """Tách theo dấu cách, bỏ dấu câu ở hai đầu ("k?" -> "k"); "=))" thành rỗng và bị bỏ.

    Không tách theo chuỗi chữ cái: "ng0" sẽ thành "ng" và cặp ("ng", "ngo") lọt vào bảng
    như một dạng viết tắt, dù thật ra là số thay chữ.
    """
    tokens = (EDGE_PUNCTUATION.sub("", t) for t in unicodedata.normalize("NFC", text).split())
    return [t.lower() for t in tokens if t]


def extract_word_pairs(original: str, normalized: str) -> list[tuple[str, str]]:
    """Gióng hàng hai câu theo từ, trả các chỗ thay 1 từ <-> 1 từ: [("ko", "không"), ...].

    Chỉ giữ cặp mà cả hai bên đều chỉ gồm chữ cái ("đ.ịt", "ng0" bị bỏ). Chỗ thay 1-n hoặc
    n-1 bị bỏ vì gióng hàng dễ sai và ViLexNorm không gán viết tắt cụm từ.
    """
    noisy, clean = _split_tokens(original), _split_tokens(normalized)
    matcher = difflib.SequenceMatcher(a=noisy, b=clean, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace" and i2 - i1 == j2 - j1:
            # Cùng số từ hai bên: gióng từng cặp theo vị trí ("t k" <-> "tôi không").
            pairs += [
                (a, b)
                for a, b in zip(noisy[i1:i2], clean[j1:j2], strict=True)
                if WORD.fullmatch(a) and WORD.fullmatch(b)
            ]
    return pairs


ELONGATED_RUN = re.compile(r"(.)\1{2,}")


def count_abbreviations(pairs: Iterable[tuple[str, str]]) -> Counter[tuple[str, str]]:
    """Đếm (từ chuẩn, dạng viết tắt) qua mọi cặp câu, bỏ cặp 5 biến đổi kia đã sinh được.

    Chữ lặp từ 3 lần được rút trước khi đếm ("lunnn" cộng vào "lun"): kéo dài là việc của
    `elongate`, để trong bảng thì một từ sẽ chịu hai biến đổi cùng lúc.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for original, normalized in pairs:
        for noisy, standard in extract_word_pairs(original, normalized):
            noisy = ELONGATED_RUN.sub(r"\1", noisy)
            if not explained_by_char_transforms(noisy, standard):
                counts[(standard, noisy)] += 1
    return counts


def build_abbreviations(counts: Counter[tuple[str, str]], min_count: int) -> Abbreviations:
    """Giữ cặp gặp ít nhất `min_count` lần; dạng viết tắt xếp theo tần suất giảm dần."""
    table: Abbreviations = {}
    for (standard, noisy), count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count >= min_count:
            table.setdefault(standard, {})[noisy] = count
    return table


def save_abbreviations(table: Abbreviations, path: str | Path) -> None:
    """Ghi bảng ra TSV: normalized, variant, count."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["normalized", "variant", "count"])
        for standard, variants in table.items():
            for noisy, count in variants.items():
                writer.writerow([standard, noisy, count])


def load_abbreviations(path: str | Path = DEFAULT_ABBREV_PATH) -> Abbreviations:
    """Đọc bảng viết tắt; báo lỗi nếu dòng trùng hoặc count không dương."""
    table: Abbreviations = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            standard, noisy, count = row["normalized"], row["variant"], int(row["count"])
            if count <= 0 or noisy in table.get(standard, {}):
                raise ValueError(f"Dòng sai trong bảng viết tắt: {row}")
            table.setdefault(standard, {})[noisy] = count
    return table


def _comparable(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def _report_build(pairs: list[tuple[str, str]], args) -> None:
    """Xây bảng từ ViLexNorm TRAIN, in thống kê để ghi vào D-024, rồi ghi file."""
    from src.data import TEXT_COL, load_vihsd
    from src.normalize import load_lexicon

    counts = count_abbreviations(pairs)
    print(f"Cặp câu ViLexNorm train: {len(pairs)}; cặp từ (sau lọc): {sum(counts.values())}")
    for min_count in (2, 3, 5):
        table = build_abbreviations(counts, min_count)
        entries = sum(len(v) for v in table.values())
        print(f"  min_count={min_count}: {len(table)} từ chuẩn, {entries} mục")

    table = build_abbreviations(counts, args.min_count)
    entries = [(s, n, c) for s, variants in table.items() for n, c in variants.items()]
    top = {(s, n) for s, n, _ in sorted(entries, key=lambda e: -e[2])[:50]}
    # Độ bão hòa: xây lại trên một phần train (xáo theo seed), xem bảng còn tăng không.
    shuffled = random.Random(args.seed).sample(pairs, len(pairs))
    print(f"\nĐộ bão hòa (min_count={args.min_count}, seed {args.seed}):")
    for fraction in (0.25, 0.5, 0.75, 1.0):
        part = count_abbreviations(shuffled[: int(fraction * len(pairs))])
        size = sum(count >= args.min_count for count in part.values())
        # Mục phổ biến của bảng đầy đủ đã xuất hiện ít nhất một lần trong phần này chưa.
        print(f"  {fraction:.0%}: {size} mục; có {len(top & set(part))}/50 mục phổ biến nhất")

    lexicon = load_lexicon()
    total = sum(c for _, _, c in entries)
    same_token = [(s, n, c) for s, n, c in entries if n in lexicon]
    same_mapping = [(s, n, c) for s, n, c in same_token if lexicon[n].lower() == s]
    print("\nTrùng với resources/teencode.tsv (keep):")
    for name, rows in (("cùng dạng viết tắt", same_token), ("cùng cả cặp", same_mapping)):
        mass = sum(c for _, _, c in rows) / total
        print(f"  {name}: {len(rows)}/{len(entries)} mục, {mass:.2%} theo tần suất")

    vilexnorm = {_comparable(original) for original, _ in pairs}
    print("\nCâu ViLexNorm train trùng nguyên văn với ViHSD (chỉ so văn bản, không đọc nhãn):")
    for split in ("train", "dev", "test"):
        texts = load_vihsd(split, args.data_dir)[TEXT_COL].map(_comparable)
        print(f"  {split}: {texts.isin(vilexnorm).sum()}/{len(texts)} câu")

    print("\n50 mục phổ biến nhất [từ chuẩn -> viết tắt (số lần)]:")
    for s, n, c in sorted(entries, key=lambda e: -e[2])[:50]:
        print(f"  {s} -> {n} ({c})")
    save_abbreviations(table, args.out)
    print(f"\nĐã ghi {len(entries)} mục vào {args.out}")


def main() -> None:
    """Xem thử nhiễu trên mẫu TRAIN, hoặc xây bảng viết tắt từ ViLexNorm train (--build-abbrev)."""
    import argparse

    import pandas as pd

    from src.data import TEXT_COL, load_vihsd
    from src.utils import load_config, set_seed

    parser = argparse.ArgumentParser(description="Xem thử nhiễu / xây bảng viết tắt.")
    parser.add_argument("--config", default="configs/noise.yaml")
    parser.add_argument("--data-dir", default="data/raw/vihsd")
    parser.add_argument("--sentences", type=int, default=2000, help="số câu train lấy mẫu")
    parser.add_argument("--samples", type=int, default=5, help="số ví dụ in ra mỗi biến đổi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build-abbrev", metavar="CSV", help="ViLexNorm train.csv -> bảng")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--out", default=DEFAULT_ABBREV_PATH)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.build_abbrev:
        df = pd.read_csv(args.build_abbrev, index_col=0, keep_default_na=False)
        _report_build(list(zip(df["original"], df["normalized"], strict=True)), args)
        return

    config = load_config(args.config)
    levels, transforms = config["levels"], config["transforms"]
    abbreviations = (
        load_abbreviations(config["abbreviations"]) if "abbreviations" in config else None
    )
    train = load_vihsd("train", args.data_dir)
    texts = train[TEXT_COL].sample(min(args.sentences, len(train)), random_state=args.seed)
    texts = [unicodedata.normalize("NFC", t) for t in texts]
    total_words = sum(len(WORD.findall(t)) for t in texts)

    runs = [([name], "medium") for name in transforms]
    runs += [(transforms, level) for level in levels]
    for names, level in runs:
        noisy = perturb_texts(texts, names, levels[level], args.seed, abbreviations)
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
