"""Text utilities tuned for legal writing.

Legal text breaks naive sentence splitters: "Bell Atlantic Corp. v. Twombly,
550 U.S. 544 (2007)" has four periods and zero sentence boundaries. It also
writes numbers twice ("thirty (30) days") and in words ("two years").
"""
from __future__ import annotations

import re

# Abbreviations that end with a period but don't end a sentence.
ABBREVIATIONS = {
    "v", "vs", "u.s", "inc", "corp", "co", "ltd", "llc", "no", "nos", "sec", "secs",
    "art", "cl", "ch", "para", "et", "al", "id", "cf", "e.g", "i.e", "etc", "jr", "sr",
    "mr", "mrs", "ms", "dr", "st", "a.m", "p.m", "supp", "app", "cir", "ct", "f", "s",
    "l.ed", "ed", "2d", "3d", "4th", "fed", "reg", "stat", "mo", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "by", "with", "at", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "their", "his", "her", "any", "each", "all", "such", "which", "who", "whom",
    "than", "then", "there", "per", "into", "upon", "after", "before", "within", "under",
    "has", "have", "had", "will", "would", "if", "when", "where", "so", "up", "also",
    # modal and negation words are handled by dedicated checks, not by word overlap
    "must", "may", "shall", "can", "could", "should", "might", "not", "no", "never",
    "without", "do", "does", "did", "only", "solely", "exclusively",
    "fail", "fails", "failed", "failure", "neither", "nor", "none",
    # paraphrase filler: "the period lasts 12 months" vs "for twelve months"
    "period", "lasts", "last", "duration",
}

NEGATIONS = {"not", "no", "never", "without", "neither", "nor", "none", "cannot", "n't",
             "fail", "fails", "failed", "failure"}
OBLIGATION = {"must", "shall", "required", "requires", "require", "mandatory", "obligated"}
PERMISSION = {"may", "can", "optional", "optionally", "permitted", "allowed"}
EXCLUSIVE = {"only", "solely", "exclusively", "sole"}

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*")


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by whitespace and a capital letter or digit,
    unless the word before the period is a known abbreviation or an initial."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    out, start = [], 0
    for m in re.finditer(r"[.!?](?=\s+[A-Z0-9\"(\[])", text):
        word = re.search(r"([A-Za-z0-9.]+)$", text[start:m.start()])
        prev = word.group(1).lower().rstrip(".") if word else ""
        if prev in ABBREVIATIONS or len(prev) == 1 or re.fullmatch(r"\d+(\.\d+)*", prev):
            continue
        out.append(text[start:m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("\u2019", "'"))


def content_words(text: str) -> list[str]:
    return [t for t in tokens(text) if not t[0].isdigit() and t not in STOPWORDS and len(t) > 1]


def same_word(a: str, b: str) -> bool:
    """Cheap stemming: words match if they share a 4+ letter prefix and differ
    by at most 3 trailing letters (terminate / termination, accrue / accrues)."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k >= 4 and max(len(a), len(b)) - k <= 3


def quantities(text: str) -> set[tuple[float, str]]:
    """Numbers paired with what they count: 'one (1) year' -> {(1, 'year')},
    '$500' -> {(500, '$')}, '1.5%' -> {(1.5, '%')}. Lets the checker tell
    'two years' apart from 'two adults' in the same paragraph."""
    t = text.lower()
    t = re.sub(r"(?:^|(?<=[.!?]\s))\d+(?:\.\d+)+\s+(?=[a-z])", " ", t)
    t = re.sub(r"\((\d[\d,.]*)\)", " ", t)          # "thirty (30)" -> "thirty"
    t = re.sub(r"\$\s?", " $", t).replace("%", " % ")
    toks = re.findall(r"\$|%|\d+(?:,\d{3})*(?:\.\d+)?|[a-z]+(?:-[a-z]+)?", t)
    out: set[tuple[float, str]] = set()
    for i, tok in enumerate(toks):
        value = None
        if tok[0].isdigit():
            value = float(tok.replace(",", ""))
        elif all(p in WORD_NUMBERS for p in tok.split("-")):
            value = float(sum(WORD_NUMBERS[p] for p in tok.split("-")))
        if value is None:
            continue
        nxt_tok = toks[i + 1] if i + 1 < len(toks) else ""
        scale = {"thousand": 1e3, "million": 1e6, "billion": 1e9}.get(nxt_tok)
        if scale:
            value *= scale
        unit = ""
        if i > 0 and toks[i - 1] == "$":
            unit = "$"
        else:
            for nxt in toks[i + 1 + bool(scale):i + 3 + bool(scale)]:
                if nxt == "%":
                    unit = "%"
                    break
                if nxt[0].isalpha() and nxt not in STOPWORDS:
                    unit = nxt
                    break
        out.add((value, unit))
    return out


def negation_count(toks: list[str]) -> int:
    return sum(1 for t in toks if t in NEGATIONS or t.endswith("n't"))


def negated_words(toks: list[str], window: int = 2) -> set[str]:
    """Content words that appear shortly after a negation ('does not roll over')."""
    out = set()
    for i, t in enumerate(toks):
        if t in NEGATIONS or t.endswith("n't"):
            out.update(w for w in toks[i + 1:i + 1 + window]
                       if w not in STOPWORDS and not w[0].isdigit())
    return out


def modality(toks: list[str]) -> str | None:
    """'obligation', 'permission', or None. 'No person may' and 'may not' are
    prohibitions, which count as obligations."""
    for i, t in enumerate(toks):
        if t in OBLIGATION:
            return "obligation"
        if t in PERMISSION:
            before = toks[max(0, i - 3):i]
            after = toks[i + 1:i + 2]
            if any(b in NEGATIONS for b in before) or (after and after[0] in NEGATIONS):
                return "obligation"
            return "permission"
    return None
