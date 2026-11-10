"""Text sources and the one normalizer every script shares.

WikiText-103 (raw) is detokenized and cut into sentences. Only sentences made of plain
letters and basic punctuation are kept, so that the reference words, the words Kokoro
speaks and the words a recognizer prints are the same strings (no digits, no hyphens,
no abbreviations to expand)."""
import re

_APOS = "'"
_OK = re.compile(r"^[A-Z][A-Za-z ,;:'\"]+[.?!]$")
_ABBR = re.compile(r"\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|No|Co|Inc|Ltd|Mt|Gen|Col|Lt|Sgt|Capt|Rev|Prof)\.$")


def norm(s: str) -> str:
    """Lowercase, keep letters/digits/apostrophes, collapse spaces."""
    s = s.lower().replace("’", _APOS).replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", s)  # drop quotes, keep in-word apostrophes
    return " ".join(s.split())


def detok(line: str) -> str:
    line = line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    line = re.sub(r" ([,.;:?!)])", r"\1", line)
    line = re.sub(r"\( ", "(", line)
    line = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", line)
    line = re.sub(r'" ([^"]*?) "', r'"\1"', line)
    return line.strip()


def sentences(lines, min_words=6, max_words=22):
    """Yield clean, unique sentences from raw WikiText lines."""
    seen = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("="):
            continue
        for s in re.split(r"(?<=[.?!]) (?=[A-Z])", detok(line)):
            s = s.strip()
            n = len(s.split())
            if not (min_words <= n <= max_words) or not _OK.match(s) or _ABBR.search(s):
                continue
            if s.count('"') % 2 or re.search(r"\b[A-Z]{2,}\b|\b[A-Z]\b(?<!\bI)(?<!\bA)", s):
                continue  # unbalanced quotes, acronyms, stray initials
            if s in seen:
                continue
            seen.add(s)
            yield s


def wikitext_sentences(split: str, **kw):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    return sentences(ds["text"], **kw)
