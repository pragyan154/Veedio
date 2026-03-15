from typing import Iterable


_INDEPENDENT_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}

_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॅ": "e", "ॉ": "o",
}

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ळ": "l",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
}

_SIGNS = {
    "ं": "n", "ँ": "n", "ः": "h", "ऽ": "'",
    "।": ".", "॥": ".",
}

_DIGITS = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

_HALANT = "्"


def devanagari_to_roman(text: str) -> str:
    if not text:
        return text

    out = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch in _INDEPENDENT_VOWELS:
            out.append(_INDEPENDENT_VOWELS[ch])
            i += 1
            continue

        if ch in _CONSONANTS:
            base = _CONSONANTS[ch]
            next_ch = text[i + 1] if i + 1 < n else ""

            if next_ch == _HALANT:
                out.append(base)
                i += 2
                continue

            if next_ch in _MATRAS:
                out.append(base + _MATRAS[next_ch])
                i += 2
                continue

            out.append(base + "a")
            i += 1
            continue

        if ch in _MATRAS:
            out.append(_MATRAS[ch])
            i += 1
            continue

        if ch in _SIGNS:
            out.append(_SIGNS[ch])
            i += 1
            continue

        if ch in _DIGITS:
            out.append(_DIGITS[ch])
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def ensure_hinglish_roman(text: str) -> str:
    return devanagari_to_roman(text or "")


def ensure_hinglish_roman_headings(headings):
    if isinstance(headings, (list, tuple)):
        return [ensure_hinglish_roman(str(h)) for h in headings]
    return ensure_hinglish_roman(str(headings or ""))
