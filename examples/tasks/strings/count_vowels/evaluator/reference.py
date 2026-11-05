import unicodedata


def count_vowels(text: str) -> int:
    t = unicodedata.normalize("NFKD", text).lower()
    return sum(1 for c in t if c in "aeiou")
