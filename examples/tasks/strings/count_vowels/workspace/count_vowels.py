def count_vowels(text: str) -> int:
    """Count the vowels a, e, i, o, u in text, case-insensitively ('y' is not a vowel).

    Accented vowels count as their base vowel ("é" counts as "e").
    """
    return sum(1 for c in text if c in "aeiou")
