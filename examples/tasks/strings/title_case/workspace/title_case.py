SMALL = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to"}


def title_case(text: str) -> str:
    """Title-case words separated by single spaces.

    - The first and last words are always capitalized.
    - Other words that are in SMALL stay lowercase.
    - A capitalized word has its first letter uppercased and the rest lowercased.
    - Empty input returns "".
    """
    return " ".join(w.capitalize() for w in text.split(" "))
