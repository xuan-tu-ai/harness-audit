def reverse_words(text: str) -> str:
    """Reverse the order of the words in text.

    Words are separated by any whitespace; the result joins them with single spaces
    and has no leading or trailing whitespace.
    """
    return " ".join(reversed(text.split(" ")))
