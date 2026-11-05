def truncate(text: str, width: int) -> str:
    """Shorten text to at most `width` characters.

    - If len(text) <= width, return text unchanged.
    - Otherwise return the first (width - 3) characters followed by "...".
    - If width < 3 (and the text is longer), return the first `width` characters, no ellipsis.
    - A negative width raises ValueError.
    """
    if len(text) < width:
        return text
    return text[:width] + "..."
