def wrap_text(text: str, width: int) -> list[str]:
    """Greedy word wrap.

    - Words are separated by any whitespace (spaces, tabs, newlines, repeated spaces).
    - Each line holds as many whole words as fit in `width` characters, joined by single spaces.
    - A word longer than `width` goes on its own line, unbroken.
    - Empty or whitespace-only text returns [].
    """
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) < width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines
