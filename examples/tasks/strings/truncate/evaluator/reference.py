def truncate(text: str, width: int) -> str:
    if width < 0:
        raise ValueError("width must be >= 0")
    if len(text) <= width:
        return text
    if width < 3:
        return text[:width]
    return text[: width - 3] + "..."
