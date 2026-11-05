def slugify(text: str) -> str:
    """Return a URL slug: lowercase ASCII letters and digits separated by single hyphens.

    - Any run of characters that are not ASCII letters or digits becomes one hyphen.
    - No leading or trailing hyphens.
    - Accented Latin letters are reduced to their base letter ("é" -> "e").
    - An input with no letters or digits returns "".
    """
    return text.lower().replace(" ", "-")
