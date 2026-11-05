def flatten(nested: list) -> list:
    """Flatten arbitrarily nested lists and tuples into one flat list, preserving order.

    Strings are atoms (never split into characters). An empty list returns [].
    """
    out = []
    for x in nested:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out
