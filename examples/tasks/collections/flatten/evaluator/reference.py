def flatten(nested: list) -> list:
    out = []
    for x in nested:
        if isinstance(x, (list, tuple)):
            out.extend(flatten(list(x)))
        else:
            out.append(x)
    return out
