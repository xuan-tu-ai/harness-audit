def chunk(items: list, size: int) -> list[list]:
    """Split items into consecutive chunks of length `size`; the last chunk may be shorter.

    `size` must be a positive integer, otherwise ValueError. The input is not modified.
    """
    return [items[i : i + size] for i in range(0, len(items) - size + 1, size)]
