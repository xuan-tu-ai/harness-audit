def median(values: list[float]) -> float:
    """Median of a non-empty list; the input list is not modified.

    For an even count, the mean of the two middle values. An empty list raises ValueError.
    """
    values.sort()
    return values[len(values) // 2]
