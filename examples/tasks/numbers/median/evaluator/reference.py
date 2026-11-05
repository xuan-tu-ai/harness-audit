def median(values: list[float]) -> float:
    if not values:
        raise ValueError("empty")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
