def clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        raise ValueError("lo > hi")
    if x != x:
        return x
    return max(lo, min(x, hi))
