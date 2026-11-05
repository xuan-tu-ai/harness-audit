def clamp(x: float, lo: float, hi: float) -> float:
    """Limit x to the closed interval [lo, hi].

    Raises ValueError if lo > hi. A NaN x is returned unchanged.
    """
    return max(lo, min(x, hi))
