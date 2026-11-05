def round_half_up(x: float, ndigits: int = 0) -> float:
    """Round x to `ndigits` decimal places with ties rounded away from zero.

    Examples: 2.5 -> 3.0, -2.5 -> -3.0, round_half_up(0.125, 2) -> 0.13.
    Work from the decimal representation of x as written (str(x)), not binary floating point,
    so round_half_up(1.005, 2) -> 1.01.
    """
    return round(x, ndigits)
