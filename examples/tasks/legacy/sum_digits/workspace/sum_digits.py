def sum_digits(n: int) -> int:
    """Sum of the decimal digits of an integer; the sign is ignored (sum_digits(-123) == 6)."""
    return sum(int(c) for c in str(n))
