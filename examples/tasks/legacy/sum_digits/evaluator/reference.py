def sum_digits(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))
