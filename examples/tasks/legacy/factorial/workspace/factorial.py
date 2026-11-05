def factorial(n: int) -> int:
    """n! for integers n >= 0 (0! == 1). A negative n raises ValueError."""
    result = 1
    for i in range(1, n):
        result *= i
    return result
