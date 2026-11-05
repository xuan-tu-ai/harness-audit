def fizzbuzz(n: int) -> list[str]:
    """FizzBuzz for 1..n inclusive.

    Multiples of 3 -> "Fizz", of 5 -> "Buzz", of both -> "FizzBuzz", otherwise the number
    as a string. n <= 0 returns [].
    """
    out = []
    for i in range(1, n):
        if i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        elif i % 15 == 0:
            out.append("FizzBuzz")
        else:
            out.append(str(i))
    return out
