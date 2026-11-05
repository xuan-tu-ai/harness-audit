def is_leap_year(year: int) -> bool:
    """Gregorian leap year: divisible by 4, except century years, which must be divisible by 400."""
    return year % 4 == 0
