def percent_change(old: float, new: float) -> float | None:
    """Percentage change from old to new, rounded to 2 decimals.

    Uses abs(old) as the base, so a move from -10 to -5 is +50.0.
    Returns None when old is 0 (the change is undefined).
    """
    return round((new - old) / old * 100, 2)
