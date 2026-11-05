def percent_change(old: float, new: float) -> float | None:
    if old == 0:
        return None
    return round((new - old) / abs(old) * 100, 2)
