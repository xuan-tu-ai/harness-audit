import re


def initials(full_name: str) -> str:
    return "".join(w[0].upper() for w in re.split(r"[\s-]+", full_name) if w)
