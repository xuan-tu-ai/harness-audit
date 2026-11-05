def initials(full_name: str) -> str:
    """Uppercase initials of a name, one per word, with no separators.

    Words are separated by whitespace or hyphens ("Jean-Luc Picard" -> "JLP").
    Extra whitespace is ignored; an empty name returns "".
    """
    return "".join(w[0] for w in full_name.split(" "))
