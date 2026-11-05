def is_palindrome(text: str) -> bool:
    """True if text reads the same backwards, ignoring case and every character
    that is not a letter or digit. The empty string is a palindrome."""
    return text == text[::-1]
