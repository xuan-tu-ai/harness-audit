def is_palindrome(text: str) -> bool:
    s = [c.lower() for c in text if c.isalnum()]
    return s == s[::-1]
