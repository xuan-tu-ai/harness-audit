import math  # noqa: F401

from is_palindrome import is_palindrome


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def not_mutated(fn, value):
    copy = list(value)
    fn(value)
    return value == copy


def test_00():
    assert is_palindrome("A man, a plan, a canal: Panama") is True


def test_01():
    assert is_palindrome("") is True


def test_02():
    assert is_palindrome("ab") is False


def test_03():
    assert is_palindrome("No 'x' in Nixon") is True


def test_04():
    assert is_palindrome("123 21!") is True


def test_05():
    assert is_palindrome("hello") is False
