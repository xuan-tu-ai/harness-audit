import math  # noqa: F401

from truncate import truncate


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
    assert truncate("abc", 3) == "abc"


def test_01():
    assert truncate("abcdef", 5) == "ab..."


def test_02():
    assert truncate("abcdef", 2) == "ab"


def test_03():
    assert truncate("", 0) == ""


def test_04():
    assert raises(ValueError, truncate, "abc", -1)


def test_05():
    assert truncate("abcd", 3) == "..."
