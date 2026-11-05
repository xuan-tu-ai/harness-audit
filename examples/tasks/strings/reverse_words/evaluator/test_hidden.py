import math  # noqa: F401

from reverse_words import reverse_words


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
    assert reverse_words("") == ""


def test_01():
    assert reverse_words("one") == "one"


def test_02():
    assert reverse_words("a\tb\nc") == "c b a"


def test_03():
    assert reverse_words("x  y") == "y x"


def test_04():
    assert reverse_words("   ") == ""


def test_05():
    assert reverse_words("  a b ") == "b a"
