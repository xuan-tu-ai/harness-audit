import math  # noqa: F401

from initials import initials


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
    assert initials("Jean-Luc Picard") == "JLP"


def test_01():
    assert initials("  Alan   Turing ") == "AT"


def test_02():
    assert initials("") == ""


def test_03():
    assert initials("émile zola") == "ÉZ"


def test_04():
    assert initials("x") == "X"


def test_05():
    assert initials("grace hopper") == "GH"
