import math  # noqa: F401

from wrap_text import wrap_text


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
    assert wrap_text("one  two", 10) == ["one two"]


def test_01():
    assert wrap_text("a\nb", 5) == ["a b"]


def test_02():
    assert wrap_text("supercalifragilistic is long", 5) == ["supercalifragilistic", "is", "long"]


def test_03():
    assert wrap_text("   ", 4) == []


def test_04():
    assert wrap_text("the quick brown fox", 10) == ["the quick", "brown fox"]


def test_05():
    assert wrap_text("a b c", 3) == ["a b", "c"]
