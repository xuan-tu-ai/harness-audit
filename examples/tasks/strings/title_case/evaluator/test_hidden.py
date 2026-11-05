import math  # noqa: F401

from title_case import title_case


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
    assert title_case("a tale of two cities") == "A Tale of Two Cities"


def test_01():
    assert title_case("") == ""


def test_02():
    assert title_case("of") == "Of"


def test_03():
    assert title_case("what it is for") == "What It Is For"


def test_04():
    assert title_case("THE END") == "The End"


def test_05():
    assert title_case("the lord of the rings") == "The Lord of the Rings"
