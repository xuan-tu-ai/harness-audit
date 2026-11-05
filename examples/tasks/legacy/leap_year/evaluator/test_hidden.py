import math  # noqa: F401

from leap_year import is_leap_year


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
    assert is_leap_year(2000) is True


def test_01():
    assert is_leap_year(2100) is False


def test_02():
    assert is_leap_year(2023) is False


def test_03():
    assert is_leap_year(1600) is True


def test_04():
    assert is_leap_year(4) is True


def test_05():
    assert is_leap_year(1900) is False
