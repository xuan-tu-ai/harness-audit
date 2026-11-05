import math  # noqa: F401

from sum_digits import sum_digits


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
    assert sum_digits(0) == 0


def test_01():
    assert sum_digits(-7) == 7


def test_02():
    assert sum_digits(99999) == 45


def test_03():
    assert sum_digits(10**20) == 1


def test_04():
    assert sum_digits(-100) == 1


def test_05():
    assert sum_digits(-45) == 9
