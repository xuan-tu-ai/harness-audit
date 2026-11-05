import math  # noqa: F401

from median import median


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
    assert median([5]) == 5


def test_01():
    assert raises(ValueError, median, [])


def test_02():
    assert median([1, 3]) == 2.0


def test_03():
    assert not_mutated(median, [3, 1, 2])


def test_04():
    assert median([-5, -1, -3]) == -3


def test_05():
    assert median([1.5, 2.5, 3.5, 4.5]) == 3.0
