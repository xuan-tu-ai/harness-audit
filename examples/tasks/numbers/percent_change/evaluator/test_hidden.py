import math  # noqa: F401

from percent_change import percent_change


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
    assert percent_change(-10, -5) == 50.0


def test_01():
    assert percent_change(50, 25) == -50.0


def test_02():
    assert percent_change(3, 4) == 33.33


def test_03():
    assert percent_change(0, 0) is None


def test_04():
    assert percent_change(10, 10) == 0.0


def test_05():
    assert percent_change(-4, 4) == 200.0
