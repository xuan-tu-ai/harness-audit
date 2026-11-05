import math  # noqa: F401

from clamp import clamp


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
    assert clamp(11, 0, 10) == 10


def test_01():
    assert clamp(-1, 0, 10) == 0


def test_02():
    assert clamp(0, 0, 0) == 0


def test_03():
    assert math.isnan(clamp(float("nan"), 0, 1))


def test_04():
    assert raises(ValueError, clamp, 1, 5, 0)


def test_05():
    assert clamp(-5.5, -10, -1) == -5.5
