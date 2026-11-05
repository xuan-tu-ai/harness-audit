import math  # noqa: F401

from round_half_up import round_half_up


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
    assert round_half_up(-2.5) == -3.0


def test_01():
    assert round_half_up(0.125, 2) == 0.13


def test_02():
    assert round_half_up(1.005, 2) == 1.01


def test_03():
    assert round_half_up(3.14159, 3) == 3.142


def test_04():
    assert round_half_up(0.5) == 1.0


def test_05():
    assert round_half_up(2.675, 2) == 2.68
