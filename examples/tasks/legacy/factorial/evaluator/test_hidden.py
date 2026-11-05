import math  # noqa: F401

from factorial import factorial


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
    assert factorial(1) == 1


def test_01():
    assert factorial(10) == 3628800


def test_02():
    assert raises(ValueError, factorial, -1)


def test_03():
    assert factorial(20) == 2432902008176640000


def test_04():
    assert factorial(2) == 2


def test_05():
    assert factorial(5) == 120
