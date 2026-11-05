import math  # noqa: F401

from flatten import flatten


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
    assert flatten([]) == []


def test_01():
    assert flatten(["ab", ["cd"]]) == ["ab", "cd"]


def test_02():
    assert flatten([(1, 2), [3, (4,)]]) == [1, 2, 3, 4]


def test_03():
    assert flatten([[[]]]) == []


def test_04():
    assert flatten([1, [2, [3, [4, [5]]]]]) == [1, 2, 3, 4, 5]


def test_05():
    assert flatten([None, [None]]) == [None, None]
