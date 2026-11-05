import math  # noqa: F401

from chunk import chunk


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
    assert chunk([], 3) == []


def test_01():
    assert chunk([1], 5) == [[1]]


def test_02():
    assert raises(ValueError, chunk, [1, 2], -1)


def test_03():
    assert raises(ValueError, chunk, [1, 2], 0)


def test_04():
    assert chunk(list("abcdefg"), 3) == [["a", "b", "c"], ["d", "e", "f"], ["g"]]


def test_05():
    assert chunk([1, 2, 3], 2) == [[1, 2], [3]]
