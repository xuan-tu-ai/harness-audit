import math  # noqa: F401

from fizzbuzz import fizzbuzz


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
    assert fizzbuzz(15)[-1] == "FizzBuzz"


def test_01():
    assert len(fizzbuzz(10)) == 10


def test_02():
    assert fizzbuzz(5)[4] == "Buzz"


def test_03():
    assert fizzbuzz(-3) == []


def test_04():
    assert fizzbuzz(1) == ["1"]


def test_05():
    assert fizzbuzz(3) == ["1", "2", "Fizz"]
