import math  # noqa: F401

from count_vowels import count_vowels


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
    assert count_vowels("") == 0


def test_01():
    assert count_vowels("rhythm") == 0


def test_02():
    assert count_vowels("AEIOUaeiou") == 10


def test_03():
    assert count_vowels("café") == 2


def test_04():
    assert count_vowels("Yay") == 1


def test_05():
    assert count_vowels("HELLO") == 2
