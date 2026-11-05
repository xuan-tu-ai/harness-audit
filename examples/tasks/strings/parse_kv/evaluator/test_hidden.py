import math  # noqa: F401

from parse_kv import parse_kv


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
    assert parse_kv("a=1;;b=2") == {"a": "1", "b": "2"}


def test_01():
    assert parse_kv("url=a=b") == {"url": "a=b"}


def test_02():
    assert parse_kv("") == {}


def test_03():
    assert raises(ValueError, parse_kv, "novalue")


def test_04():
    assert parse_kv("x=;y=2") == {"x": "", "y": "2"}


def test_05():
    assert parse_kv("a=1;") == {"a": "1"}
