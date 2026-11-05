import math  # noqa: F401

from slugify import slugify


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
    assert slugify("Hello World") == "hello-world"


def test_01():
    assert slugify("Hello, World!") == "hello-world"


def test_02():
    assert slugify("Crème Brûlée") == "creme-brulee"


def test_03():
    assert slugify("") == ""


def test_04():
    assert slugify("---") == ""


def test_05():
    assert slugify("a__b") == "a-b"


def test_06():
    assert slugify("ABC 123") == "abc-123"
