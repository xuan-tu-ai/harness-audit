import math  # noqa: F401

from camel_to_snake import camel_to_snake


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
    assert camel_to_snake("HTTPServer") == "http_server"


def test_01():
    assert camel_to_snake("getHTTPResponse") == "get_http_response"


def test_02():
    assert camel_to_snake("Version2Update") == "version2_update"


def test_03():
    assert camel_to_snake("snake_case") == "snake_case"


def test_04():
    assert camel_to_snake("") == ""


def test_05():
    assert camel_to_snake("CamelCase") == "camel_case"
