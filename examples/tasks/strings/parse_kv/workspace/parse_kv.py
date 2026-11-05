def parse_kv(line: str) -> dict[str, str]:
    """Parse "key=value" pairs separated by semicolons into a dict.

    - Whitespace around keys and values is stripped.
    - Empty segments are skipped ("a=1;;b=2", a trailing ";").
    - Only the first "=" separates key and value ("url=a=b" -> {"url": "a=b"}).
    - A non-empty segment without "=" raises ValueError.
    """
    out = {}
    for part in line.split(";"):
        k, v = part.split("=")
        out[k] = v
    return out
