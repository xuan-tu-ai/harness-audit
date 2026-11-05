def parse_kv(line: str) -> dict[str, str]:
    out = {}
    for part in line.split(";"):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"no '=' in {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out
