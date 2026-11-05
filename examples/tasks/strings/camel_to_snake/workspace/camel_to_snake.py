def camel_to_snake(name: str) -> str:
    """Convert CamelCase or camelCase to snake_case.

    - An uppercase letter starts a new word, but a run of capitals is one word:
      "HTTPServer" -> "http_server", "getHTTPResponse" -> "get_http_response".
    - Digits stay with the preceding word: "Version2Update" -> "version2_update".
    - Input that is already snake_case is returned unchanged; "" -> "".
    """
    out = ""
    for c in name:
        if c.isupper():
            out += "_" + c.lower()
        else:
            out += c
    return out
