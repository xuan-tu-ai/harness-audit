from __future__ import annotations

import json

from harness_audit.audit import run_audit
from harness_audit.report import to_html, to_markdown, write_reports
from harness_audit.synth import generate


def test_reports_render_all_claims(tmp_path) -> None:
    records, manifest = generate("double_tokens")
    res = run_audit(records, manifest)
    paths = write_reports(res, tmp_path)
    md = paths["markdown"].read_text()
    for claim in ("Selection", "Activation", "Attribution", "Improvement"):
        assert f"## {claim}" in md
    assert "Harness-RSI" in md and "CC BY 4.0" in md
    data = json.loads(paths["json"].read_text())
    assert [c["verdict"] for c in data["claims"]] == ["PASS", "PASS", "PARTIAL", "PASS"]
    page = paths["html"].read_text()
    assert page.startswith("<!doctype html>") and "<svg" in page and "<script" not in page


def test_markdown_escapes_nothing_dangerous_in_html() -> None:
    records, manifest = generate("clean")
    manifest.experiment = "<script>alert(1)</script>"
    res = run_audit(records, manifest)
    assert "<script>alert" not in to_html(res)
    assert "PASS" in to_markdown(res)
