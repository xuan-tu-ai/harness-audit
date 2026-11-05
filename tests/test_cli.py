from __future__ import annotations

import json

from typer.testing import CliRunner

from harness_audit.cli import app

runner = CliRunner()


def test_synth_then_audit_clean_exits_zero(tmp_path) -> None:
    r = runner.invoke(app, ["synth", "--scenario", "clean", "--out", str(tmp_path / "d")])
    assert r.exit_code == 0, r.output
    r = runner.invoke(
        app, ["audit", str(tmp_path / "d/runs.jsonl"), "-m", str(tmp_path / "d/manifest.yaml"), "-o", str(tmp_path / "rep")]
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "rep/report.html").exists() and (tmp_path / "rep/report.md").exists()
    assert [c["verdict"] for c in json.loads((tmp_path / "rep/audit.json").read_text())["claims"]] == ["PASS"] * 4


def test_fail_verdict_exits_one_and_fail_on_controls_it(tmp_path) -> None:
    runner.invoke(app, ["synth", "-s", "evaluator_changed", "-o", str(tmp_path / "d")])
    args = ["audit", str(tmp_path / "d/runs.jsonl"), "-m", str(tmp_path / "d/manifest.yaml"), "-o", str(tmp_path / "rep"), "-q"]
    assert runner.invoke(app, args).exit_code == 1
    assert runner.invoke(app, [*args, "--fail-on", "never"]).exit_code == 0


def test_partial_with_fail_on_partial(tmp_path) -> None:
    runner.invoke(app, ["synth", "-s", "double_tokens", "-o", str(tmp_path / "d")])
    args = ["audit", str(tmp_path / "d/runs.jsonl"), "-m", str(tmp_path / "d/manifest.yaml"), "-o", str(tmp_path / "rep"), "-q"]
    assert runner.invoke(app, args).exit_code == 0
    assert runner.invoke(app, [*args, "--fail-on", "partial"]).exit_code == 1


def test_bad_input_exits_two_with_line_number(tmp_path) -> None:
    p = tmp_path / "runs.jsonl"
    p.write_text('{"run_id": "1"}\n')
    r = runner.invoke(app, ["audit", str(p)])
    assert r.exit_code == 2
    assert "runs.jsonl:1" in r.output


def test_audit_without_manifest_on_init_example(tmp_path) -> None:
    assert runner.invoke(app, ["init", str(tmp_path)]).exit_code == 0
    r = runner.invoke(app, ["audit", str(tmp_path / "runs.example.jsonl"), "-o", str(tmp_path / "rep")])
    assert r.exit_code in (0, 1), r.output
    assert "Attribution" in r.output


def test_report_rerenders_from_json(tmp_path) -> None:
    runner.invoke(app, ["synth", "-o", str(tmp_path / "d")])
    runner.invoke(
        app, ["audit", str(tmp_path / "d/runs.jsonl"), "-m", str(tmp_path / "d/manifest.yaml"), "-o", str(tmp_path / "a"), "-q"]
    )
    r = runner.invoke(app, ["report", str(tmp_path / "a/audit.json"), "-o", str(tmp_path / "b")])
    assert r.exit_code == 0 and (tmp_path / "b/report.md").read_text() == (tmp_path / "a/report.md").read_text()


def test_synth_list() -> None:
    r = runner.invoke(app, ["synth", "-s", "list"])
    assert "retention_regression" in r.output
