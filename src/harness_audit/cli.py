"""Command-line interface: ``harness-audit audit | report | synth | init | run``."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import typer

from . import __version__
from .audit import run_audit
from .io import InputError, read_manifest, read_runs, write_manifest, write_runs
from .report import write_reports
from .schema import AuditResult, Verdict

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Check whether a harness change A → B is supported by evidence for Selection, Activation, "
    "Attribution and Improvement (criteria from Harness-RSI, CC BY 4.0).",
)

EXIT_OK, EXIT_VERDICT, EXIT_INPUT = 0, 1, 2


class FailOn(str, Enum):
    fail = "fail"
    partial = "partial"
    unknown = "unknown"
    never = "never"


_BAD = {
    FailOn.fail: {Verdict.FAIL},
    FailOn.partial: {Verdict.FAIL, Verdict.PARTIAL},
    FailOn.unknown: {Verdict.FAIL, Verdict.PARTIAL, Verdict.UNKNOWN},
    FailOn.never: set(),
}


def exit_code(res: AuditResult, fail_on: FailOn) -> int:
    """``1`` when any claim's verdict is in the ``--fail-on`` set, else ``0``."""
    return EXIT_VERDICT if any(c.verdict in _BAD[fail_on] for c in res.claims) else EXIT_OK


def _print_summary(res: AuditResult, paths: dict[str, Path] | None) -> None:
    colors = {Verdict.PASS: "green", Verdict.PARTIAL: "yellow", Verdict.FAIL: "red", Verdict.UNKNOWN: "white"}
    typer.echo(f"harness-audit {res.version}: {res.experiment}")
    typer.echo(f"primary comparison: {res.primary_comparison}")
    for c in res.claims:
        v = typer.style(f"{c.verdict.value:<8}", fg=colors[c.verdict], bold=True)
        typer.echo(f"  {c.claim:<12} {v} rubric {c.rubric_grade}  {c.summary}")
    for w in res.warnings:
        typer.echo(typer.style(f"  warning: {w}", fg="yellow"))
    if paths:
        typer.echo("report: " + ", ".join(str(p) for p in paths.values()))


@app.command()
def audit(
    runs: Path = typer.Argument(..., help="JSONL file with one RunRecord per line."),
    manifest: Path | None = typer.Option(None, "--manifest", "-m", help="Experiment manifest (YAML or JSON)."),
    out: Path = typer.Option(Path("report"), "--out", "-o", help="Directory for audit.json, report.md, report.html."),
    fail_on: FailOn = typer.Option(FailOn.fail, "--fail-on", help="Exit 1 when any claim has this verdict or worse."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Do not print the summary."),
) -> None:
    """Audit existing run data (no agent is run)."""
    try:
        records = read_runs(runs)
        man = read_manifest(manifest)
    except InputError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(EXIT_INPUT) from e
    res = run_audit(records, man)
    paths = write_reports(res, out)
    if not quiet:
        _print_summary(res, paths)
    raise typer.Exit(exit_code(res, fail_on))


@app.command()
def report(
    audit_json: Path = typer.Argument(..., help="audit.json written by `harness-audit audit`."),
    out: Path = typer.Option(Path("report"), "--out", "-o"),
) -> None:
    """Re-render Markdown and HTML from an existing audit.json."""
    try:
        res = AuditResult.model_validate(json.loads(audit_json.read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:
        typer.echo(f"error: {audit_json}: {e}", err=True)
        raise typer.Exit(EXIT_INPUT) from e
    paths = write_reports(res, out)
    typer.echo("report: " + ", ".join(str(p) for p in paths.values()))


@app.command()
def synth(
    scenario: str = typer.Option("clean", "--scenario", "-s", help="Scenario name, or 'list'."),
    out: Path = typer.Option(Path("synthetic"), "--out", "-o"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Write a synthetic experiment (runs.jsonl + manifest.yaml) with one planted defect."""
    from .synth import SCENARIOS, generate

    if scenario == "list":
        for k, v in SCENARIOS.items():
            typer.echo(f"{k:<22} {v}")
        return
    try:
        records, man = generate(scenario, seed=seed)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(EXIT_INPUT) from e
    write_runs(out / "runs.jsonl", records)
    write_manifest(out / "manifest.yaml", man)
    typer.echo(f"wrote {len(records)} simulated runs to {out / 'runs.jsonl'} and {out / 'manifest.yaml'}")


@app.command()
def init(out: Path = typer.Argument(Path("."), help="Directory for the templates.")) -> None:
    """Write a commented manifest template and a minimal runs.jsonl example."""
    from .templates import MANIFEST_TEMPLATE, RUNS_EXAMPLE

    out.mkdir(parents=True, exist_ok=True)
    for name, text in (("manifest.yaml", MANIFEST_TEMPLATE), ("runs.example.jsonl", RUNS_EXAMPLE)):
        p = out / name
        if p.exists():
            typer.echo(f"skip {p} (exists)")
            continue
        p.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {p}")


@app.command()
def run(
    experiment: Path = typer.Argument(..., help="experiment.yaml describing tasks, harnesses, seeds and model."),
    max_cost: float | None = typer.Option(
        None, "--max-cost", help="Hard cap in USD for this invocation (required unless --dry-run)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan and a cost estimate; call no API."),
    seeds: int | None = typer.Option(None, "--seeds", help="Override the number of replicates per task."),
    aa: bool | None = typer.Option(None, "--aa/--no-aa", help="Also run an independent A/A replicate arm on the report split."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Directory for runs.jsonl, manifest.yaml and the report."),
    cache_dir: Path | None = typer.Option(
        None, "--cache", help="Run cache directory (default: <experiment dir>/.harness-audit-cache)."
    ),
    concurrency: int = typer.Option(2, "--concurrency", "-j", min=1, max=16),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation before spending money."),
    fail_on: FailOn = typer.Option(FailOn.fail, "--fail-on"),
) -> None:
    """Run harness A and B on a task set with an adapter, score, cache, and audit."""
    from .adapters import FatalAdapterError
    from .runner import BudgetError, Runner, load_experiment

    try:
        exp = load_experiment(experiment, seeds=seeds, aa=aa)
    except InputError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(EXIT_INPUT) from e
    runner = Runner(exp, cache_dir=cache_dir, concurrency=concurrency)
    plan = runner.plan()
    try:
        typer.echo(runner.describe(plan))
    except InputError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(EXIT_INPUT) from e
    if dry_run:
        raise typer.Exit(EXIT_OK)
    if max_cost is None:
        typer.echo("error: --max-cost is required for a real run (use --dry-run to estimate first).", err=True)
        raise typer.Exit(EXIT_INPUT)
    est = runner.estimate(plan)
    if est.to_run and not yes:
        typer.confirm(
            f"Spend up to ${max_cost:.2f} (estimate ${est.typical_usd:.2f}, worst case ${est.worst_usd:.2f})?", abort=True
        )
    try:
        records = runner.execute(plan, max_cost=max_cost, progress=typer.echo)
    except (BudgetError, FatalAdapterError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(EXIT_INPUT) from e
    man = runner.manifest()
    write_runs(out / "runs.jsonl", records)
    write_manifest(out / "manifest.yaml", man)
    res = run_audit(records, man)
    paths = write_reports(res, out / "report")
    typer.echo(f"spent ${runner.spent_usd:.4f} (SDK estimate) on {runner.n_executed} new runs; {runner.n_cached} from cache")
    _print_summary(res, paths)
    raise typer.Exit(exit_code(res, fail_on))


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True, help="Show the version."),
) -> None:
    pass


def main() -> None:
    app()
