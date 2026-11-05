"""Render an AuditResult as Markdown, self-contained HTML, and JSON."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .schema import AuditResult, CheckResult, ClaimResult, Status, Verdict

SOURCE_NOTE = (
    "Criteria follow *Self-Evolving Agent Harness* (Wei et al., 2026, arXiv:2610.18813) and the claim-audit rubric "
    "of [zgcagi/Harness-RSI](https://github.com/zgcagi/Harness-RSI) (`docs/evidence-criteria.md`, "
    "`data/claim-audit-rubric.md`), licensed CC BY 4.0. harness-audit is an independent implementation."
)
ICON = {Status.PASS: "✅", Status.WARN: "◐", Status.FAIL: "❌", Status.SKIP: "∅", Status.NOTE: "⚑"}
VERDICT_ICON = {Verdict.PASS: "✅", Verdict.PARTIAL: "◐", Verdict.FAIL: "❌", Verdict.UNKNOWN: "❔"}


def _f(x: Any, nd: int = 3, signed: bool = False) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return f"{x:+.{nd}f}" if signed else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
    return str(x)


def _detail_rows(ch: CheckResult) -> tuple[list[str], list[list[str]]] | None:
    """Optional small table of key numbers for a check (shared by Markdown and HTML)."""
    m = ch.metrics
    if ch.id == "envelope" and "per_task_means" in m:
        nd = {"cost_usd": 5, "wall_s": 1}
        rows = [
            [k, _f(v["A"], nd.get(k, 1)), _f(v["B"], nd.get(k, 1)), _f(v["ratio_B_over_A"], 2)]
            for k, v in m["per_task_means"].items()
        ]
        return ["per-task mean", "A", "B", "B / A"], rows
    if ch.id == "reached_or_carried" and "per_artifact" in m:

        def cell(count: object, rate: object) -> str:
            return f"{count} ({_f(rate, 2)})" if rate is not None else f"{count} (undefined)"

        rows = [
            [
                p,
                str(d["applicable"]),
                cell(d["available"], d["load_rate"]),
                cell(d["reached"], d["reach_rate"]),
                cell(d["carried"], d["carry_rate"]),
                _f(d.get("signature_rate_A"), 2),
                _f(d.get("signature_rate_B"), 2),
            ]
            for p, d in m["per_artifact"].items()
        ]
        head = ["artifact", "applicable runs", "load", "reach", "carry", "signature rate A", "signature rate B"]
        return head, rows
    if ch.id == "heldout_gain" and "per_group" in m:
        rows = [
            [
                g,
                "proposal" if d["in_proposal_groups"] else "held-out",
                str(d["m_tasks"]),
                _f(d["score_A"]),
                _f(d["score_B"]),
                _f(d["mean"], signed=True),
                f"[{_f(d['ci_low'], signed=True)}, {_f(d['ci_high'], signed=True)}]",
            ]
            for g, d in m["per_group"].items()
        ]
        return ["group", "role", "tasks", "score A", "score B", "Δ", "CI"], rows
    if ch.id == "admission_gate" and "epsilon" in m:
        return ["split", "m tasks", "mean Δ loss", "ε", "bound", "K", "T", "δ"], [
            [
                m["split"],
                str(m["m_tasks"]),
                _f(m["delta_hat_loss"], signed=True),
                _f(m["epsilon"]),
                _f(m["bound"], signed=True),
                str(m["K"]),
                str(m["T"]),
                _f(m["delta"], 2),
            ]
        ]
    return None


def _claim_line(c: ClaimResult) -> str:
    return f"| **{c.claim}** | {VERDICT_ICON[c.verdict]} {c.verdict.value} | {c.rubric_grade} | {c.summary} |"


def to_markdown(res: AuditResult) -> str:
    out = [
        f"# harness-audit report: {res.experiment}",
        "",
        f"Primary comparison: {res.primary_comparison}. Records: {res.n_records}. harness-audit {res.version}.",
        "",
        "| Claim | Verdict | Rubric grade | Summary |",
        "|---|---|:-:|---|",
    ]
    out += [_claim_line(c) for c in res.claims]
    out += [
        "",
        "Rubric grade: C met, O partial, N not met, U unknown, computed from the rubric's own conditions only. "
        "The verdict can be stricter (see each check).",
    ]
    if res.warnings:
        out += ["", "**Data warnings**", ""] + [f"- {w}" for w in res.warnings]
    for c in res.claims:
        out += ["", f"## {c.claim}: {c.verdict.value} (rubric {c.rubric_grade})", "", "| | Check | Finding |", "|:-:|---|---|"]
        for ch in c.checks:
            tag = "" if ch.rubric else " *(harness-audit)*"
            out.append(f"| {ICON[ch.status]} | {ch.title}{tag} | {ch.reason} |")
        for ch in c.checks:
            d = _detail_rows(ch)
            if d:
                head, rows = d
                out += ["", f"*{ch.title}*", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                out += ["| " + " | ".join(r) + " |" for r in rows]
    out += [
        "",
        "Legend: ✅ pass · ◐ partial · ❌ fail · ∅ missing data · ⚑ advisory note (does not change the verdict).",
        "",
        "---",
        "",
        SOURCE_NOTE,
        "",
    ]
    return "\n".join(out)


_CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e3e0;--card:#fff;
--pass:#1a7f37;--warn:#9a6700;--fail:#cf222e;--skip:#6e7781;--note:#5a32a3;--bar:#3a6ea5}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121214;--fg:#e8e8ea;--muted:#9a9aa2;
--line:#2c2c31;--card:#1a1a1d;--pass:#3fb950;--warn:#d29922;--fail:#f85149;--skip:#8b949e;--note:#a371f7;--bar:#6ca0dc}}
:root[data-theme=dark]{--bg:#121214;--fg:#e8e8ea;--muted:#9a9aa2;--line:#2c2c31;--card:#1a1a1d;--pass:#3fb950;
--warn:#d29922;--fail:#f85149;--skip:#8b949e;--note:#a371f7;--bar:#6ca0dc}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 16px 48px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:28px 0 8px;display:flex;gap:10px;align-items:center}.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card .name{font-weight:600}.card .v{font-size:20px;font-weight:700;margin:4px 0}.card .s{font-size:13px;color:var(--muted)}
.b{display:inline-block;border-radius:999px;padding:1px 10px;font-size:12px;font-weight:700;color:#fff}
.PASS,.pass{background:var(--pass)}.PARTIAL,.warn{background:var(--warn)}.FAIL,.fail{background:var(--fail)}
.UNKNOWN,.skip{background:var(--skip)}.note{background:var(--note)}
.tPASS{color:var(--pass)}.tPARTIAL{color:var(--warn)}.tFAIL{color:var(--fail)}.tUNKNOWN{color:var(--skip)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:6px 0 10px}th,td{text-align:left;padding:6px 8px;
border-bottom:1px solid var(--line);vertical-align:top}th{font-weight:600;color:var(--muted)}
.wrap{overflow-x:auto}.crit{font-size:12px;color:var(--muted)}.warnbox{border-left:3px solid var(--warn);padding:6px 12px;
background:var(--card)}svg text{fill:var(--fg);font:12px system-ui,sans-serif}svg text.muted{fill:var(--muted)}footer{margin-top:32px;font-size:13px;color:var(--muted)}
a{color:var(--bar)}
"""


def _forest(res: AuditResult) -> str:
    """Inline SVG: point estimates with intervals for the primary Δ, A/A band, groups, retention."""
    rows: list[tuple[str, float, float, float]] = []
    try:
        m = res.check("margin").metrics
        rows.append(("Δ score B − A (report)", m["mean"], m["ci_low"], m["ci_high"]))
    except KeyError:
        pass
    try:
        aa = res.check("aa_noise").metrics.get("aa")
        if aa:
            rows.append(("A/A noise band", aa["mean"], aa["ci_low"], aa["ci_high"]))
    except KeyError:
        pass
    try:
        for g, d in (res.check("heldout_gain").metrics.get("per_group") or {}).items():
            role = "proposal group" if d["in_proposal_groups"] else "held-out group"
            rows.append((f"{g} ({role})", d["mean"], d["ci_low"], d["ci_high"]))
    except KeyError:
        pass
    try:
        dr = res.check("retention").metrics.get("drop")
        if dr:
            rows.append(("retention Δ (B − A)", -dr["mean"], -dr["ci_high"], -dr["ci_low"]))
    except KeyError:
        pass
    if not rows:
        return ""
    lo = min(-0.1, *(r[2] for r in rows))
    hi = max(0.1, *(r[3] for r in rows))
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    w, left, right, rh = 760, 230, 30, 26
    h = rh * len(rows) + 34

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (w - left - right)

    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Effect estimates with intervals">',
        f'<line x1="{x(0):.1f}" y1="6" x2="{x(0):.1f}" y2="{h - 26}" style="stroke:var(--muted)" stroke-dasharray="3 3"/>',
    ]
    for i, (label, mean, a, b) in enumerate(rows):
        y = 18 + i * rh
        parts.append(f'<text x="{left - 10}" y="{y + 4}" text-anchor="end">{html.escape(label)}</text>')
        parts.append(
            f'<line x1="{x(a):.1f}" y1="{y}" x2="{x(b):.1f}" y2="{y}" style="stroke:var(--bar)" stroke-width="3" '
            'stroke-linecap="round"/>'
        )
        parts.append(f'<circle cx="{x(mean):.1f}" cy="{y}" r="5" style="fill:var(--bar)"/>')
        parts.append(f'<text x="{x(b) + 8:.1f}" y="{y + 4}" class="muted">{mean:+.3f}</text>')
    for v in (lo + pad, 0.0, hi - pad):
        parts.append(f'<text x="{x(v):.1f}" y="{h - 8}" text-anchor="middle" class="muted">{v:+.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _table_html(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    trs = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def to_html(res: AuditResult) -> str:
    e = html.escape
    cards = "".join(
        f'<div class="card"><div class="name">{e(c.claim)}</div><div class="v t{c.verdict.value}">{c.verdict.value}</div>'
        f'<div class="s">rubric grade {c.rubric_grade}</div><div class="s">{e(c.summary)}</div></div>'
        for c in res.claims
    )
    body = [
        f"<h1>harness-audit report</h1><div class='muted'>{e(res.experiment)} · {e(res.primary_comparison)} · "
        f"{res.n_records} records · v{e(res.version)}</div>",
        f'<div class="grid">{cards}</div>',
    ]
    forest = _forest(res)
    if forest:
        body.append(f'<div class="card"><div class="name">Effect estimates (intervals at the audit level)</div>{forest}</div>')
    if res.warnings:
        body.append(
            '<h2>Data warnings</h2><div class="warnbox">' + "".join(f"<div>{e(w)}</div>" for w in res.warnings) + "</div>"
        )
    for c in res.claims:
        body.append(
            f'<h2>{e(c.claim)} <span class="b {c.verdict.value}">{c.verdict.value}</span>'
            f'<span class="muted" style="font-size:13px;font-weight:400">rubric {c.rubric_grade}</span></h2>'
        )
        rows = []
        for ch in c.checks:
            tag = "" if ch.rubric else ' <span class="muted">(harness-audit)</span>'
            crit = f'<div class="crit">{e(ch.criterion)}</div>' if ch.criterion else ""
            rows.append(
                f'<tr><td><span class="b {ch.status.value}">{ch.status.value}</span></td>'
                f"<td>{e(ch.title)}{tag}</td><td>{e(ch.reason)}{crit}</td></tr>"
            )
        body.append(
            '<div class="wrap"><table><thead><tr><th>status</th><th>check</th><th>finding</th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div>"
        )
        for ch in c.checks:
            d = _detail_rows(ch)
            if d:
                body.append(f'<div class="muted" style="margin-top:6px">{e(ch.title)}</div>' + _table_html(*d))
    note = SOURCE_NOTE.replace(
        "[zgcagi/Harness-RSI](https://github.com/zgcagi/Harness-RSI)",
        '<a href="https://github.com/zgcagi/Harness-RSI">zgcagi/Harness-RSI</a>',
    )
    note = note.replace("*Self-Evolving Agent Harness*", "<em>Self-Evolving Agent Harness</em>").replace("`", "")
    body.append(
        f"<footer>{note} Status legend: pass, warn = partial, fail, skip = missing data, note = advisory "
        "(does not change the verdict).</footer>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width,initial-scale=1'><title>harness-audit report</title>"
        f"<style>{_CSS}</style></head><body><main>{''.join(body)}</main></body></html>"
    )


def to_json(res: AuditResult) -> str:
    return json.dumps(json.loads(res.model_dump_json()), indent=2, ensure_ascii=False)


def write_reports(res: AuditResult, out_dir: str | Path) -> dict[str, Path]:
    """Write ``audit.json``, ``report.md`` and ``report.html`` into ``out_dir``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = {"json": d / "audit.json", "markdown": d / "report.md", "html": d / "report.html"}
    paths["json"].write_text(to_json(res), encoding="utf-8")
    paths["markdown"].write_text(to_markdown(res), encoding="utf-8")
    paths["html"].write_text(to_html(res), encoding="utf-8")
    return paths
