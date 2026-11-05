# harness-audit — implementation plan (v0.1)

`harness-audit` checks whether a reported harness change **A → B** is supported by
evidence for each of the four claims defined in *Self-Evolving Agent Harness*
(Wei et al., 2026, arXiv:2610.18813; companion repository
[zgcagi/Harness-RSI](https://github.com/zgcagi/Harness-RSI), CC BY 4.0):
**Selection, Activation, Attribution, Improvement**. It reads run data (or produces
it with an adapter) and emits one verdict per claim — `PASS / PARTIAL / FAIL / UNKNOWN` —
with the reasons and the numbers behind them.

## 0. Sources read (M0)

| Source | What it fixes for the code |
|---|---|
| `docs/evidence-criteria.md` (Harness-RSI, 2026-10-17 snapshot) | The C / O / N definitions for each claim; unit of comparison = one primary comparison. |
| `data/claim-audit-rubric.md` (Harness-RSI, incl. revisions of 2026-10-09, 2026-10-10, 2026-10-14) | Decision rules: Selection channels, Activation grades and existence rule, the three Attribution conditions and the "clearing the variation" standard, Improvement = task distance. |
| `README.md` §3.2 (Harness-RSI) | The admission gate `Δ̂ + sqrt(2·ln(K·T/δ)/m) < 0`, its assumptions (independent exogenous tasks, losses in [0,1], fixed model and evaluator), and "repeated rollouts reduce execution noise but do not add independent tasks". |
| `claude-agent-sdk` 0.2.178 source (`types.py`, `query.py`, `_internal/client.py`, `_internal/transport/subprocess_cli.py`) and docs `code.claude.com/docs/en/agent-sdk/{skills,hooks,cost-tracking}` | Adapter API: `query(prompt=, options=ClaudeAgentOptions(...))`; hooks `PreToolUse`/`PostToolUse` via `HookMatcher` (supported by `query()`, which always streams internally); skills are `<cwd>/.claude/skills/<name>/SKILL.md`, enabled with `skills=[...]`, discovered through `setting_sources`; `SystemMessage(subtype="init").data["skills"]` lists loaded skills; `ResultMessage.total_cost_usd` (client-side estimate), `.usage`, `.num_turns`, `.duration_ms`; per-query cap `max_budget_usd`. |
| `claude-api` skill model table (cached 2026-10-06) | Haiku = `claude-haiku-5-5`, $0.10 / $0.50 per MTok input/output for prompts ≤100K tokens ($0.50 / $2.50 above). Sampling parameters are rejected on the current API version, so a "seed" is a replicate index, not a sampling seed. |

Rubric quotations used in code docstrings and the README carry the attribution
"Harness-RSI, CC BY 4.0" and a link to the source file.

## 1. Verdict model

Each claim is evaluated by a set of **checks**. A check has a status:

- `pass` — condition met;
- `warn` — the rubric's partial (`O`) conditions;
- `fail` — the rubric's not-met (`N`) conditions, plus a few hard failures stated below;
- `skip` — data needed for the check is missing;
- `note` — an advisory flag reported prominently that does **not** change the verdict.

Claim verdict: any `fail` → **FAIL**; else any `warn` → **PARTIAL**; else if every
required check is `skip` → **UNKNOWN**; else **PASS**. A required check that is
`skip` while others pass yields **PARTIAL** (the claim is only partly evidenced).

Each claim also reports a **rubric grade** (`C / O / N / U`) computed from the checks
tagged `rubric=True` only, so a reader can see where harness-audit is stricter than the
published rubric. Two places are deliberately stricter, both requested in the spec:

1. Improvement also requires that the gain holds on held-out task groups and that
   retention does not regress (the rubric grades task distance only).
2. A changed evaluator or a changed model is a hard **FAIL** for Attribution
   (the rubric grades it `O`; the paper's PAC analysis assumes a fixed model and evaluator,
   and the rubric's scope layer puts model changes outside harness comparisons).

## 2. Mapping from paper criteria to checks

The primary comparison is **arm B vs arm A on the `report` split** (the scored runs
behind the headline). Activation, Attribution and Improvement are read on it;
Selection reads the whole task registry.

### Selection — "The reported estimate must be unselected, or its inference must account for selection."

| Paper criterion | Check id | Implementation | Status rule |
|---|---|---|---|
| Reporting outcomes kept separate from **proposal** (update generated from reporting tasks' scored evidence) | `split_disjoint_proposal` | `report ∩ proposal` by `task_id`, and by `task_hash` when present (near-duplicate) | overlap = all report tasks → `fail` (N); some → `warn` (O: "reporting set partly coincides with the evolution set"); none → `pass` |
| Separate from **admission** ("a held-out score used for promotion") | `split_disjoint_admission`, `admission_data` | `report ∩ admission`; manifest `selection.admission_data ∈ {admission, report, proposal, none}` | overlap or `admission_data = report` → `warn`; `none` (unconditional commit) → `pass` ("having no selection step is not N") |
| Separate from **report choice** (round/config/hyperparameter picked on reporting partition) | `report_choice` | manifest `selection.report_choice ∈ {fixed, selected}` and `selection.n_compared` | `selected` → `fail` (N: "a headline configuration … picked on the reporting partition") |
| Reused holdout | `reused_holdout` | manifest `selection.report_uses` (times the report split informed a decision) | `> 1` → `warn` ("reused holdout"; cf. Dwork et al. 2015 cited by the paper) |
| Partition shown, not asserted | `partition_shown` | every scored record has a `split`, and the manifest task registry lists task → split | missing → `warn` (O: "split names asserted without the partition being shown") |
| Admission gate on **fresh** tasks (paper §3.2) | `admission_gate` | per-task loss `ℓ = 1 − mean_seed(score)`; `Δ̂ = mean_task(ℓ_B − ℓ_A)`; `m` = number of **independent tasks** (seeds averaged first); `ε = sqrt(2·ln(K·T/δ)/m)`; gate passes iff `Δ̂ + ε < 0`. Computed on `admission` (or on `report` when no admission split, labelled as such) | gate not cleared → `warn`; `m = 0` → `skip`. Extra seeds never increase `m`. |

### Activation — "Identify whether the accepted update operated in the scored runs."

Events per run: `artifact_loaded` (available), `artifact_reached` (reached),
`artifact_carried` (carried), `tool_call` (with optional short `summary`).
Carried can also be derived from **signatures** declared in the manifest for an artifact
(e.g. `{tool: Bash, pattern: "python .*test"}`): a matching `tool_call` after the artifact's
first `reached` event (or after `loaded` for artifacts whose kind has no reach event, such as a
system prompt — "for interpreted guidance, trajectory adherence records whether execution
followed it").

| Paper criterion | Check id | Implementation | Status rule |
|---|---|---|---|
| Availability is the weakest grade; load rates alone are `O` | `available` | rate of B report runs with `artifact_loaded` per changed artifact | reported; used by `reached_or_carried` |
| **C** requires reached or carried evidence tied to the primary comparison; "one confirmed case can qualify; always record coverage" | `reached_or_carried` | per changed artifact: count of B report runs with reached / carried; coverage string `k/n scored runs` | ≥1 run reached/carried for any changed artifact → `pass`; only loaded → `warn` (O); traces present but artifact never loaded/reached → `fail` (N); no events at all → `skip` → UNKNOWN |
| Evidence must come from the scored runs | `bound_to_primary` | only `report`-split B runs count; events from other splits are listed separately | reported |
| Baseline contamination | `baseline_clean` | A runs that load an artifact with B's hash | any → `warn` |
| Conditional effect (spec) | `conditional_effect` | mean score of B runs where the update was reached vs not reached, each paired with A on the same tasks | reported |
| "Improvement not via the update path" (spec) | `gain_without_activation` | reached rate `< thresholds.low_reach` (default 0.2) while `Δ` is positive and its CI excludes 0, or gain in not-reached runs ≥ gain in reached runs | `note` (does not change the verdict: per rubric, "a counterfactual score change belongs to Attribution") |

### Attribution — "isolate the update, control inference resources, and show a margin beyond measured execution variation."

| Paper criterion | Check id | Implementation | Status rule |
|---|---|---|---|
| Arms differ only in the evolved component: evaluator | `same_evaluator` | set of `evaluator_hash` per arm | differ → `fail` (hard rule from spec) |
| … model | `same_model` | set of `model` per arm | differ → `fail` (out of scope as a harness comparison) |
| … harness | `isolated_update` | A and B each have one `harness_hash`; hashes differ; manifest `changed_artifacts` non-empty; manifest `confounds` (e.g. `max_turns`, `effort`, tools changed) empty | identical harnesses → `fail`; confounds or drifting hashes → `warn` |
| **Control inference resources** of the scored runs (calls, generated tokens, turns, tool calls) | `envelope` | per-task paired ratios B/A of `output_tokens`, `num_turns`, `tool_calls`, `cost_usd`; input-token delta reported separately as "context added by the artifact" (the artifact's own context is "the update's own cost … but it has to be declared") | any of output tokens / turns / tool calls / cost ratio > `thresholds.envelope_ratio` (default 1.25) → `warn` ("declared but unmatched"); usage missing → `warn` (not accounted); else `pass`. Also reports cost-normalised score (score per USD, Δscore per extra USD). |
| **Margin beyond measured variation**: "a test distinguishes it from zero at the 5% level; an interval for the difference excludes zero" | `margin` | paired bootstrap over **tasks** (tasks resampled; each task contributes its seed-mean difference; seeds are never resampled independently), 10 000 resamples, percentile CI at level `1 − α/k` (`k = selection.n_compared`, Bonferroni for "a selected maximum"); paired sign-flip permutation test over tasks (exact when `m ≤ 16`, else 20 000 Monte-Carlo flips) | CI lower bound > 0 and `p < α` → `pass`; otherwise `warn` (O: "measured variation that the margin does not clear") |
| Measured variation from repeated executions of frozen states (A/A) | `aa_noise` | if an `A2` arm exists (same harness as A, independent runs) compute `Δ_AA` with the same bootstrap; else if A has ≥2 seeds per task, split-half A/A (even vs odd seeds). Report the A/A 95% band and `Δ_AB` against it | `Δ_AB` ≤ upper edge of A/A band → `warn`; A/A interval itself excludes 0 → `warn` (unstable evaluation); no A/A possible → `note` |
| "Variation across judges, candidate harnesses, task orders, or costs does not satisfy this condition" | (design) | only paired task outcomes and repeated frozen-state runs are used | — |
| `N`: one run per task, no variance estimate, no resource accounting | `measured_variation` | fewer than 2 paired tasks | `fail` |

### Improvement — task distance, plus (spec) held-out gain and retention

| Paper criterion | Check id | Implementation | Status rule |
|---|---|---|---|
| **Task distance**: held-out tasks / unseen benchmark → C; partly overlapping → O; only the tasks that generated the update → N | `task_distance` | report tasks vs proposal tasks (`task_id`/`task_hash`) and vs proposal **groups** (`proposal_groups` from the manifest, or the groups of the proposal split) | all report tasks are proposal tasks → `fail`; partial overlap → `warn`; disjoint → `pass` (label "held-out tasks, same groups" or "held-out groups") |
| Gain holds beyond the groups that formed the update (spec) | `heldout_gain` | per-group paired Δ with bootstrap CI; pooled Δ on report tasks whose group ∉ proposal groups | held-out CI lower > 0 → `pass`; otherwise `warn` (with "gain confined to proposal groups" when same-group Δ is significant); no held-out group → `note` |
| Retention: "probes on an earlier distribution outside the loop's control"; non-regression with a user threshold (spec) | `retention` | on `retention` split: drop `d = score_A − score_B` per task, bootstrap CI; Hoeffding certificate `d̂ + sqrt(2·ln(1/δ)/m) ≤ thresholds.regression` reported as "certified non-regression" | `d̂ > threshold` and CI lower > 0 → `fail`; `d̂ > threshold` otherwise → `warn`; else `pass`; no retention runs → `warn` ("retention untested") |
| Baseline and round recorded, not graded | `context` | manifest `baseline`, `round` echoed in the report | `note` |

## 3. Data schema (framework-independent core)

### `runs.jsonl` — one line per run (`RunRecord`)

```json
{
  "run_id": "r-000123",
  "task_id": "strings/strip_accents", "task_group": "strings", "task_hash": "sha256:…",
  "split": "report",                       // proposal | admission | report | retention
  "arm": "B",                              // A | B | A2 (A2 = independent replicate of A)
  "harness_hash": "sha256:…",
  "seed": 0,                               // replicate index (sampling seeds are not available on the API)
  "model": "claude-haiku-5-5",
  "score": 0.75,                           // in [0,1], from an evaluator outside the harness
  "evaluator_hash": "sha256:…",
  "status": "ok",                          // ok | error | budget_exceeded | timeout
  "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0, "cost_usd": 0.0, "wall_s": 0.0,
            "tool_calls": 0, "num_turns": 0},
  "events": [
    {"t": 0.0, "type": "artifact_loaded",  "artifact": "skills/python-bugfix/SKILL.md", "hash": "sha256:…"},
    {"t": 3.1, "type": "artifact_reached", "artifact": "skills/python-bugfix/SKILL.md", "evidence": "Skill tool invoked: python-bugfix"},
    {"t": 5.4, "type": "tool_call", "name": "Bash", "args_hash": "sha256:…", "summary": "python test_public.py"}
  ]
}
```

Everything except `run_id, task_id, split, arm, harness_hash, model, score, evaluator_hash`
is optional; missing data produces `skip`/UNKNOWN, never `pass`. `events: null` means "no trace
collected" (Activation UNKNOWN); `events: []` means "traced, nothing happened".

### `manifest.yaml` — the experiment

```yaml
experiment: example-python-bugfix
baseline: harness_a (pre-update state)      # recorded, not graded
round: 1
arms: {baseline: A, candidate: B, replicate: A2}
harness: {A: "sha256:…", B: "sha256:…"}
changed_artifacts:                          # computed from the harness directory diff in `run` mode
  - path: skills/python-bugfix/SKILL.md
    kind: skill                             # skill | system_prompt | file | tool | code | config
    change: added                           # added | modified | removed
    hash_a: null
    hash_b: "sha256:…"
    signatures: [{tool: Bash, pattern: "python3? .*test"}]
confounds: []                               # non-artifact differences, e.g. "max_turns 10 -> 20"
proposal_groups: [strings]
selection: {K: 1, T: 1, delta: 0.05, admission_data: admission,
            report_uses: 1, report_choice: fixed, n_compared: 1}
thresholds: {alpha: 0.05, envelope_ratio: 1.25, regression: 0.02, low_reach: 0.2}
tasks:                                      # task registry: shows the partition
  - {task_id: strings/strip_accents, split: report, group: strings, task_hash: "sha256:…"}
```

### Output (`AuditResult`)

`report/audit.json` (full machine-readable result), `report/report.md`, `report/report.html`
(self-contained, inline CSS + SVG). Exit code: `0` no FAIL, `1` at least one FAIL
(configurable `--fail-on {fail,partial,never}`), `2` input/schema error.

## 4. Architecture

```
src/harness_audit/
├── schema.py        # pydantic: RunRecord, Usage, Event, Manifest, CheckResult, ClaimResult, AuditResult
├── io.py            # JSONL/YAML read+write, line-numbered validation errors
├── stats.py         # hoeffding_gate, paired_bootstrap_ci, sign_flip_permutation_test, per-task aggregation
├── data.py          # PairedView: per-task seed means for two arms on a split (the one place pairing happens)
├── claims/{selection,activation,attribution,improvement}.py   # pure functions: (records, manifest) -> ClaimResult
├── audit.py         # runs the four claims, builds AuditResult
├── report.py        # Markdown, HTML, JSON renderers
├── synth.py         # synthetic data generator with planted defects (scenarios)
├── hashing.py       # sha256 of files / directories (sorted relative paths + contents)
├── harness.py       # harness directory loader (harness.yaml, system prompt, skills), diff -> changed_artifacts
├── tasks.py         # task directory loader (task.yaml, workspace/, evaluator/)
├── evaluator.py     # out-of-harness scorer: runs hidden tests in a subprocess, score = passed/total
├── cost.py          # price table + dry-run estimator
├── runner.py        # run mode: task × arm × seed plan, cache, budget guard, concurrency
├── adapters/
│   ├── base.py      # Adapter protocol: async run(task, harness, seed, workdir, budget) -> AdapterResult
│   └── claude_agent_sdk.py   # hooks -> events, ResultMessage -> usage
└── cli.py           # Typer app: audit, run, report, init, synth
```

The core (`schema`, `io`, `stats`, `data`, `claims`, `audit`, `report`) imports nothing from any
agent framework. `claude-agent-sdk` is an optional extra (`pip install harness-audit[claude]`)
imported lazily by the adapter.

### Claude Agent SDK adapter (M3)

- **Harness = directory**: `harness.yaml` (allowed tools, `max_turns`, `effort`, optional
  `system_prompt_file`, `skills`, `signatures`), `system_prompt.md`, `skills/<name>/SKILL.md`, any other
  files. `harness_hash` = sha256 over sorted `(relative path, file bytes)` of the whole directory.
- **Isolation**: each run gets a fresh temp workdir outside any git repo; the task workspace is copied in;
  skills are copied to `<workdir>/.claude/skills/`; `setting_sources=["project"]` so the host's
  `~/.claude` is never loaded; `permission_mode="dontAsk"` with an explicit `allowed_tools` list;
  `max_budget_usd` per run; `model` from the experiment.
- **Events**: `artifact_loaded` from the `init` system message (`data["skills"]`, plus the system prompt
  file which is available by construction); `artifact_reached` from a `PreToolUse` hook when
  `tool_name == "Skill"` names a changed skill, or a `Read` of a changed artifact file; `tool_call` for
  every `PreToolUse` with `args_hash` and a ≤160-char summary for Bash commands and file paths only
  (no file contents, no environment).
- **Usage**: `ResultMessage.total_cost_usd` (SDK estimate; recomputed from tokens with our price table
  if missing), `usage` input/output/cache tokens, `num_turns`, `duration_ms`; tool calls counted by hook.
- **Scoring**: the evaluator (hidden tests) lives in the task directory, never in the workdir the agent
  sees; it runs after the agent finishes on a copy of the final workspace. `evaluator_hash` = hash of the
  evaluator directory.

### Runner (M3)

- Plan = tasks (splits admission, report, retention by default) × arms (A, B, optional A2) × seeds.
  A2 uses seeds offset by 1000 so its cache entries never collide with A.
- Cache key = sha256(task_hash, harness_hash, seed, model, adapter id). Cache entry = RunRecord + final
  workspace, so a changed evaluator re-scores without paying for the agent run again.
- `--dry-run`: prints the plan, cached vs to-run counts, and a cost estimate (typical and worst case at
  `max_turns`) — no API call.
- `--max-cost USD` is required for a real run; the runner never starts a run whose worst-case estimate
  would push the spent total past the cap, and gives each run `max_budget_usd = min(per_run_cap, remaining)`.
  Overshoot is bounded by one response (the SDK includes the crossing response in `total_cost_usd`).
- Interactive confirmation of the estimate unless `--yes`. `ANTHROPIC_API_KEY` is read only by the
  SDK subprocess from the environment; harness-audit never logs, stores, or hashes it.

## 5. Milestones

| | Scope | Done when |
|---|---|---|
| **M0** | Read sources, this plan | `PLAN.md` committed |
| **M1** | schema, io, stats, data, claims, audit, report core; `synth.py` scenarios; tests | each planted defect is caught by its check, the clean scenario is PASS ×4 |
| **M2** | CLI `audit / report / synth / init`, Markdown + HTML + JSON, exit codes, `action.yml` | `harness-audit audit runs.jsonl --manifest m.yaml --out report/` works on user JSONL without the SDK |
| **M3** | harness/task loaders, evaluator, cost estimator, runner with cache/budget/dry-run/A-A, Claude Agent SDK adapter, `examples/` | dry-run on `examples/` prints an estimate; adapter unit-tested with SDK message/hook objects (no API) |
| **M4** | Dogfood with Haiku on `examples/` | **blocked in this session** (no `ANTHROPIC_API_KEY`); needs approval of the dry-run estimate first |
| **M5** | README (Quick Start, four claims, minimal JSONL, adapter guide, known limits, screenshot), CI (ruff + pytest), CITATION.cff, LICENSE | CI green locally; real-run numbers in README are `TBD` until M4 |

### Planted defects (M1 test matrix)

| Scenario | Planted defect | Check that must catch it | Expected verdict |
|---|---|---|---|
| `clean` | none | — | PASS, PASS, PASS, PASS |
| `split_leak` | half of the report tasks also in proposal | `split_disjoint_proposal`, `task_distance` | Selection PARTIAL, Improvement PARTIAL |
| `never_activated` | skill loaded in B but never reached; B still gains | `reached_or_carried` (+ `gain_without_activation` note) | Activation PARTIAL |
| `double_tokens` | B gains only by using ~2× output tokens and turns | `envelope` | Attribution PARTIAL |
| `evaluator_changed` | B scored by a different evaluator | `same_evaluator` | Attribution FAIL |
| `same_group_only` | gain only on the proposal group, zero on held-out groups | `heldout_gain` | Improvement PARTIAL |
| `retention_regression` | B drops on retention tasks beyond the threshold | `retention` | Improvement FAIL |

Each defect scenario also asserts that the *other* claims keep their clean verdict, so a check that
fires everywhere does not pass the suite.

## 6. Risks and open points

1. **Statistical power with few tasks.** The admission gate needs a large effect at small `m`
   (`m = 8`, `K = T = 1`, `δ = 0.05` → `ε ≈ 0.87`). Real dogfood runs will likely show the gate as not
   cleared; the report says so rather than hiding it.
2. **Activation for interpreted guidance** is only as good as the declared signatures; adherence can
   also occur in A, so the report shows the signature's baseline rate in A next to the rate in B.
3. **SDK cost is a client-side estimate** (docs: "not authoritative billing data"). The budget guard uses
   it plus a margin; the README says so.
4. **Claude Agent SDK internals move fast** (bundled CLI 2.1.318). The adapter only uses documented
   options and hook inputs; it is pinned to `claude-agent-sdk>=0.2.178` and isolated behind the
   `Adapter` protocol.
5. **The agent executes Bash** in a temp workdir on the host. Runs use `dontAsk` + an allow-list; the
   README recommends a container for untrusted tasks.
6. **No API key in this session**: M4 cannot run here; the adapter's live path is exercised only by M4.
7. **Nondeterminism**: no sampling seed exists, so replicates are independent draws; caching keys on the
   replicate index makes reruns reproducible from cache, not from the API.
