# Token Harbor smoke test (2026-11-08)

Real runs through `adapter: openai_compat` (harness-audit's own minimal agent loop, **not Claude Code**),
on the 8 `report` tasks of `examples/`, harness A (no skill) vs harness B (`python-bugfix` skill), one seed.
Configs: `examples/experiment.tokenharbor.yaml`, `examples/experiment.tokenharbor-haiku.yaml`. Prices are
those published by `GET https://tokenharbor.ai/v1/models` that day; the gateway did not report a per-call
cost, so cost is computed from tokens and those prices.

| Model | Runs | Cost | Score A | Score B | Skill loaded / reached in B |
|---|---:|---:|---:|---:|---|
| `deepseek-v4.1-flash` | 16 | $0.0588 | 8/8 tasks at 1.00 | 8/8 tasks at 1.00 | 8/8 / 8/8 |
| `claude-haiku-5.5` | 16 | $0.0173 | 8/8 tasks at 1.00 | 8/8 tasks at 1.00 | 8/8 / 8/8 |

Every run solved its task with or without the skill, so these tasks cannot show a score gain; the
verdicts (Selection, Attribution, Improvement PARTIAL) say so. The runs were a test of the adherence grader.

## Adherence ("did the agent follow the skill?")

Rate of each rubric item, B counted from the moment the skill was invoked, A over the whole run. After the
grader fix described below:

| Rubric item | Required | DeepSeek B | DeepSeek A | Haiku B | Haiku A |
|---|---|---:|---:|---:|---:|
| followed (every required item) | | 5/8 | 0/8 | 8/8 | 0/8 |
| read_function | no | 8/8 | 8/8 | 8/8 | 8/8 |
| fix | yes | 8/8 | 8/8 | 8/8 | 8/8 |
| public_tests | yes | 8/8 | 8/8 | 8/8 | 8/8 |
| edge_case_asserts | yes | 8/8 | 0/8 | 8/8 | 0/8 |
| fix_after_failure | yes | 0/3 (5 n/a) | n/a | n/a | n/a |
| rerun_after_fix | yes | 0/3 (5 n/a) | n/a | n/a | n/a |

The skill's distinctive step, edge-case `assert`s with `python -c`, appears in every B run of both models
and in no A run.

## What reading the traces found

- **Grader false negative (fixed).** Haiku ran both checks in one command,
  `python test_public.py && python -c "... assert ..."`. The grader required the second step in a later
  call and scored it 0/8. Ordered steps may now share one call when their matches appear in order; the
  reverse order still fails (`tests/test_adherence.py`).
- **Rubric stricter than the agent's behaviour (open).** In the 3 DeepSeek runs where a check failed, the
  agent corrected its own `assert` and re-ran it until it passed, without editing the code. The rubric item
  `fix_after_failure` requires an edit to a `.py` file, so those runs count as not followed. Whether a wrong
  test should be fixed like wrong code is a decision about the rubric, not the grader.
