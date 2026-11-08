# Polyglot tasks (Exercism Python)

The 34 Python exercises of the [Aider polyglot benchmark](https://github.com/Aider-AI/polyglot-benchmark)
(commit `7e0611e`), which are the hardest exercises of the [Exercism Python track](https://github.com/exercism/python)
(MIT License, see `NOTICE`). `scripts/import_polyglot.py` wrote this directory:

- `workspace/`: the stub file and `INSTRUCTIONS.md` (introduction, instructions, Python append). No
  tests are given to the agent, as in the polyglot benchmark's first attempt.
- `evaluator/`: the Exercism unit tests (`test_<module>.py`, plus helper modules) and the Exercism
  example solution as `reference.py`. Score = share of hidden unit tests passed.
- The prompt follows the polyglot benchmark's wording.

`tests/test_evaluator.py` checks that every reference solution scores 1.0 and every stub scores low.

## How the report split was chosen

The aim was tasks the baseline does not always solve, so that a skill has room to help. Calibration
ran the **baseline only** (harness A, Claude Code without the skill), one run per exercise and model,
with `scripts/calibrate.py` in a cache kept apart from the experiment (so the scored A runs are fresh
runs, not the ones the selection looked at). The rule was fixed before looking at the scores:

1. drop exercises both models solved fully (`retention`: kept to check that nothing regresses, not run);
2. of the rest, take the 12 whose mean score over the two models is closest to 0.5 (`report`);
3. the others are `admission`, not run.

The skill (`harness_b/skills/spec-to-tests`) and its rubric were written before any run of these tasks.

A first calibration was discarded: before runs were isolated, an agent searched the whole filesystem
with `grep -r /` and read one task's hidden tests and reference solution, and in two other runs
downloaded the upstream Exercism tests with `curl`. Every run now uses the adapter's `isolate` option
(no files outside its working directory, no network except the model endpoint), and calibration was
redone. The scores below are from the isolated calibration (2026-11-08). One run per cell: single runs
are noisy (the discarded calibration gave several different scores for the same exercise).

| exercise | deepseek-v4.1-flash | claude-haiku-5.5 | mean | split |
|---|---:|---:|---:|---|
| proverb | 0.00 | 1.00 | 0.50 | report |
| forth | 0.96 | 0.00 | 0.48 | report |
| grep | 0.12 | 1.00 | 0.56 | report |
| react | 1.00 | 0.14 | 0.57 | report |
| dot-dsl | 0.58 | 0.58 | 0.58 | report |
| rest-api | 1.00 | 0.33 | 0.67 | report |
| sgf-parsing | 1.00 | 0.35 | 0.67 | report |
| tree-building | 0.85 | 0.54 | 0.69 | report |
| hangman | 0.57 | 0.86 | 0.71 | report |
| zipper | 1.00 | 0.43 | 0.71 | report |
| pov | 0.73 | 0.93 | 0.83 | report |
| grade-school | 1.00 | 0.75 | 0.88 | report |
| phone-number | 0.90 | 0.90 | 0.90 | admission |
| list-ops | 0.92 | 0.92 | 0.92 | admission |
| poker | 1.00 | 0.84 | 0.92 | admission |
| book-store | 0.05 | 0.05 | 0.05 | admission |
| paasio | 0.04 | 0.04 | 0.04 | admission |
| wordy | 1.00 | 0.96 | 0.98 | admission |
| go-counting | 0.00 | 0.00 | 0.00 | admission |
| affine-cipher | 1.00 | 1.00 | 1.00 | retention |
| beer-song | 1.00 | 1.00 | 1.00 | retention |
| bottle-song | 1.00 | 1.00 | 1.00 | retention |
| bowling | 1.00 | 1.00 | 1.00 | retention |
| connect | 1.00 | 1.00 | 1.00 | retention |
| dominoes | 1.00 | 1.00 | 1.00 | retention |
| food-chain | 1.00 | 1.00 | 1.00 | retention |
| pig-latin | 1.00 | 1.00 | 1.00 | retention |
| robot-name | 1.00 | 1.00 | 1.00 | retention |
| scale-generator | 1.00 | 1.00 | 1.00 | retention |
| simple-linked-list | 1.00 | 1.00 | 1.00 | retention |
| transpose | 1.00 | 1.00 | 1.00 | retention |
| two-bucket | 1.00 | 1.00 | 1.00 | retention |
| variable-length-quantity | 1.00 | 1.00 | 1.00 | retention |
| zebra-puzzle | 1.00 | 1.00 | 1.00 | retention |
