"""Templates written by ``harness-audit init``."""

MANIFEST_TEMPLATE = """\
# harness-audit manifest: describes one A -> B comparison.
experiment: my-experiment
baseline: "harness A (pre-update state)"   # recorded, not graded
round: 1

arms: {baseline: A, candidate: B, replicate: A2}   # A2 = optional independent A/A replicate

harness:            # optional: expected harness_hash per arm (checked against the records)
  A: "sha256:..."
  B: "sha256:..."

changed_artifacts:  # what differs between A and B
  - path: skills/my-skill/SKILL.md
    kind: skill     # skill | system_prompt | file | tool | code | config
    change: added   # added | modified | removed
    hash_b: "sha256:..."
    rubric:         # optional: the artifact's instructions, checked against each trajectory (adherence).
                    # carried = every required item met, counting from when the artifact was reached.
      - {id: fix, text: "Fix the code", tool: "Edit|Write", pattern: '\\.py$'}
      - {id: tests, text: "Then run the tests", tool: Bash, pattern: "pytest", after: [fix]}
      - {id: refix, text: "If tests fail, fix and rerun", tool: "Edit|Write", on_failure_of: [tests]}
      - {id: plan, text: "Plan before editing", check: judge, required: false}   # needs an external judge
    # signatures: [{tool: Bash, pattern: "pytest"}]   # shorthand: a one-item rubric
    applies_to: [strings]   # optional: task ids or groups (globs) where the update is meant to apply,
                            # declared before seeing outcomes; the denominator of the load/reach rates

confounds: []       # other differences between arms (e.g. "max_turns 10 -> 20"); any entry -> Attribution PARTIAL

proposal_groups: [strings]   # task groups that formed the update (default: groups of the proposal split)

selection:
  K: 1              # candidates compared per round
  T: 1              # rounds
  delta: 0.05       # gate confidence: admit iff mean loss diff + sqrt(2 ln(K T / delta) / m) < 0
  admission_data: admission   # admission | report | proposal | none (unconditional commit)
  report_uses: 1    # how many decisions the report split has informed (>1 = reused holdout)
  report_choice: fixed        # fixed | selected (round/config picked on the report split)
  n_compared: 1     # rows the reported configuration was chosen among

oracle:             # how the scorer was kept out of the agent's reach
  outside_workspace: true   # hidden tests and reference solutions are not in the agent's workspace
  protected: ["(^|/)evaluator/", "test_hidden"]   # regexes; a traced tool call matching one touched the scorer

thresholds:
  alpha: 0.05
  envelope_ratio: 1.25   # B/A ratio of output tokens, turns or tool calls that counts as unmatched compute
  regression: 0.02       # allowed mean score drop on the retention split
  low_reach: 0.2         # activation rate below which a gain is flagged as not via the update path

tasks:              # the task registry shows the partition (task_hash detects renamed duplicates)
  - {task_id: t-001, split: proposal, group: strings, task_hash: "sha256:..."}
  - {task_id: t-101, split: report, group: strings}
  - {task_id: t-201, split: report, group: numbers}
  - {task_id: t-301, split: retention, group: legacy}
"""

RUNS_EXAMPLE = """\
{"run_id": "1", "task_id": "t-101", "task_group": "strings", "split": "report", "arm": "A", "harness_hash": "sha256:aa", "seed": 0, "model": "my-model", "score": 0.5, "evaluator_hash": "sha256:ev", "usage": {"output_tokens": 900, "num_turns": 5, "tool_calls": 7, "cost_usd": 0.004}, "events": [{"t": 1, "type": "tool_call", "name": "Bash", "summary": "pytest -q"}]}
{"run_id": "2", "task_id": "t-101", "task_group": "strings", "split": "report", "arm": "B", "harness_hash": "sha256:bb", "seed": 0, "model": "my-model", "score": 1.0, "evaluator_hash": "sha256:ev", "usage": {"output_tokens": 950, "num_turns": 5, "tool_calls": 8, "cost_usd": 0.0045}, "events": [{"t": 0, "type": "artifact_loaded", "artifact": "skills/my-skill/SKILL.md"}, {"t": 2, "type": "artifact_reached", "artifact": "skills/my-skill/SKILL.md", "evidence": "skill invoked"}, {"t": 3, "type": "tool_call", "name": "Bash", "summary": "pytest -q"}]}
"""
