"""Run mode: plan task × arm × replicate, run with an adapter, score, cache, budget-guard.

Cost control: a real run needs ``--max-cost``. Before a job starts, its worst-case
estimate is reserved; a job whose reservation would push the total past the cap is
not started. Each job also gets the SDK's per-query ``max_budget_usd``. Cached jobs
cost nothing. When a run reports no cost, its reservation is counted as spent.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .adapters import Adapter, AdapterResult, FatalAdapterError, RunRequest, load_adapter
from .cost import estimate_run_usd
from .evaluator import evaluator_hash, score_workspace
from .harness import Harness, diff_harnesses, load_harness
from .hashing import sha256_bytes
from .io import InputError
from .schema import Arms, Manifest, Oracle, RunRecord, SelectionConfig, TaskEntry, Thresholds
from .tasks import Task, load_tasks

A2_SEED_OFFSET = 1000


class ExperimentConfig(BaseModel):
    """Contents of ``experiment.yaml`` (paths are relative to the file)."""

    model_config = ConfigDict(extra="forbid")

    experiment: str = "experiment"
    baseline: str | None = None
    round: int | None = None
    model: str = "claude-haiku-5-5"
    adapter: str = "claude_agent_sdk"
    tasks_dir: str = "tasks"
    harness_a: str = "harness_a"
    harness_b: str = "harness_b"
    seeds: int = Field(3, ge=1)
    aa: bool = True
    splits: list[str] = Field(default_factory=lambda: ["admission", "report", "retention"])
    per_run_max_usd: float = Field(0.25, gt=0)
    timeout_s: float = Field(600, gt=0)
    eval_timeout_s: float = Field(60, gt=0)
    proposal_groups: list[str] | None = None
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    thresholds: Thresholds = Field(default_factory=Thresholds)


@dataclass
class Experiment:
    path: Path
    config: ExperimentConfig
    tasks: list[Task]
    all_tasks: list[Task]
    harness_a: Harness
    harness_b: Harness


def load_experiment(path: str | Path, seeds: int | None = None, aa: bool | None = None) -> Experiment:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        cfg = ExperimentConfig.model_validate(raw)
    except FileNotFoundError as e:
        raise InputError(f"{p}: file not found") from e
    except (yaml.YAMLError, ValidationError) as e:
        raise InputError(f"{p}: {e}") from e
    if seeds is not None:
        cfg.seeds = seeds
    if aa is not None:
        cfg.aa = aa
    base = p.parent
    all_tasks = load_tasks(base / cfg.tasks_dir)
    tasks = [t for t in all_tasks if t.spec.split in cfg.splits]
    if not tasks:
        raise InputError(f"{p}: no tasks in splits {cfg.splits}")
    return Experiment(p, cfg, tasks, all_tasks, load_harness(base / cfg.harness_a), load_harness(base / cfg.harness_b))


@dataclass(frozen=True)
class Job:
    task: Task
    arm: str
    harness: Harness
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.arm}-{self.task.id}-s{self.seed}"


@dataclass
class Estimate:
    n_jobs: int
    cached: int
    to_run: int
    typical_usd: float
    worst_usd: float
    per_run_typical: dict[str, float] = field(default_factory=dict)
    per_run_worst: dict[str, float] = field(default_factory=dict)
    calibrated: bool = False


class BudgetError(Exception):
    pass


class Runner:
    """Plans, executes (with cache and budget guard) and assembles records + manifest."""

    def __init__(
        self,
        exp: Experiment,
        cache_dir: Path | None = None,
        concurrency: int = 2,
        adapter: Adapter | None = None,
        workdir_base: Path | None = None,
    ) -> None:
        self.exp = exp
        self.cache_dir = Path(cache_dir) if cache_dir else exp.path.parent / ".harness-audit-cache"
        self.concurrency = concurrency
        self._adapter = adapter
        self.workdir_base = workdir_base
        self.spent_usd = 0.0
        self.n_executed = 0
        self.n_cached = 0
        self.skipped: list[str] = []

    @property
    def adapter(self) -> Adapter:
        if self._adapter is None:
            self._adapter = load_adapter(self.exp.config.adapter)
        return self._adapter

    @property
    def adapter_id(self) -> str:
        return self.adapter.id

    # ---- planning -------------------------------------------------------------------------
    def plan(self) -> list[Job]:
        c = self.exp.config
        jobs: list[Job] = []
        for t in self.exp.tasks:
            for s in range(c.seeds):
                jobs.append(Job(t, "A", self.exp.harness_a, s))
                jobs.append(Job(t, "B", self.exp.harness_b, s))
                if c.aa and t.spec.split == "report":
                    jobs.append(Job(t, "A2", self.exp.harness_a, s + A2_SEED_OFFSET))
        return jobs

    def cache_key(self, job: Job) -> str:
        blob = json.dumps([job.task.task_hash, job.harness.hash, job.seed, self.exp.config.model, self.adapter_id])
        return sha256_bytes(blob.encode()).split(":", 1)[1]

    def _entry(self, job: Job) -> Path:
        k = self.cache_key(job)
        return self.cache_dir / k[:2] / k

    def cached(self, job: Job) -> RunRecord | None:
        f = self._entry(job) / "record.json"
        if not f.exists():
            return None
        try:
            return RunRecord.model_validate_json(f.read_text(encoding="utf-8"))
        except ValidationError:
            return None

    def _extra_context(self, h: Harness) -> int:
        # System prompt (~3 chars/token) plus ~70 tokens per skill listing entry; a skill body
        # enters the context only when invoked and is covered by growth_per_call.
        return len(h.system_prompt or "") // 3 + 70 * len(h.skills)

    def estimate(self, jobs: list[Job]) -> Estimate:
        c = self.exp.config
        todo = [j for j in jobs if self.cached(j) is None]
        observed = [r.usage.cost_usd for j in jobs if (r := self.cached(j)) and r.usage and r.usage.cost_usd]
        per_t: dict[str, float] = {}
        per_w: dict[str, float] = {}
        for arm, h in (("A", self.exp.harness_a), ("B", self.exp.harness_b)):
            t, w = estimate_run_usd(c.model, h.config.max_turns, self._extra_context(h))
            per_t[arm], per_w[arm] = t, min(w, c.per_run_max_usd)
        per_t["A2"], per_w["A2"] = per_t["A"], per_w["A"]
        calibrated = len(observed) >= 5
        if calibrated:
            mean = sum(observed) / len(observed)
            per_t = {k: mean for k in per_t}
        typical = sum(per_t[j.arm] for j in todo)
        worst = sum(per_w[j.arm] for j in todo)
        return Estimate(len(jobs), len(jobs) - len(todo), len(todo), typical, worst, per_t, per_w, calibrated)

    def describe(self, jobs: list[Job]) -> str:
        c = self.exp.config
        est = self.estimate(jobs)
        by = Counter((j.task.spec.split, j.arm) for j in jobs)
        lines = [
            f"experiment: {c.experiment}  model: {c.model}  adapter: {c.adapter}",
            f"harness A {self.exp.harness_a.hash[:19]}  B {self.exp.harness_b.hash[:19]}",
            f"tasks: {len(self.exp.tasks)} in splits {c.splits}; replicates: {c.seeds}; A/A arm: {'yes' if c.aa else 'no'}",
            "jobs: " + ", ".join(f"{s}/{a}={n}" for (s, a), n in sorted(by.items())),
            f"total {est.n_jobs} jobs: {est.cached} cached, {est.to_run} to run",
            f"estimated cost of jobs to run: typical ${est.typical_usd:.2f}, worst case ${est.worst_usd:.2f} "
            f"(per run typical ${est.per_run_typical['B']:.4f}, worst ${est.per_run_worst['B']:.4f}"
            f"{', calibrated on cached runs' if est.calibrated else ', heuristic'})",
        ]
        return "\n".join(lines)

    # ---- execution ------------------------------------------------------------------------
    def execute(self, jobs: list[Job], max_cost: float, progress: Callable[[str], None] | None = None) -> list[RunRecord]:
        if max_cost <= 0:
            raise BudgetError("--max-cost must be positive")
        return asyncio.run(self._execute(jobs, max_cost, progress or (lambda s: None)))

    async def _execute(self, jobs: list[Job], max_cost: float, progress: Callable[[str], None]) -> list[RunRecord]:
        records: dict[str, RunRecord] = {}
        todo: list[Job] = []
        for j in jobs:
            r = self.cached(j)
            if r is None:
                todo.append(j)
                continue
            r = self._rescore_if_needed(j, r)
            records[j.run_id] = r
            self.n_cached += 1
        if todo:
            preflight = getattr(self.adapter, "preflight", None)
            if preflight is not None:
                preflight()
        reserved = 0.0
        lock = asyncio.Lock()
        sem = asyncio.Semaphore(self.concurrency)
        stop = False

        async def one(j: Job) -> None:
            nonlocal reserved, stop
            async with sem:
                async with lock:
                    if stop:
                        self.skipped.append(j.run_id)
                        return
                    cap = self.exp.config.per_run_max_usd
                    remaining = max_cost - self.spent_usd - reserved
                    if cap > remaining:
                        stop = True
                        self.skipped.append(j.run_id)
                        progress(f"budget: stopping before {j.run_id} (remaining ${remaining:.4f} < per-run cap ${cap:.4f})")
                        return
                    reserved += cap
                try:
                    rec = await self._run_job(j, cap)
                finally:
                    async with lock:
                        reserved -= cap
                cost = rec.usage.cost_usd if rec.usage and rec.usage.cost_usd is not None else cap
                async with lock:
                    self.spent_usd += cost
                    self.n_executed += 1
                    records[j.run_id] = rec
                progress(
                    f"[{self.n_executed}/{len(todo)}] {j.run_id}: score {rec.score:.2f} status {rec.status} "
                    f"cost ${cost:.4f} total ${self.spent_usd:.4f}"
                )

        await asyncio.gather(*(one(j) for j in todo))
        if self.skipped:
            progress(f"budget: {len(self.skipped)} job(s) not run; audit uses the {len(records)} completed runs")
        return [records[j.run_id] for j in jobs if j.run_id in records]

    def _workdir(self) -> Path:
        base = self.workdir_base
        if base is not None:
            base.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="harness-audit-run-", dir=base))

    async def _run_job(self, j: Job, budget: float) -> RunRecord:
        c = self.exp.config
        work = self._workdir()
        eh_before = evaluator_hash(j.task)
        try:
            shutil.copytree(j.task.workspace, work, dirs_exist_ok=True)
            req = RunRequest(j.task, j.harness, work, j.seed, c.model, budget, c.timeout_s)
            try:
                res = await self.adapter.run(req)
            except FatalAdapterError:
                raise
            except Exception as e:  # noqa: BLE001 - one broken run must not lose the others
                res = AdapterResult(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
            eh_after = evaluator_hash(j.task)  # before scoring: did the agent's run touch the scorer?
            for p in self.adapter.installed_paths():
                shutil.rmtree(work / p, ignore_errors=True)
            entry = self._entry(j)
            final = entry / "workspace"
            if final.exists():
                shutil.rmtree(final)
            shutil.copytree(work, final, ignore=shutil.ignore_patterns("__pycache__"))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        sc = await asyncio.to_thread(score_workspace, j.task, final, c.eval_timeout_s)
        rec = RunRecord(
            run_id=j.run_id,
            task_id=j.task.id,
            task_group=j.task.spec.group,
            task_hash=j.task.task_hash,
            split=j.task.spec.split,
            arm=j.arm,
            harness_hash=j.harness.hash,
            seed=j.seed,
            model=res.model or c.model,
            score=sc.score,
            evaluator_hash=eh_before,
            evaluator_hash_after=eh_after,
            status=res.status if not sc.error else f"{res.status};eval_error",
            usage=res.usage,
            events=res.events,
        )
        (entry / "record.json").write_text(rec.model_dump_json(), encoding="utf-8")
        (entry / "detail.json").write_text(
            json.dumps(
                {
                    "error": res.error,
                    "eval": {"passed": sc.passed, "total": sc.total, "failures": sc.failures, "error": sc.error},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return rec

    def _rescore_if_needed(self, j: Job, r: RunRecord) -> RunRecord:
        eh = evaluator_hash(j.task)
        if r.evaluator_hash == eh:
            return r
        final = self._entry(j) / "workspace"
        if not final.exists():
            return r
        sc = score_workspace(j.task, final, self.exp.config.eval_timeout_s)
        # A deliberate evaluator change between experiments is not tampering: carry the run's own
        # before/after verdict over to the new hash, keeping a mismatch if the run changed the scorer.
        before, after = r.evaluator_hash, r.evaluator_hash_after
        if after in (None, before):
            after = after and eh
        elif after == eh:  # the scorer now on disk is the one this run left behind
            after = before
        r = r.model_copy(update={"score": sc.score, "evaluator_hash": eh, "evaluator_hash_after": after})
        (self._entry(j) / "record.json").write_text(r.model_dump_json(), encoding="utf-8")
        return r

    # ---- manifest -------------------------------------------------------------------------
    def manifest(self) -> Manifest:
        c = self.exp.config
        arts, confounds = diff_harnesses(self.exp.harness_a, self.exp.harness_b)
        registry = [
            TaskEntry(task_id=t.id, split=t.spec.split, group=t.spec.group, task_hash=t.task_hash) for t in self.exp.all_tasks
        ]
        return Manifest(
            experiment=c.experiment,
            baseline=c.baseline or f"harness_a ({self.exp.harness_a.config.name})",
            round=c.round,
            arms=Arms(baseline="A", candidate="B", replicate="A2"),
            harness={"A": self.exp.harness_a.hash, "B": self.exp.harness_b.hash},
            changed_artifacts=arts,
            confounds=confounds,
            proposal_groups=c.proposal_groups,
            selection=c.selection,
            oracle=self.oracle(),
            thresholds=c.thresholds,
            tasks=registry,
        )

    def oracle(self) -> Oracle:
        """Hidden tests and references are copied in only after the agent finishes; flag any trace that names them."""
        names = sorted({f.name for t in self.exp.all_tasks for f in t.evaluator_dir.rglob("*") if f.is_file()})
        names = [n for n in names if not n.endswith(".pyc")]
        protected = [r"(^|[/\\\s])evaluator[/\\]"] + [r"(^|[/\\\s'\"])" + re.escape(n) + r"\b" for n in names]
        return Oracle(outside_workspace=True, protected=protected)
