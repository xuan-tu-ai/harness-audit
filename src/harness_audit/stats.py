"""Statistics used by the claims: admission gate, paired bootstrap, permutation test.

All paired procedures operate on **per-task** differences (each task's seed-mean
under arm B minus its seed-mean under arm A). Seeds are never resampled on their
own: repeated rollouts reduce execution noise but do not add independent tasks
(Harness-RSI README §3.2, CC BY 4.0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class GateResult:
    """Outcome of the confidence-gated admission rule."""

    delta_hat: float
    epsilon: float
    m: int
    K: int
    T: int
    delta: float
    passed: bool

    @property
    def bound(self) -> float:
        return self.delta_hat + self.epsilon


def hoeffding_epsilon(m: int, K: int = 1, T: int = 1, delta: float = 0.05) -> float:
    """Return ``sqrt(2·ln(K·T/δ)/m)``, the gate's confidence width for ``m`` tasks."""
    if m <= 0:
        return math.inf
    return math.sqrt(2.0 * math.log(K * T / delta) / m)


def hoeffding_gate(loss_diffs: np.ndarray | list[float], K: int = 1, T: int = 1, delta: float = 0.05) -> GateResult:
    """Apply ``Δ̂ + sqrt(2·ln(K·T/δ)/m) < 0`` to per-task loss differences.

    ``loss_diffs[i]`` is ``loss(candidate) − loss(incumbent)`` on independent task ``i``,
    with losses in [0, 1] (so differences lie in [−1, 1]).
    """
    d = np.asarray(loss_diffs, dtype=float)
    m = int(d.size)
    if m == 0:
        return GateResult(math.nan, math.inf, 0, K, T, delta, False)
    if np.any(np.abs(d) > 1.0 + 1e-12):
        raise ValueError("loss differences must lie in [-1, 1]")
    eps = hoeffding_epsilon(m, K, T, delta)
    dh = float(d.mean())
    return GateResult(dh, eps, m, K, T, delta, bool(dh + eps < 0))


@dataclass(frozen=True)
class PairedEstimate:
    """Mean paired difference with a bootstrap interval and a permutation p-value."""

    mean: float
    ci_low: float
    ci_high: float
    level: float
    se: float
    p_value: float
    m: int
    exact_p: bool

    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "mean": round(self.mean, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "ci_level": self.level,
            "se_boot": round(self.se, 6),
            "p_value": round(self.p_value, 6),
            "m_tasks": self.m,
            "exact_p": self.exact_p,
        }


def paired_bootstrap(
    diffs: np.ndarray | list[float],
    level: float = 0.95,
    n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of per-task differences.

    Tasks are resampled with replacement; returns ``(ci_low, ci_high, se)``.
    """
    d = np.asarray(diffs, dtype=float)
    m = d.size
    if m == 0:
        return (math.nan, math.nan, math.nan)
    if m == 1:
        return (float(d[0]), float(d[0]), math.inf)
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, m, size=(n_boot, m))
    means = d[idx].mean(axis=1)
    alpha = 1.0 - level
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi), float(means.std(ddof=1)))


def sign_flip_test(
    diffs: np.ndarray | list[float],
    n_perm: int = 20_000,
    rng: np.random.Generator | None = None,
    exact_max_m: int = 16,
) -> tuple[float, bool]:
    """Two-sided paired permutation (sign-flip) test of mean difference = 0.

    Under the null each task's difference is symmetric around zero, so its sign
    is exchangeable. Exact enumeration for ``m <= exact_max_m``; Monte Carlo
    otherwise (with the +1 correction). Returns ``(p_value, exact)``.
    """
    d = np.asarray(diffs, dtype=float)
    m = d.size
    if m == 0:
        return (1.0, True)
    obs = abs(d.mean())
    tol = 1e-12
    if m <= exact_max_m:
        signs = np.array(list(product((1.0, -1.0), repeat=m)))
        stats = np.abs((signs * d).mean(axis=1))
        return (float(np.mean(stats >= obs - tol)), True)
    rng = rng or np.random.default_rng(0)
    signs = rng.choice((1.0, -1.0), size=(n_perm, m))
    stats = np.abs((signs * d).mean(axis=1))
    return (float((np.sum(stats >= obs - tol) + 1) / (n_perm + 1)), False)


def paired_estimate(
    diffs: np.ndarray | list[float],
    level: float = 0.95,
    n_boot: int = 10_000,
    n_perm: int = 20_000,
    seed: int = 0,
) -> PairedEstimate:
    """Bootstrap CI plus sign-flip p-value for per-task paired differences."""
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    lo, hi, se = paired_bootstrap(d, level=level, n_boot=n_boot, rng=rng)
    p, exact = sign_flip_test(d, n_perm=n_perm, rng=np.random.default_rng(seed + 1))
    mean = float(d.mean()) if d.size else math.nan
    return PairedEstimate(mean, lo, hi, level, se, p, int(d.size), exact)


def safe_ratio(num: float, den: float) -> float | None:
    """``num / den`` or ``None`` when the denominator is zero or missing."""
    if den is None or num is None or den == 0:
        return None
    return float(num) / float(den)
