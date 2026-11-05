from __future__ import annotations

import math

import numpy as np
import pytest

from harness_audit.stats import (
    hoeffding_epsilon,
    hoeffding_gate,
    paired_bootstrap,
    paired_estimate,
    sign_flip_test,
)


def test_hoeffding_epsilon_matches_formula() -> None:
    assert hoeffding_epsilon(8, 1, 1, 0.05) == pytest.approx(math.sqrt(2 * math.log(20) / 8))
    assert hoeffding_epsilon(100, 5, 10, 0.1) == pytest.approx(math.sqrt(2 * math.log(500) / 100))
    assert hoeffding_epsilon(0) == math.inf


def test_gate_passes_only_when_bound_negative() -> None:
    m = 40
    eps = hoeffding_epsilon(m)
    assert hoeffding_gate(np.full(m, -(eps + 0.01))).passed
    assert not hoeffding_gate(np.full(m, -(eps - 0.01))).passed
    assert not hoeffding_gate([]).passed


def test_gate_counts_tasks_not_seeds() -> None:
    # Ten tasks: epsilon depends on m=10 regardless of how many seeds were averaged.
    g = hoeffding_gate(np.full(10, -0.5))
    assert g.m == 10
    assert g.epsilon == pytest.approx(hoeffding_epsilon(10))


def test_gate_rejects_out_of_range_differences() -> None:
    with pytest.raises(ValueError):
        hoeffding_gate([1.5])


def test_bootstrap_ci_contains_mean_and_is_deterministic() -> None:
    d = np.random.default_rng(0).normal(0.2, 0.1, 30)
    lo, hi, se = paired_bootstrap(d, rng=np.random.default_rng(1))
    lo2, hi2, _ = paired_bootstrap(d, rng=np.random.default_rng(1))
    assert lo < d.mean() < hi
    assert (lo, hi) == (lo2, hi2)
    assert se == pytest.approx(d.std(ddof=1) / math.sqrt(30), rel=0.15)


def test_sign_flip_exact_small_sample() -> None:
    # All five differences positive: only the all-positive and all-negative sign patterns
    # reach the observed |mean|, so p = 2/32.
    p, exact = sign_flip_test([1, 1, 1, 1, 1])
    assert exact and p == pytest.approx(2 / 32)
    p0, _ = sign_flip_test([0.0, 0.0, 0.0])
    assert p0 == 1.0


def test_sign_flip_monte_carlo_large_sample() -> None:
    d = np.random.default_rng(3).normal(0.0, 1.0, 40)
    p, exact = sign_flip_test(d, n_perm=5000)
    assert not exact and 0.0 < p <= 1.0


def test_paired_estimate_null_vs_effect() -> None:
    rng = np.random.default_rng(5)
    null = paired_estimate(rng.normal(0, 0.2, 30))
    eff = paired_estimate(rng.normal(0.3, 0.2, 30))
    assert not eff.ci_low <= 0 <= eff.ci_high and eff.p_value < 0.01
    assert null.ci_low <= 0 <= null.ci_high or null.p_value > 0.01
