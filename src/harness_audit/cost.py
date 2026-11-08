"""Prices and the dry-run cost estimator.

Prices are USD per million tokens. Input/output prices for Claude Haiku 5.5 are
the published first-party rates for prompts up to 100K tokens; cache prices use
the usual multipliers (read 0.1x input, 5-minute write 1.25x input) and are an
assumption flagged in the README. The SDK's own ``total_cost_usd`` is preferred
for actual runs; these prices are used for estimates and as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Usage


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float


PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
    "claude-haiku-4-5": Price(input=1.00, output=5.00, cache_read=0.10, cache_write=1.25),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00),
}


def register_prices(table: dict[str, dict[str, float]]) -> None:
    """Add prices (USD per million tokens) for models without a built-in price; cache prices default to input."""
    for model, p in table.items():
        PRICES[model] = Price(
            input=p["input"],
            output=p["output"],
            cache_read=p.get("cache_read", p["input"]),
            cache_write=p.get("cache_write", p["input"]),
        )


def price_for(model: str) -> Price:
    if model in PRICES:
        return PRICES[model]
    raise KeyError(f"no price for model {model!r}; add it to harness_audit.cost.PRICES")


def cost_from_usage(model: str, u: Usage) -> float:
    p = price_for(model)
    return (
        (u.input_tokens or 0) * p.input
        + (u.output_tokens or 0) * p.output
        + (u.cache_read_input_tokens or 0) * p.cache_read
        + (u.cache_creation_input_tokens or 0) * p.cache_write
    ) / 1e6


@dataclass(frozen=True)
class RunShape:
    """Assumed shape of one agent run for estimation (tokens are per model call)."""

    base_context: int = 9000  # tool definitions + system prompt + task prompt
    growth_per_call: int = 900  # tool results and messages appended per call
    output_per_call_typical: int = 400
    output_per_call_worst: int = 1500
    calls_typical: int = 8


def estimate_run_usd(model: str, max_turns: int, extra_context: int = 0, shape: RunShape | None = None) -> tuple[float, float]:
    """Return ``(typical, worst_case)`` USD for one run.

    Typical: ``calls_typical`` calls with the prefix served from the prompt cache.
    Worst case: ``max_turns`` calls, nothing cached, long outputs.
    """
    shape = shape or RunShape()
    p = price_for(model)
    base = shape.base_context + extra_context
    n = shape.calls_typical
    reads = sum(base + i * shape.growth_per_call for i in range(n))
    writes = base + n * shape.growth_per_call
    typical = (reads * p.cache_read + writes * p.cache_write + n * shape.output_per_call_typical * p.output) / 1e6
    m = max(max_turns, 1)
    worst_in = sum(base + i * shape.growth_per_call for i in range(m))
    worst = (worst_in * p.input + m * shape.output_per_call_worst * p.output) / 1e6
    return typical, worst
