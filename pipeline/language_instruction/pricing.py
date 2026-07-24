"""Token accounting and approximate USD cost estimation for VLM runs."""

from __future__ import annotations

from dataclasses import dataclass

# USD per 1 million tokens, as (input_price, output_price).
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-5.6-sol": (1.25, 10.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    # Google Gemini
    "gemini-3.6-flash": (0.30, 2.50),
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.0),
    # Local weights run on your own hardware — no per-token API charge.
    "dummy": (0.0, 0.0),
}


@dataclass
class Usage:
    """Cumulative token usage across one VLM's API calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.calls += 1

    def __add__(self, other):
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.calls + other.calls,
        )

@dataclass
class CostEstimate:
    """Approximate USD cost for a model's token usage."""

    model: str
    usage: Usage
    input_cost: float | None
    output_cost: float | None

    @property
    def total(self) -> float | None:
        if self.input_cost is None or self.output_cost is None:
            return None
        return self.input_cost + self.output_cost

def price_for_model(model: str | None) -> tuple[float, float] | None:
    """Look up `(input, output)` per-million prices for a model name."""
    if not model:
        return None
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    prefix_matches = [name for name in MODEL_PRICING if model.startswith(name)]
    if prefix_matches:
        return MODEL_PRICING[max(prefix_matches, key=len)]
    return None

def estimate_cost(model: str | None, usage: Usage) -> CostEstimate:
    """Estimate the USD cost of `usage` for `model` (unknown if unpriced)."""
    price = price_for_model(model)
    if price is None:
        return CostEstimate(model or "?", usage, None, None)
    input_per_million, output_per_million = price
    return CostEstimate(
        model=model or "?",
        usage=usage,
        input_cost=usage.input_tokens / 1_000_000 * input_per_million,
        output_cost=usage.output_tokens / 1_000_000 * output_per_million,
    )

@dataclass
class RunCost:
    """Combined generation + judge cost for one trajectory run."""

    generation: CostEstimate
    judge: CostEstimate | None
    num_steps: int

    @property
    def total(self) -> float | None:
        parts = [self.generation] + ([self.judge] if self.judge is not None else [])
        totals = [p.total for p in parts]
        if any(t is None for t in totals):
            return None
        return sum(t for t in totals if t is not None)

    @property
    def per_step(self) -> float | None:
        """Total cost divided by the number of instruction-generation steps."""
        total = self.total
        if total is None or self.num_steps <= 0:
            return None
        return total / self.num_steps
