"""Cost tracking with a hard per-run dollar ceiling.

The runner calls the tracker on every camera edit — paid or cached. The ceiling
is enforced *before* the paid call (`precheck`), so a bug or a runaway config can
never exceed the budget: it raises `BudgetExceeded` and the run aborts with
partial results written. Cached calls log `cost_usd: 0.0, cached: true` so a
rerun of the same config visibly costs $0.

Hosted image models bill by token, so the dollar figures are conservative
per-image estimates (see `backends.py` price tables), labelled `estimated`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


class BudgetExceeded(RuntimeError):
    """Raised (before a paid call) when the $ ceiling would be exceeded."""


@dataclass
class CostRecord:
    backend: str
    model: str
    cost_usd: float
    cached: bool
    ts: str
    estimated: bool = True
    example_id: str | None = None
    camera: str | None = None
    note: str | None = None


@dataclass
class CostTracker:
    ceiling_usd: float
    costs_path: Path | None = None
    spent_usd: float = 0.0
    records: list[CostRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.costs_path is not None:
            self.costs_path = Path(self.costs_path)
            self.costs_path.parent.mkdir(parents=True, exist_ok=True)

    def remaining(self) -> float:
        return self.ceiling_usd - self.spent_usd

    def precheck(self, est_cost: float, *, what: str = "edit") -> None:
        if self.spent_usd + est_cost > self.ceiling_usd + 1e-9:
            raise BudgetExceeded(
                f"{what}: est ${est_cost:.4f} would push spend "
                f"${self.spent_usd:.4f} -> ${self.spent_usd + est_cost:.4f} "
                f"past ceiling ${self.ceiling_usd:.2f}. Aborting."
            )

    def record(
        self,
        *,
        backend: str,
        model: str,
        cost_usd: float,
        cached: bool,
        estimated: bool = True,
        example_id: str | None = None,
        camera: str | None = None,
        note: str | None = None,
    ) -> CostRecord:
        cost = 0.0 if cached else round(cost_usd, 6)
        rec = CostRecord(
            backend=backend, model=model, cost_usd=cost, cached=cached,
            estimated=estimated, ts=datetime.now(timezone.utc).isoformat(),
            example_id=example_id, camera=camera, note=note,
        )
        self.spent_usd += cost
        self.records.append(rec)
        if self.costs_path is not None:
            with self.costs_path.open("a") as fh:
                fh.write(json.dumps(asdict(rec)) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return rec

    def summary(self) -> dict:
        paid = [r for r in self.records if not r.cached]
        by_backend: dict[str, float] = {}
        for r in paid:
            by_backend[r.backend] = round(by_backend.get(r.backend, 0.0) + r.cost_usd, 6)
        return {
            "ceiling_usd": self.ceiling_usd,
            "spent_usd": round(self.spent_usd, 6),
            "remaining_usd": round(self.remaining(), 6),
            "num_paid_calls": len(paid),
            "num_cached_calls": sum(1 for r in self.records if r.cached),
            "by_backend_usd": by_backend,
        }
