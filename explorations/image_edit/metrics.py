"""Pixel metrics for the image-edit harness, and what each one is for.

Shared by `run_experiment.py`, `make_report.py` and `analyze.py`. The two video
harnesses each carry their own copy of `diff()`; here one module holds it,
because three scripts in this directory disagreeing about how a difference is
computed would be a silent measurement bug rather than a style problem.

Every number is a mean absolute pixel difference on 0-255 RGB, the same unit the
`explorations/cosmos3` and `explorations/dreamzero` tables use, so rows from all
three harnesses can sit side by side.

One deliberate difference from the video harnesses. There, everything being
compared is the same size, and `diff(a, b)` resamples `a` onto `b`. Here a
generated subgoal comes back far larger than the DROID source (1672x941 from a
320x180 frame, measured), so argument order would decide whether the reference
gets upsampled — and upsampling the reference invents detail that shows up as
difference the model never produced. `diff` therefore always compares on the
SMALLER of the two grids and is symmetric.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


def load_rgb(path: str | Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def load_rgb_bytes(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def _to_grid(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image

    if arr.shape[0] == height and arr.shape[1] == width:
        return arr
    return np.asarray(
        Image.fromarray(arr.astype(np.uint8)).resize((width, height), Image.Resampling.LANCZOS)
    )


def diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference, computed on whichever grid is smaller."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return float(np.abs(_to_grid(a, h, w).astype(float) - _to_grid(b, h, w).astype(float)).mean())


def mean(values: list[float]) -> float:
    """Mean that yields NaN rather than raising on an empty list."""
    return float(np.mean(values)) if values else float("nan")


def fmt(value: float | None, places: int = 1) -> str:
    """Format a metric, rendering missing/NaN as 'n/a' instead of a fake number."""
    if value is None or value != value:  # NaN
        return "n/a"
    return f"{value:.{places}f}"
