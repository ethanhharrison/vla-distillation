"""One-off: distribution of end-effector movement across a sample of DROID
shards, to sanity-check the staleness filter's threshold
(pipeline/dense_description/label_dataset.py's SMALL_MOVE_M) against real
data, before trusting it at scale.

No VLM/LLM calls - this only reads the (tiny) robot_state arrays TFRecord
already stores, via the same `robot_state_arrays`/`window_starts`/`is_stale`
the production labeler uses. Streams shards directly from GCS (confirmed:
tf.data.TFRecordDataset reads "gs://" paths with zero local disk footprint -
it's pure network streaming, not a download), so the cost here is time
(~1-2 min/shard) and shared network bandwidth, not disk space or API spend.

Usage:
    python scripts/measure_gripper_movement.py --num-shards 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dense_description.label_dataset import is_stale, robot_state_arrays, window_starts

BUCKET = "pranav-us-central1"
PREFIX = "datasets/droid/success/"
FPS = 15.0
CLIP_SECONDS = 4.0
STRIDE_SECONDS = 1.0
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "droid_staleness_stats"


def scan_shard(gcs_uri: str) -> list[tuple[float, bool]]:
    """(moved_meters, is_stale) for every candidate window in every episode
    of one shard."""
    clip_frames = int(round(CLIP_SECONDS * FPS))
    stride_frames = int(round(STRIDE_SECONDS * FPS))
    out = []
    for state in robot_state_arrays(gcs_uri):
        if state is None:
            continue
        gripper, xyz = state
        for start in window_starts(len(gripper), clip_frames, stride_frames):
            end = min(start + clip_frames - 1, len(gripper) - 1)
            moved = float(np.linalg.norm(xyz[end] - xyz[start]))
            out.append((moved, is_stale(gripper, xyz, start, end)))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-shards", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent shard streams (network-bound, not CPU-bound).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_shards = tf.io.gfile.listdir(f"gs://{BUCKET}/{PREFIX}")
    rng = random.Random(args.seed)
    sampled = rng.sample(all_shards, min(args.num_shards, len(all_shards)))
    print(f"{len(all_shards)} shards available, sampling {len(sampled)} (seed={args.seed})")

    results: list[tuple[float, bool]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(scan_shard, f"gs://{BUCKET}/{PREFIX}{name}"): name for name in sampled}
        for i, fut in enumerate(as_completed(futs), 1):
            name = futs[fut]
            windows = fut.result()
            results.extend(windows)
            n_stale = sum(1 for _, s in windows if s)
            elapsed = time.time() - t0
            print(f"[{i}/{len(sampled)}] {name}: {len(windows)} windows, {n_stale} stale "
                  f"({elapsed:.0f}s elapsed)")

    moved = np.array([m for m, _ in results])
    stale = np.array([s for _, s in results])
    summary = {
        "num_shards": len(sampled),
        "shard_names": sampled,
        "num_windows": len(results),
        "num_stale": int(stale.sum()),
        "stale_fraction": round(float(stale.mean()), 4),
        "moved_meters": {
            "mean": round(float(moved.mean()), 5),
            "median": round(float(np.median(moved)), 5),
            "std": round(float(moved.std()), 5),
            "min": round(float(moved.min()), 5),
            "max": round(float(moved.max()), 5),
            "percentiles": {
                str(p): round(float(np.percentile(moved, p)), 5) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
            },
        },
        "current_threshold_m": 0.03,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(moved * 100, bins=80, range=(0, min(50, float(moved.max() * 100))), color="#4c78a8")
    ax.axvline(3.0, color="#e45756", linestyle="--", label="current threshold (3 cm)")
    ax.set_xlabel("end-effector displacement over one window (cm)")
    ax.set_ylabel("count")
    ax.set_title(f"DROID gripper movement per {CLIP_SECONDS:g}s window "
                 f"({len(sampled)} shards, {len(results)} windows, {summary['stale_fraction']:.1%} stale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "histogram.png", dpi=150)
    print(f"\nWrote {output_dir / 'histogram.png'} and {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
