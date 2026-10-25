"""Production dense-description labeling for one DROID shard.

Scans a partner-format .tfrecord, slides a fixed-length window across every
episode at a fixed stride, and generates a dense "Shared"/"Per-view"
description for each window via `generate_fewshot.generate()` - the same
few-shot describer used throughout prompt optimization, just called in a
loop instead of once.

Two things this deliberately omits, on purpose:
  - No judge/grading call. These labels feed two downstream training
    pipelines (pipeline/subgoal_image, and a foresight VLM trained to
    predict them from the current frame alone) rather than a graded
    benchmark, so the strict rubric-pass/fail framing used during prompt
    optimization doesn't obviously apply here. See
    outputs/dense_prompt_evals/optimization_log.md for the reasoning.
  - No staleness-skip accounting beyond a log line. A skipped window costs
    nothing and needs no record - recomputing the skip decision next run is
    free (it's a local numpy check), so there is nothing to resume.

Staleness filter (free, local, runs before any paid call): a window is
skipped if the gripper's state barely changed AND the end-effector moved
less than a few centimeters over the window - i.e. frame and frame-N-later
would look almost the same. Thresholds reuse the already-validated values
from scripts/build_dense_eval_sets.py's `classify_window`.

Safety: every paid call goes through a hard dollar ceiling (`CostTracker`,
reused from pipeline.subgoal_image.cost) that refuses to make a call that
would push spend past --budget-usd. The default ceiling is small ($2) on
purpose - raising it is an explicit, visible choice, never an accident.
Do NOT pass a budget anywhere near the team's ~$1000 full-DROID estimate
without explicit sign-off; nothing in this script authorizes that run.

Resuming: each window writes its own JSON file, named by episode id and
start step. A window already on disk is skipped before any cost is
incurred, so a killed run (e.g. the shared box's OOM watchdog) can just be
re-invoked with the same arguments.

Each window's output file is exactly what `generate_fewshot.generate()`
returns, so it's directly viewable with the existing visualizer:
    python scripts/summarize_dense_description_fewshot.py <path-to-one.json>

Usage:
    python -m pipeline.dense_description.label_dataset \
        datasets/droid/success/success-00285.tfrecord \
        --budget-usd 2.0 --limit-episodes 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import tensor_pb2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))  # to import the pipeline package

from pipeline.dense_description.generate_fewshot import DEFAULT_VIDEO_CAMERAS, FewshotConfig, build_vlm_for_config, generate
from pipeline.language_instruction.trajectory import load_trajectories
from pipeline.subgoal_image.cost import BudgetExceeded, CostTracker

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dense_description_labels"
FPS = 15.0

# Same thresholds scripts/build_dense_eval_sets.py already validated for
# telling "something happened" from "idle" via gripper/cartesian state.
GRIP_OPEN = 0.1
GRIP_CLOSED = 0.3
SMALL_MOVE_M = 0.03  # end-effector displacement below this = negligible

# A conservative per-call estimate for the budget precheck (observed average
# is ~$0.0117/call for gpt-6.1-sol at reasoning_effort=low; see
# outputs/dense_prompt_evals/train/06_v2_full_train/summary.json). The
# precheck only needs to be an upper bound - actual spend is logged exactly
# afterward via CostTracker.record().
EST_COST_PER_CALL_USD = 0.02


def _array(raw: bytes) -> np.ndarray:
    proto = tensor_pb2.TensorProto()
    proto.ParseFromString(raw)
    return tf.make_ndarray(proto)


def robot_state_arrays(record_path: str | Path) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Per-episode (gripper_position, cartesian_position[:, :3]) arrays, in
    the same order `load_trajectories` yields episodes. `None` for an
    episode missing these fields (e.g. not a DROID-robot_state-shaped
    record) - its windows are then never skipped as stale, since we can't
    tell. Accepts a local path or a "gs://" URI (pass it as a plain string,
    not wrapped in Path - see pipeline/common/gcs.py)."""
    out: list[tuple[np.ndarray, np.ndarray] | None] = []
    for raw in tf.data.TFRecordDataset([str(record_path)]):
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        f = example.features.feature
        try:
            gripper = _array(f["observation/robot_state/gripper_position"].bytes_list.value[0]).ravel()
            xyz = _array(f["observation/robot_state/cartesian_position"].bytes_list.value[0])[:, :3]
            out.append((gripper, xyz))
        except (KeyError, IndexError):
            out.append(None)
    return out


def is_stale(gripper: np.ndarray, xyz: np.ndarray, start: int, end: int) -> bool:
    """True if the robot barely moved between `start` and `end` - the
    gripper didn't meaningfully open/close and the end-effector moved less
    than SMALL_MOVE_M. Such a window would look almost the same at both
    ends, so describing it would just waste a call."""
    g = gripper[start : end + 1]
    moved = float(np.linalg.norm(xyz[end] - xyz[start]))
    grip_changed = g.max() - g.min() > (GRIP_CLOSED - GRIP_OPEN) / 2
    return not grip_changed and moved <= SMALL_MOVE_M


def window_starts(length: int, clip_frames: int, stride_frames: int) -> list[int]:
    if length < clip_frames:
        return []
    return list(range(0, length - clip_frames + 1, stride_frames))


def output_path(output_dir: Path, episode_id: str, start_step: int) -> Path:
    safe_id = episode_id.replace("/", "_")
    return output_dir / f"{safe_id}__s{start_step:06d}.json"


def label_shard(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_frames = int(round(args.clip_seconds * FPS))
    stride_frames = int(round(args.stride_seconds * FPS))

    tracker = CostTracker(ceiling_usd=args.budget_usd, costs_path=output_dir / "costs.jsonl")

    n_episodes = n_windows_total = n_stale = n_generated = n_skipped_existing = 0
    # args.record is kept as a plain string, never wrapped in Path(...) - see
    # pipeline/common/gcs.py: Path() silently corrupts a "gs://" URI.
    states = robot_state_arrays(args.record)

    for index, trajectory in enumerate(load_trajectories(args.record, DEFAULT_VIDEO_CAMERAS)):
        if args.episode_index is not None and index != args.episode_index:
            continue
        if args.limit_episodes is not None and n_episodes >= args.limit_episodes:
            break
        n_episodes += 1
        episode_id = trajectory.metadata.get("episode_id") or f"{Path(args.record).stem}_{index}"
        state = states[index] if index < len(states) else None

        for start in window_starts(trajectory.length, clip_frames, stride_frames):
            n_windows_total += 1
            end = min(start + clip_frames - 1, trajectory.length - 1)

            if not args.no_staleness_filter and state is not None and is_stale(state[0], state[1], start, end):
                n_stale += 1
                continue

            out_path = output_path(output_dir, episode_id, start)
            if out_path.exists():
                n_skipped_existing += 1
                continue

            try:
                tracker.precheck(EST_COST_PER_CALL_USD, what=f"{episode_id}@{start}")
            except BudgetExceeded as exc:
                print(f"\nBudget ceiling reached: {exc}")
                print_summary(tracker, n_episodes, n_windows_total, n_stale, n_generated, n_skipped_existing)
                return

            config = FewshotConfig(
                record_path=args.record, provider=args.provider, model=args.model,
                reasoning_effort=args.reasoning_effort, prompt_template=args.prompt_template,
                contrast_factor=args.contrast, example_index=index, start_step=start,
                clip_seconds=args.clip_seconds, fps=FPS,
            )
            # A fresh VLM per call, not reused across the loop: Usage (pricing.py)
            # is cumulative across a VLM instance's calls, so reusing one here
            # would make estimate_cost() read running-total tokens as if they
            # were this call's alone - every call after the first would report
            # an inflated, ever-growing "cost".
            result = generate(config, vlm=build_vlm_for_config(config))
            cost = result["run"]["cost"]["total_cost_usd"] or 0.0
            tracker.record(backend=args.provider, model=args.model, cost_usd=cost, cached=False,
                            estimated=False, example_id=episode_id, note=f"start={start}")
            out_path.write_text(json.dumps(result, indent=2))
            n_generated += 1
            print(f"{episode_id}@{start}: ${cost:.4f} (spent ${tracker.spent_usd:.4f} / ${args.budget_usd:.2f})")

    print_summary(tracker, n_episodes, n_windows_total, n_stale, n_generated, n_skipped_existing)


def print_summary(tracker: CostTracker, n_episodes: int, n_windows_total: int, n_stale: int,
                   n_generated: int, n_skipped_existing: int) -> None:
    print(f"\n{n_episodes} episodes, {n_windows_total} candidate windows: "
          f"{n_stale} stale (skipped free), {n_skipped_existing} already labeled, "
          f"{n_generated} newly generated.")
    print(json.dumps(tracker.summary(), indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", help="Path to one partner-format DROID .tfrecord shard.")
    parser.add_argument("--provider", default="openai",
                         help="VLM provider (default: openai). Use 'dummy' for a zero-cost smoke test of "
                              "windowing/staleness/resume logic.")
    parser.add_argument("--clip-seconds", type=float, default=4.0,
                         help="Window length in seconds (default: 4.0, matching the subgoal generator's frame/frame+4s pairing).")
    parser.add_argument("--stride-seconds", type=float, default=1.0,
                         help="Seconds between successive window starts (default: 1.0 -> ~20 windows on a "
                              "median ~22s DROID episode before staleness filtering; see the optimization log).")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--prompt-template", default="perview_fix",
                         help="Named template in prompts.py - 'perview_fix' is round 1/2's validated best for this model/effort.")
    parser.add_argument("--contrast", type=float, default=1.5)
    parser.add_argument("--budget-usd", type=float, default=2.0,
                         help="Hard dollar ceiling for this run (default: $2.00, intentionally small). "
                              "Raise this explicitly and deliberately - never pass anything near full-DROID "
                              "scale (~$1000s estimated) without separate, explicit sign-off.")
    parser.add_argument("--limit-episodes", type=int, default=None, help="Only the first N episodes of the shard.")
    parser.add_argument("--episode-index", type=int, default=None, help="Process only this one episode index (0-based).")
    parser.add_argument("--no-staleness-filter", action="store_true",
                         help="Generate for every window, including ones the staleness filter would skip. "
                              "For comparison/debugging - normal runs should leave this off.")
    parser.add_argument("--output-dir", default=None,
                         help=f"Default: {DEFAULT_OUTPUT_ROOT}/<shard-stem>. Use a different dir per "
                              "model/prompt/contrast config so label sets don't mix.")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = str(DEFAULT_OUTPUT_ROOT / Path(args.record).stem)
    return args


def main(argv: list[str] | None = None) -> None:
    label_shard(parse_args(argv))


if __name__ == "__main__":
    main()
