"""Batch-run frames-vs-video dense description comparisons across several clip
windows (and durations) of one trajectory, writing one self-contained batch
directory: one subfolder per window (frame run, video run, comparison.html)
plus an index linking them all.

Usage:
    python scripts/batch_compare_dense_descriptions.py \
        datasets/droid/success/success-00285.tfrecord \
        --window 60:2 --window 180:2 --window 300:2 --window 60:4 --window 300:4
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # to import the pipeline package

import compare_dense_description_modes as cmp_mod

from pipeline.dense_description.generate import DenseConfig, generate_dense_descriptions, write_txt
from pipeline.dense_description.generate_video import (
    DEFAULT_VIDEO_CAMERAS,
    VideoDenseConfig,
    generate_dense_video_descriptions,
    write_results,
)
from pipeline.language_instruction.vlm import build_vlm

DEFAULT_BATCH_DIR = PROJECT_ROOT / "outputs" / "dense_description_comparisons"


def parse_window(spec: str) -> tuple[int, float]:
    """Parse 'START' or 'START:SECONDS' into (start_step, clip_seconds)."""
    start_s, _, seconds_s = spec.partition(":")
    return int(start_s), (float(seconds_s) if seconds_s else 2.0)


def window_slug(start_step: int, clip_seconds: float) -> str:
    return f"s{start_step:04d}_{clip_seconds:g}s"


def run_window(
    record_path: Path,
    start_step: int,
    clip_seconds: float,
    *,
    provider: str,
    cameras: tuple[str, str],
    example_index: int,
    fps: float,
    ex_dir: Path,
) -> dict:
    """Run both modes on one clip window; return an index entry."""
    frame_vlm = build_vlm(provider)
    frame_config = DenseConfig(
        record_path=record_path,
        provider=provider,
        cameras=cameras,
        example_index=example_index,
        clip_seconds=clip_seconds,
        fps=fps,
        max_clips=1,
        start_step=start_step,
        save_images=True,
        image_dir=ex_dir / "frame_images",
        output_path=ex_dir / "frames.txt",
    )
    frame_result = generate_dense_descriptions(frame_config, vlm=frame_vlm)
    frame_txt = write_txt(frame_result, frame_vlm, frame_config.output_path)

    video_vlm = build_vlm(provider)
    video_config = VideoDenseConfig(
        record_path=record_path,
        provider=provider,
        video_cameras=cameras,
        example_index=example_index,
        clip_seconds=clip_seconds,
        fps=fps,
        max_clips=1,
        start_step=start_step,
        save_videos=1,
    )
    video_run_dir = ex_dir / "video"
    video_result = generate_dense_video_descriptions(video_config, video_run_dir, vlm=video_vlm)
    video_json_path = write_results(video_result, video_vlm, video_run_dir)

    frame_run = cmp_mod.parse_run(frame_txt)
    video_payload = json.loads(video_json_path.read_text())
    comparison_html = cmp_mod.render_html(frame_run, video_payload, video_run_dir, frame_txt, video_json_path)
    (ex_dir / "comparison.html").write_text(comparison_html)

    instruction = frame_result.clips[0].language_instruction if frame_result.clips else ""
    end_step = frame_result.clips[0].end_step if frame_result.clips else start_step
    return {
        "dir": ex_dir.name,
        "start_step": start_step,
        "end_step": end_step,
        "clip_seconds": clip_seconds,
        "instruction": instruction,
    }


def write_index(batch_dir: Path, record_path: Path, entries: list[dict]) -> Path:
    rows = "".join(
        f'<li><a href="{html.escape(e["dir"])}/comparison.html">{html.escape(e["dir"])}</a> — '
        f'steps {e["start_step"]}-{e["end_step"]} ({e["clip_seconds"]:g}s) — '
        f'“{html.escape(e["instruction"])}”</li>'
        for e in entries
    )
    out = batch_dir / "index.html"
    out.write_text(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Dense descriptions — frames vs video (batch)</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 800px;
         margin: 40px auto; padding: 0 20px; line-height: 1.6; }}
  li {{ margin-bottom: 10px; }}
</style></head><body>
  <h1>Dense descriptions: frames vs video</h1>
  <p>{html.escape(str(record_path))}</p>
  <ul>{rows}</ul>
</body></html>""")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", help="Path to a .tfrecord file.")
    parser.add_argument(
        "--window",
        action="append",
        required=True,
        metavar="START[:SECONDS]",
        help="A clip window to run both modes on, e.g. 180:2 or 60:4 (default 2s). Repeatable.",
    )
    parser.add_argument("--provider", default="gemini")
    parser.add_argument(
        "--cameras",
        nargs=2,
        metavar=("TOP", "BOTTOM"),
        default=list(DEFAULT_VIDEO_CAMERAS),
        help="The two cameras used by both modes (frame stills + stacked video).",
    )
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--output",
        default=None,
        help="Batch directory (defaults to outputs/dense_description_comparisons/<record>_<timestamp>).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cameras = tuple(args.cameras)
    windows = [parse_window(w) for w in args.window]
    record_path = Path(args.record)

    batch_dir = (
        Path(args.output)
        if args.output
        else DEFAULT_BATCH_DIR / f"{record_path.stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for start_step, clip_seconds in windows:
        name = window_slug(start_step, clip_seconds)
        print(f"=== {name} ===")
        entries.append(
            run_window(
                record_path,
                start_step,
                clip_seconds,
                provider=args.provider,
                cameras=cameras,
                example_index=args.example_index,
                fps=args.fps,
                ex_dir=batch_dir / name,
            )
        )

    index_path = write_index(batch_dir, record_path, entries)
    print(f"\nWrote batch of {len(entries)} comparisons to {batch_dir}")
    print(f"Open {index_path}")


if __name__ == "__main__":
    main()
