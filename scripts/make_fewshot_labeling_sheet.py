"""Build a labeling sheet for hand-writing few-shot dense descriptions.

Renders a handful of clips from different trajectories (start/end stills +
the stacked shoulder+wrist video, same convention as the rest of
dense_description) into a self-contained HTML page for you to look at, and a
matching YAML skeleton for you to write descriptions into. Makes no VLM calls
- purely local rendering - so it's cheap to rerun with different windows if a
given clip isn't interesting enough to write about.

Usage:
    python scripts/make_fewshot_labeling_sheet.py \
        --record datasets/droid/success/success-00285.tfrecord \
        --clip 1:60:2 --clip 4:90:2 --clip 9:150:4 --clip 12:60:2 --clip 13:120:4
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # to import the pipeline package

import yaml

from pipeline.dense_description.generate import DROID_FPS
from pipeline.dense_description.generate_video import DEFAULT_VIDEO_CAMERAS
from pipeline.dense_description.imaging import DEFAULT_ROTATE_180_CAMERAS, maybe_rotate
from pipeline.dense_description.video import render_clip_video
from pipeline.language_instruction.trajectory import load_trajectory

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dense_description_fewshot"
EXAMPLES_YAML = PROJECT_ROOT / "pipeline" / "dense_description" / "fewshot_examples.yaml"


def parse_clip(spec: str) -> tuple[int, int, float]:
    """Parse 'EXAMPLE_INDEX:START_STEP:SECONDS' (seconds optional, default 2)."""
    parts = spec.split(":")
    example_index, start_step = int(parts[0]), int(parts[1])
    clip_seconds = float(parts[2]) if len(parts) > 2 else 2.0
    return example_index, start_step, clip_seconds


def _img_tag(jpeg: bytes, alt: str) -> str:
    import base64

    enc = base64.b64encode(jpeg).decode("ascii")
    return f'<img src="data:image/jpeg;base64,{enc}" alt="{html.escape(alt)}">'


def build_example(
    record_path: Path,
    example_index: int,
    start_step: int,
    clip_seconds: float,
    cameras: tuple[str, ...],
    fps: float,
    ex_id: str,
    media_dir: Path,
) -> dict:
    trajectory = load_trajectory(record_path, cameras, example_index)
    clip_frames = int(round(clip_seconds * fps))
    end_step = min(start_step + clip_frames - 1, trajectory.length - 1)
    if end_step - start_step < 1:
        raise ValueError(
            f"clip {ex_id}: start_step={start_step} leaves too few frames "
            f"in a trajectory of length {trajectory.length}"
        )
    instruction = trajectory.metadata.get("language_instruction1", "")

    media_dir.mkdir(parents=True, exist_ok=True)
    video_path = media_dir / f"{ex_id}.mp4"
    render_clip_video(trajectory, start_step, end_step, cameras, fps, video_path)

    start_frames = {
        cam: maybe_rotate(jpeg, cam) for cam, jpeg in trajectory.frame(start_step, cameras).items()
    }
    end_frames = {
        cam: maybe_rotate(jpeg, cam) for cam, jpeg in trajectory.frame(end_step, cameras).items()
    }
    for cam, jpeg in start_frames.items():
        (media_dir / f"{ex_id}_start_{cam}.jpeg").write_bytes(jpeg)
    for cam, jpeg in end_frames.items():
        (media_dir / f"{ex_id}_end_{cam}.jpeg").write_bytes(jpeg)

    return {
        "id": ex_id,
        "record": str(record_path),
        "example_index": example_index,
        "start_step": start_step,
        "end_step": end_step,
        "clip_seconds": clip_seconds,
        "fps": fps,
        "cameras": list(cameras),
        "language_instruction": instruction,
        "video_path": video_path,
        "start_frames": start_frames,
        "end_frames": end_frames,
    }


def render_html(examples: list[dict]) -> str:
    import base64

    sections = []
    for ex in examples:
        video_b64 = base64.b64encode(ex["video_path"].read_bytes()).decode("ascii")
        frame_figs = "".join(
            f'<figure>{_img_tag(ex["start_frames"][cam], f"start/{cam}")}<figcaption>start · {html.escape(cam)}</figcaption></figure>'
            for cam in ex["cameras"]
        ) + "".join(
            f'<figure>{_img_tag(ex["end_frames"][cam], f"end/{cam}")}<figcaption>end · {html.escape(cam)}</figcaption></figure>'
            for cam in ex["cameras"]
        )
        sections.append(f"""
        <section class="ex">
          <h2>{html.escape(ex['id'])} <span class="sub">example {ex['example_index']} ·
            steps {ex['start_step']}-{ex['end_step']} ({ex['clip_seconds']:g}s)</span></h2>
          <p class="instruction">"{html.escape(ex['language_instruction'])}"</p>
          <div class="body">
            <div class="col">
              <h4>Video ({' / '.join(ex['cameras'])} top-to-bottom, {ex['fps']:g} fps)</h4>
              <video controls loop muted playsinline src="data:video/mp4;base64,{video_b64}"></video>
            </div>
            <div class="col">
              <h4>Start / end stills</h4>
              <div class="frames">{frame_figs}</div>
            </div>
          </div>
          <p class="hint">Write this one under <code>{html.escape(ex['id'])}:</code> in
            <code>pipeline/dense_description/fewshot_examples.yaml</code>.</p>
        </section>
        """)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Few-shot dense-description labeling sheet</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
         max-width: 1100px; padding: 24px; line-height: 1.45; }}
  h1 {{ margin-bottom: 2px; }} .subtitle {{ color:#888; margin-top:0; }}
  .ex {{ border-top:2px solid #8884; padding-top:14px; margin-top:30px; }}
  .ex h2 {{ margin:4px 0; }} .sub {{ color:#888; font-weight:400; font-size:14px; }}
  .instruction {{ color:#8e24aa; margin:2px 0 12px; font-size:16px; }}
  .body {{ display:grid; grid-template-columns: minmax(240px, 380px) 1fr; gap:20px; align-items:start; }}
  .col h4 {{ margin:0 0 8px; color:#888; font-size:13px; text-transform:uppercase; letter-spacing:.03em; }}
  video {{ width:100%; border-radius:10px; display:block; background:#000; }}
  .frames {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .frames figure {{ margin:0; width:150px; }} .frames img {{ width:100%; border-radius:8px; display:block; }}
  .frames figcaption {{ font-size:11px; color:#888; text-align:center; }}
  .hint {{ font-size:13px; color:#888; margin-top:10px; }} code {{ background:#8882; padding:1px 5px; border-radius:4px; }}
  @media (max-width: 760px) {{ .body {{ grid-template-columns: 1fr; }} }}
</style></head><body>
  <h1>Few-shot dense-description labeling sheet</h1>
  <p class="subtitle">Watch each clip, then write a dense description of what actually happens
    into the matching entry in <code>pipeline/dense_description/fewshot_examples.yaml</code>.</p>
  {''.join(sections)}
  <footer class="subtitle">Rendered {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</footer>
</body></html>"""  # noqa: DTZ005


_EMPTY_SHARED = {"gripper": "", "scene_change": "", "terminal_gripper_state": ""}


def write_examples_yaml(examples: list[dict], path: Path) -> None:
    """Write/update the fewshot examples skeleton, preserving any hand-written
    shared/per_view content already on disk for ids that still exist."""
    existing_by_id: dict[str, dict] = {}
    if path.is_file():
        for entry in yaml.safe_load(path.read_text()) or []:
            existing_by_id[entry["id"]] = entry

    payload = []
    for ex in examples:
        prior = existing_by_id.get(ex["id"], {})
        payload.append(
            {
                "id": ex["id"],
                "record": ex["record"],
                "example_index": ex["example_index"],
                "start_step": ex["start_step"],
                "end_step": ex["end_step"],
                "clip_seconds": ex["clip_seconds"],
                "cameras": ex["cameras"],
                "language_instruction": ex["language_instruction"],
                "shared": prior.get("shared", dict(_EMPTY_SHARED)),
                "per_view": prior.get("per_view", {cam: "" for cam in ex["cameras"]}),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Few-shot dense-description examples, hand-written.\n"
        "# Schema: shared.gripper / shared.scene_change / shared.terminal_gripper_state,\n"
        "# and one per_view.<camera_name> line per camera in `cameras`.\n"
        "# Watch the matching clip in the labeling sheet HTML before writing one.\n\n"
    )
    path.write_text(header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--record",
        default="datasets/droid/success/success-00285.tfrecord",
        help="Path to a .tfrecord file (its distinct examples are different trajectories).",
    )
    parser.add_argument(
        "--clip",
        action="append",
        required=True,
        metavar="EXAMPLE_INDEX:START_STEP[:SECONDS]",
        help="A clip to include, e.g. 9:150:4 (seconds defaults to 2). Repeatable.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        metavar="CAMERA",
        default=list(DEFAULT_VIDEO_CAMERAS),
        help="Cameras to show, top-to-bottom, in order.",
    )
    parser.add_argument("--fps", type=float, default=DROID_FPS)
    parser.add_argument("--output", default=None, help="Batch directory (defaults under outputs/dense_description_fewshot/).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    record_path = Path(args.record)
    cameras = tuple(args.cameras)
    clips = [parse_clip(c) for c in args.clip]

    out_dir = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    media_dir = out_dir / "media"

    examples = [
        build_example(record_path, example_index, start_step, clip_seconds, cameras, args.fps, f"ex{i+1:02d}", media_dir)
        for i, (example_index, start_step, clip_seconds) in enumerate(clips)
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labeling.html").write_text(render_html(examples))
    write_examples_yaml(examples, EXAMPLES_YAML)

    print(f"Wrote labeling sheet ({len(examples)} clips) to {out_dir / 'labeling.html'}")
    print(f"Wrote skeleton to fill in to {EXAMPLES_YAML}")


if __name__ == "__main__":
    main()
