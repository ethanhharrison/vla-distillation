"""Few-shot dense-description generation from start/end stills only (no video).

Loads hand-written examples from fewshot_examples.yaml - only entries with a
written `shared`/`per_view` are used - and teaches them to the model as
(images, answer) chat turns before asking it to describe a new, unlabeled
clip in the same "Shared" + "Per-view" format your own manual labeling uses.

Usage:
    .venv/bin/python -m pipeline.dense_description.generate_fewshot \
        datasets/droid/success/success-00285.tfrecord \
        --example-index 9 --start-step 150 --clip-seconds 4 --provider gemini
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import yaml

from pipeline.language_instruction.trajectory import Trajectory, load_trajectory
from pipeline.language_instruction.vlm import VLM, ChatTurn, available_providers, build_vlm

from .generate import DEFAULT_CLIP_SECONDS, DROID_FPS
from .generate_video import DEFAULT_VIDEO_CAMERAS
from .imaging import DEFAULT_ROTATE_180_CAMERAS, maybe_rotate
from .prompts import (
    fewshot_clip_header,
    fewshot_instructions,
    format_fewshot_answer,
    parse_fewshot_response,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_YAML = PROJECT_ROOT / "pipeline" / "dense_description" / "fewshot_examples.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dense_description_fewshot_runs"


def load_labeled_examples(path: Path = EXAMPLES_YAML) -> list[dict]:
    """Entries from `path` with at least one non-empty shared/per_view field."""
    raw = yaml.safe_load(path.read_text()) or []
    labeled = []
    for ex in raw:
        shared = ex.get("shared") or {}
        per_view = ex.get("per_view") or {}
        if any(str(v).strip() for v in shared.values()) or any(str(v).strip() for v in per_view.values()):
            labeled.append(ex)
    return labeled


def clip_images(
    trajectory: Trajectory,
    start_step: int,
    end_step: int,
    cameras: tuple[str, ...],
    rotate_180: frozenset[str],
) -> list[bytes]:
    """Start-frame cameras first, then end-frame cameras (same order), rotated."""
    start = trajectory.frame(start_step, cameras)
    end = trajectory.frame(end_step, cameras)
    images = [maybe_rotate(start[c], c, rotate_180) for c in cameras if c in start]
    images += [maybe_rotate(end[c], c, rotate_180) for c in cameras if c in end]
    return images


def example_turns(ex: dict, cameras: tuple[str, ...], rotate_180: frozenset[str]) -> list[ChatTurn]:
    """The (user: images+context, model: answer) turn pair for one labeled example."""
    trajectory = load_trajectory(Path(ex["record"]), cameras, ex["example_index"])
    images = clip_images(trajectory, ex["start_step"], ex["end_step"], cameras, rotate_180)
    header = fewshot_clip_header(
        language_instruction=ex.get("language_instruction", ""),
        start_step=ex["start_step"],
        end_step=ex["end_step"],
        total=trajectory.length,
        clip_seconds=ex["clip_seconds"],
    )
    answer = format_fewshot_answer(ex.get("shared") or {}, ex.get("per_view") or {}, cameras)
    return [
        ChatTurn(role="user", text=header, images=images),
        ChatTurn(role="model", text=answer),
    ]


@dataclass
class FewshotConfig:
    record_path: Path
    provider: str = "gemini"
    model: str | None = None
    cameras: tuple[str, ...] = DEFAULT_VIDEO_CAMERAS
    rotate_180_cameras: frozenset[str] = DEFAULT_ROTATE_180_CAMERAS
    example_index: int = 0
    start_step: int = 0
    clip_seconds: float = DEFAULT_CLIP_SECONDS
    fps: float = DROID_FPS
    language_instruction: str | None = None
    examples_yaml: Path = EXAMPLES_YAML
    output_dir: Path | None = None


def generate(config: FewshotConfig, vlm: VLM | None = None) -> dict:
    if vlm is None:
        vlm = build_vlm(config.provider, model=config.model)

    labeled = load_labeled_examples(config.examples_yaml)
    if not labeled:
        raise ValueError(f"No labeled few-shot examples found in {config.examples_yaml}")

    turns: list[ChatTurn] = [ChatTurn(role="user", text=fewshot_instructions(config.cameras))]
    for ex in labeled:
        turns += example_turns(ex, config.cameras, config.rotate_180_cameras)

    trajectory = load_trajectory(config.record_path, config.cameras, config.example_index)
    clip_frames = int(round(config.clip_seconds * config.fps))
    end_step = min(config.start_step + clip_frames - 1, trajectory.length - 1)
    images = clip_images(trajectory, config.start_step, end_step, config.cameras, config.rotate_180_cameras)
    instruction = config.language_instruction or trajectory.metadata.get("language_instruction1", "")

    query_header = fewshot_clip_header(
        language_instruction=instruction,
        start_step=config.start_step,
        end_step=end_step,
        total=trajectory.length,
        clip_seconds=config.clip_seconds,
    ) + " Now write your answer in the format above."
    turns.append(ChatTurn(role="user", text=query_header, images=images))

    raw = vlm.generate_chat(turns)
    parsed = parse_fewshot_response(raw, config.cameras)

    return {
        "run": {
            "record": str(config.record_path),
            "provider": config.provider,
            "model": vlm.model,
            "cameras": list(config.cameras),
            "rotate_180_cameras": sorted(config.rotate_180_cameras),
            "example_index": config.example_index,
            "start_step": config.start_step,
            "end_step": end_step,
            "clip_seconds": config.clip_seconds,
            "trajectory_length": trajectory.length,
            "language_instruction": instruction,
            "num_fewshot_examples": len(labeled),
            "fewshot_example_ids": [ex["id"] for ex in labeled],
        },
        "raw_response": raw,
        "shared": parsed["shared"],
        "per_view": parsed["per_view"],
    }


def resolve_output_path(config: FewshotConfig) -> Path:
    out_dir = config.output_dir or DEFAULT_OUTPUT_DIR
    stem = Path(config.record_path).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    return out_dir / f"{stem}_ex{config.example_index}_s{config.start_step:04d}_{config.provider}_{timestamp}.json"


def print_result(result: dict) -> None:
    run = result["run"]
    print(f"\n{run['num_fewshot_examples']} few-shot example(s): {', '.join(run['fewshot_example_ids'])}")
    print(f"Query: example {run['example_index']}, steps {run['start_step']}-{run['end_step']} "
          f"({run['clip_seconds']:g}s) — \"{run['language_instruction']}\"\n")
    print("Shared:")
    for key in ("gripper", "scene_change", "terminal_gripper_state"):
        label = {"gripper": "Gripper", "scene_change": "Scene change", "terminal_gripper_state": "Terminal gripper state"}[key]
        print(f"  {label}: {result['shared'].get(key, '')}")
    print("\nPer-view:")
    for cam in run["cameras"]:
        print(f"  {cam}: {result['per_view'].get(cam, '')}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", help="Path to a .tfrecord file (the clip to describe).")
    parser.add_argument(
        "--provider",
        default="gemini",
        help=f"VLM provider. Available: {', '.join(available_providers())}. Must implement generate_chat (gemini, dummy).",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--cameras", nargs="+", metavar="CAMERA", default=None,
                         help=f"Default: {' '.join(DEFAULT_VIDEO_CAMERAS)}.")
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument("--fps", type=float, default=DROID_FPS)
    parser.add_argument("--language-instruction", default=None)
    parser.add_argument("--examples-yaml", default=None, help=f"Default: {EXAMPLES_YAML}")
    parser.add_argument("--output", default=None, help="Output .json path.")
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> FewshotConfig:
    return FewshotConfig(
        record_path=Path(args.record),
        provider=args.provider,
        model=args.model,
        cameras=tuple(args.cameras) if args.cameras else DEFAULT_VIDEO_CAMERAS,
        example_index=args.example_index,
        start_step=args.start_step,
        clip_seconds=args.clip_seconds,
        fps=args.fps,
        language_instruction=args.language_instruction,
        examples_yaml=Path(args.examples_yaml) if args.examples_yaml else EXAMPLES_YAML,
        output_dir=Path(args.output).parent if args.output else None,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config_from_args(args)
    vlm = build_vlm(config.provider, model=config.model)

    result = generate(config, vlm=vlm)
    print_result(result)

    out_path = Path(args.output) if args.output else resolve_output_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
