"""Generate dense change-of-state descriptions for short trajectory clips.

Given a DROID `.tfrecord`, split the episode into fixed-duration clips (default
2s @ 15 fps → 30 frames). For each clip, send the start-frame and end-frame
camera views plus the episode language instruction to a VLM and ask for a dense
description of how the scene changes.

    .venv/bin/python -m pipeline.dense_description.generate \
        datasets/droid/success/success-00285.tfrecord \
        --provider gemini --save-images
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pipeline.language_instruction.pricing import RunCost, estimate_cost
from pipeline.language_instruction.trajectory import (
    DEFAULT_CAMERAS,
    Trajectory,
    load_trajectory,
)
from pipeline.language_instruction.vlm import VLM, available_providers, build_vlm

from .prompts import DENSE_DESCRIPTION_PROMPT, build_dense_prompt, parse_description

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is a project dependency
    yaml = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dense_descriptions"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "outputs" / "dense_description_images"

# DROID streams are recorded at 15 Hz (see scripts/prepare_subgoal_examples.py).
DROID_FPS = 15.0
DEFAULT_CLIP_SECONDS = 2.0

ORIGINAL_INSTRUCTION_KEYS = (
    "language_instruction1",
    "language_instruction2",
    "language_instruction3",
)


@dataclass
class DenseConfig:
    record_path: Path
    provider: str = "gemini"
    model: str | None = None
    cameras: tuple[str, ...] = DEFAULT_CAMERAS
    example_index: int = 0
    clip_seconds: float = DEFAULT_CLIP_SECONDS
    fps: float = DROID_FPS
    max_clips: int | None = None
    language_instruction: str | None = None
    output_path: Path | None = None
    save_images: bool = False
    image_dir: Path | None = None
    estimate_cost: bool = False
    prompt_template: str = DENSE_DESCRIPTION_PROMPT

    @property
    def clip_frames(self) -> int:
        frames = int(round(self.clip_seconds * self.fps))
        if frames < 2:
            raise ValueError(
                f"clip_seconds={self.clip_seconds} at fps={self.fps} yields "
                f"{frames} frame(s); need >= 2 so start and end differ"
            )
        return frames


@dataclass
class ClipDescription:
    """Dense description for one trajectory clip."""

    clip_index: int
    start_step: int
    end_step: int
    language_instruction: str
    description: str
    raw_response: str = ""
    start_image_paths: dict[str, str] = field(default_factory=dict)
    end_image_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class DenseResult:
    config: DenseConfig
    trajectory_length: int
    metadata: dict
    clips: list[ClipDescription] = field(default_factory=list)


def original_task_instructions(metadata: dict) -> list[str]:
    """Deduplicated original task instructions recorded in the trajectory."""
    instructions: list[str] = []
    for key in ORIGINAL_INSTRUCTION_KEYS:
        value = metadata.get(key)
        if value and value not in instructions:
            instructions.append(value)
    return instructions


def resolve_language_instruction(
    metadata: dict,
    override: str | None = None,
) -> str:
    """Pick the language instruction to condition dense descriptions on."""
    if override is not None and override.strip():
        return override.strip()
    originals = original_task_instructions(metadata)
    if originals:
        return originals[0]
    raise ValueError(
        "No language instruction found in trajectory metadata "
        f"(looked for {ORIGINAL_INSTRUCTION_KEYS}) and none was passed via "
        "--language-instruction."
    )


def clip_boundaries(length: int, clip_frames: int) -> list[tuple[int, int]]:
    """Non-overlapping (start, end) inclusive step pairs covering the trajectory.

    The final partial clip is kept when it still has at least 2 frames so the
    start and end views differ. Shorter leftovers are dropped.
    """
    if clip_frames < 2:
        raise ValueError(f"clip_frames must be >= 2, got {clip_frames}")
    if length < 2:
        return []

    bounds: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + clip_frames - 1, length - 1)
        if end <= start:
            break
        bounds.append((start, end))
        start += clip_frames
    return bounds


def ordered_clip_images(
    trajectory: Trajectory,
    start_step: int,
    end_step: int,
    cameras: tuple[str, ...],
) -> list[bytes]:
    """Start-frame cameras first, then end-frame cameras (same order)."""
    start = trajectory.frame(start_step, cameras)
    end = trajectory.frame(end_step, cameras)
    images: list[bytes] = []
    for cam in cameras:
        if cam in start:
            images.append(start[cam])
    for cam in cameras:
        if cam in end:
            images.append(end[cam])
    return images


def resolve_image_dir(config: DenseConfig) -> Path:
    if config.image_dir is not None:
        return Path(config.image_dir)
    return DEFAULT_IMAGE_DIR / Path(config.record_path).stem


def save_clip_frames(
    trajectory: Trajectory,
    *,
    clip_index: int,
    start_step: int,
    end_step: int,
    cameras: tuple[str, ...],
    image_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Save start/end JPEGs for a clip; return ({cam: path}, {cam: path})."""
    image_dir.mkdir(parents=True, exist_ok=True)
    start_paths: dict[str, str] = {}
    end_paths: dict[str, str] = {}
    for cam, jpeg in trajectory.frame(start_step, cameras).items():
        path = image_dir / f"clip{clip_index:04d}_start_step{start_step:04d}_{cam}.jpeg"
        path.write_bytes(jpeg)
        start_paths[cam] = str(path)
    for cam, jpeg in trajectory.frame(end_step, cameras).items():
        path = image_dir / f"clip{clip_index:04d}_end_step{end_step:04d}_{cam}.jpeg"
        path.write_bytes(jpeg)
        end_paths[cam] = str(path)
    return start_paths, end_paths


def generate_dense_descriptions(
    config: DenseConfig,
    vlm: VLM | None = None,
) -> DenseResult:
    """Run dense-description generation over clips of one trajectory."""
    trajectory = load_trajectory(
        config.record_path, config.cameras, config.example_index
    )
    if vlm is None:
        vlm = build_vlm(config.provider, model=config.model)

    language = resolve_language_instruction(
        trajectory.metadata, config.language_instruction
    )
    bounds = clip_boundaries(trajectory.length, config.clip_frames)
    if config.max_clips is not None:
        bounds = bounds[: config.max_clips]

    result = DenseResult(
        config=config,
        trajectory_length=trajectory.length,
        metadata=trajectory.metadata,
    )
    image_dir = resolve_image_dir(config) if config.save_images else None

    for clip_index, (start_step, end_step) in enumerate(bounds):
        prompt = build_dense_prompt(
            language_instruction=language,
            cameras=config.cameras,
            start_step=start_step,
            end_step=end_step,
            total=trajectory.length,
            clip_seconds=config.clip_seconds,
            fps=config.fps,
            template=config.prompt_template,
        )
        images = ordered_clip_images(
            trajectory, start_step, end_step, config.cameras
        )
        raw = vlm.generate(prompt, images)
        description = parse_description(raw)

        start_paths: dict[str, str] = {}
        end_paths: dict[str, str] = {}
        if image_dir is not None:
            start_paths, end_paths = save_clip_frames(
                trajectory,
                clip_index=clip_index,
                start_step=start_step,
                end_step=end_step,
                cameras=config.cameras,
                image_dir=image_dir,
            )

        result.clips.append(
            ClipDescription(
                clip_index=clip_index,
                start_step=start_step,
                end_step=end_step,
                language_instruction=language,
                description=description,
                raw_response=raw,
                start_image_paths=start_paths,
                end_image_paths=end_paths,
            )
        )
    return result


def _fmt_usd(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "unknown"


def build_run_cost(result: DenseResult, vlm: VLM) -> RunCost:
    generation = estimate_cost(vlm.model, vlm.usage)
    return RunCost(generation=generation, judge=None, num_steps=len(result.clips))


def resolve_output_path(config: DenseConfig, vlm: VLM) -> Path:
    if config.output_path is not None:
        return Path(config.output_path)
    stem = Path(config.record_path).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    filename = f"{stem}_{config.provider}_{timestamp}.txt"
    return DEFAULT_OUTPUT_DIR / filename


def write_txt(result: DenseResult, vlm: VLM, output_path: Path) -> Path:
    """Write a human-readable summary of the dense descriptions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = result.config
    language = (
        result.clips[0].language_instruction
        if result.clips
        else resolve_language_instruction(
            result.metadata, config.language_instruction
        )
    )
    lines: list[str] = [
        f"record: {config.record_path}",
        f"provider: {config.provider}",
        f"model: {vlm.model}",
        f"clip_seconds: {config.clip_seconds}",
        f"fps: {config.fps}",
        f"clip_frames: {config.clip_frames}",
        f"cameras: {', '.join(config.cameras)}",
        f"trajectory_length: {result.trajectory_length}",
        f"num_clips: {len(result.clips)}",
        f"language_instruction: {language}",
    ]
    if config.estimate_cost:
        run_cost = build_run_cost(result, vlm)
        gen = run_cost.generation
        lines.extend(
            [
                f"generation_input_tokens: {gen.usage.input_tokens}",
                f"generation_output_tokens: {gen.usage.output_tokens}",
                f"generation_cost_usd: {_fmt_usd(gen.total)}",
                f"estimated_cost_total_usd: {_fmt_usd(run_cost.total)}",
                f"estimated_cost_per_clip_usd: {_fmt_usd(run_cost.per_step)}",
            ]
        )
    if result.metadata:
        lines.append("metadata:")
        for key, value in result.metadata.items():
            lines.append(f"  {key}: {value}")
    lines.append("=" * 60)

    for clip in result.clips:
        lines.append(
            f"[clip {clip.clip_index}] steps {clip.start_step}-{clip.end_step}"
        )
        lines.append(f"  language: {clip.language_instruction}")
        lines.append(f"  description: {clip.description}")
        for camera, path in clip.start_image_paths.items():
            lines.append(f"  (image) start/{camera}: {path}")
        for camera, path in clip.end_image_paths.items():
            lines.append(f"  (image) end/{camera}: {path}")
        lines.append("")

    output_path.write_text("\n".join(lines))
    return output_path


def build_config_from_args(args: argparse.Namespace) -> DenseConfig:
    return DenseConfig(
        record_path=Path(args.record),
        provider=args.provider,
        model=args.model,
        cameras=tuple(args.cameras),
        example_index=args.example_index,
        clip_seconds=args.clip_seconds,
        fps=args.fps,
        max_clips=args.max_clips,
        language_instruction=args.language_instruction,
        output_path=Path(args.output) if args.output else None,
        save_images=args.save_images,
        image_dir=Path(args.image_dir) if args.image_dir else None,
        estimate_cost=args.estimate_cost,
    )


def load_config(path: str | Path) -> DenseConfig:
    """Parse a YAML dense-description config into a DenseConfig."""
    if yaml is None:
        raise ImportError("PyYAML is required to load --config files")
    config_path = Path(path).resolve()
    with config_path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Dense config must be a mapping, got {type(raw).__name__}")

    record = raw.get("record")
    if not record:
        raise ValueError("Config is missing required field 'record'")
    record_path = Path(str(record))
    if not record_path.is_absolute():
        record_path = (PROJECT_ROOT / record_path).resolve()

    cameras = raw.get("cameras")
    if cameras is None:
        camera_tuple = DEFAULT_CAMERAS
    elif isinstance(cameras, str):
        camera_tuple = (cameras,)
    else:
        camera_tuple = tuple(str(c) for c in cameras)

    output = raw.get("output")
    image_dir = raw.get("image_dir")
    return DenseConfig(
        record_path=record_path,
        provider=str(raw.get("provider", "gemini")),
        model=raw.get("model"),
        cameras=camera_tuple,
        example_index=int(raw.get("example_index", 0)),
        clip_seconds=float(raw.get("clip_seconds", DEFAULT_CLIP_SECONDS)),
        fps=float(raw.get("fps", DROID_FPS)),
        max_clips=int(raw["max_clips"]) if raw.get("max_clips") is not None else None,
        language_instruction=raw.get("language_instruction"),
        output_path=Path(str(output)) if output else None,
        save_images=bool(raw.get("save_images", False)),
        image_dir=Path(str(image_dir)) if image_dir else None,
        estimate_cost=bool(raw.get("estimate_cost", False)),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "record",
        nargs="?",
        default=None,
        help="Path to a .tfrecord file (optional when --config is set).",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="YAML config (see configs/dense_description/default.yaml).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help=f"VLM provider. Available: {', '.join(available_providers())}.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for the provider (defaults to the provider's default).",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Which camera image features to send to the VLM.",
    )
    parser.add_argument(
        "--example-index",
        type=int,
        default=None,
        help="Which example within the tfrecord to use.",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=None,
        help="Duration of each clip in seconds (DROID default fps is 15).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Assumed frame rate used to convert clip_seconds to frame count.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Only describe the first N clips (useful for quick runs).",
    )
    parser.add_argument(
        "--language-instruction",
        default=None,
        help="Override the trajectory metadata language instruction.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .txt path (defaults to outputs/dense_descriptions/).",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save start/end camera frames for each clip.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Where to save clip frames "
        "(defaults to outputs/dense_description_images/<record>/).",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Estimate approximate USD cost from token usage.",
    )
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> DenseConfig:
    """Merge optional YAML config with CLI overrides into a DenseConfig."""
    if args.config:
        config = load_config(args.config)
    elif args.record:
        config = DenseConfig(record_path=Path(args.record))
    else:
        raise SystemExit("Pass a tfrecord path or --config")

    if args.record:
        config.record_path = Path(args.record)
    if args.provider is not None:
        config.provider = args.provider
    if args.model is not None:
        config.model = args.model
    if args.cameras is not None:
        config.cameras = tuple(args.cameras)
    if args.example_index is not None:
        config.example_index = args.example_index
    if args.clip_seconds is not None:
        config.clip_seconds = args.clip_seconds
    if args.fps is not None:
        config.fps = args.fps
    if args.max_clips is not None:
        config.max_clips = args.max_clips
    if args.language_instruction is not None:
        config.language_instruction = args.language_instruction
    if args.output is not None:
        config.output_path = Path(args.output)
    if args.save_images:
        config.save_images = True
    if args.image_dir is not None:
        config.image_dir = Path(args.image_dir)
    if args.estimate_cost:
        config.estimate_cost = True
    return config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = resolve_config(args)
    vlm = build_vlm(config.provider, model=config.model)
    print(
        f"Generating dense descriptions with {vlm} "
        f"(clips of {config.clip_seconds:g}s ≈ {config.clip_frames} frames) ..."
    )
    result = generate_dense_descriptions(config, vlm=vlm)
    output_path = resolve_output_path(config, vlm)
    write_txt(result, vlm, output_path)
    print(
        f"Wrote {len(result.clips)} clip descriptions "
        f"across {result.trajectory_length} steps to {output_path}"
    )
    if config.estimate_cost:
        run_cost = build_run_cost(result, vlm)
        print(
            f"Estimated cost: ${_fmt_usd(run_cost.total)} total "
            f"(${_fmt_usd(run_cost.per_step)} per clip across {len(result.clips)} clips)"
        )
    if config.save_images:
        print(f"Saved clip frames under {resolve_image_dir(config)}")


if __name__ == "__main__":
    main()
