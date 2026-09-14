"""Generate dense change-of-state descriptions from stacked-view clip videos.

Same clip split as `pipeline.dense_description.generate`, but instead of
sending the VLM two still frames (start/end) per camera, this renders each
clip as a single video — two cameras vertically stacked, playing at the
trajectory's native fps — so the model sees every intermediate frame. Writes
one self-contained run directory (results.json + a couple of sample videos)
so it can be pulled to a local machine in one `scp -r`.

    .venv/bin/python -m pipeline.dense_description.generate_video \
        datasets/droid/success/success-00285.tfrecord \
        --provider gemini --max-clips 3 --save-videos 2
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pipeline.language_instruction.pricing import RunCost, estimate_cost
from pipeline.language_instruction.trajectory import load_trajectory
from pipeline.language_instruction.vlm import VLM, available_providers, build_vlm

from .generate import (
    DEFAULT_CLIP_SECONDS,
    DROID_FPS,
    clip_boundaries,
    resolve_language_instruction,
)
from .prompts import DENSE_VIDEO_PROMPT, build_dense_video_prompt, parse_description
from .imaging import DEFAULT_ROTATE_180_CAMERAS
from .video import render_clip_video

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dense_description_video"

DEFAULT_VIDEO_CAMERAS: tuple[str, ...] = ("shoulder_image_1", "shoulder_image_2", "wrist_image")
DEFAULT_SAVE_VIDEOS = 2


@dataclass
class VideoDenseConfig:
    record_path: Path
    provider: str = "gemini"
    model: str | None = None
    video_cameras: tuple[str, ...] = DEFAULT_VIDEO_CAMERAS
    rotate_180_cameras: frozenset[str] = DEFAULT_ROTATE_180_CAMERAS
    example_index: int = 0
    clip_seconds: float = DEFAULT_CLIP_SECONDS
    fps: float = DROID_FPS
    max_clips: int | None = None
    start_step: int = 0
    language_instruction: str | None = None
    save_videos: int = DEFAULT_SAVE_VIDEOS
    output_dir: Path | None = None
    estimate_cost: bool = False
    prompt_template: str = DENSE_VIDEO_PROMPT

    @property
    def clip_frames(self) -> int:
        frames = int(round(self.clip_seconds * self.fps))
        if frames < 2:
            raise ValueError(
                f"clip_seconds={self.clip_seconds} at fps={self.fps} yields "
                f"{frames} frame(s); need >= 2 so the video has motion"
            )
        return frames


@dataclass
class ClipVideoDescription:
    clip_index: int
    start_step: int
    end_step: int
    language_instruction: str
    description: str
    video_path: str | None = None
    raw_response: str = ""


@dataclass
class VideoDenseResult:
    config: VideoDenseConfig
    trajectory_length: int
    metadata: dict
    clips: list[ClipVideoDescription] = field(default_factory=list)


def generate_dense_video_descriptions(
    config: VideoDenseConfig,
    run_dir: Path,
    vlm: VLM | None = None,
) -> VideoDenseResult:
    """Run video-input dense-description generation over clips of one trajectory."""
    trajectory = load_trajectory(
        config.record_path, config.video_cameras, config.example_index
    )
    if vlm is None:
        vlm = build_vlm(config.provider, model=config.model)

    language = resolve_language_instruction(
        trajectory.metadata, config.language_instruction
    )
    bounds = clip_boundaries(trajectory.length, config.clip_frames, config.start_step)
    if config.max_clips is not None:
        bounds = bounds[: config.max_clips]

    result = VideoDenseResult(
        config=config,
        trajectory_length=trajectory.length,
        metadata=trajectory.metadata,
    )
    videos_dir = run_dir / "videos"

    for clip_index, (start_step, end_step) in enumerate(bounds):
        keep_video = clip_index < config.save_videos
        out_path = (
            videos_dir / f"clip{clip_index:04d}.mp4"
            if keep_video
            else Path(tempfile.mkstemp(suffix=".mp4")[1])
        )
        video_bytes = render_clip_video(
            trajectory, start_step, end_step, config.video_cameras, config.fps, out_path,
            rotate_180=config.rotate_180_cameras,
        )
        if not keep_video:
            out_path.unlink(missing_ok=True)

        prompt = build_dense_video_prompt(
            language_instruction=language,
            cameras=config.video_cameras,
            start_step=start_step,
            end_step=end_step,
            total=trajectory.length,
            clip_seconds=config.clip_seconds,
            fps=config.fps,
            template=config.prompt_template,
        )
        raw = vlm.generate_video(prompt, video_bytes)
        description = parse_description(raw)

        result.clips.append(
            ClipVideoDescription(
                clip_index=clip_index,
                start_step=start_step,
                end_step=end_step,
                language_instruction=language,
                description=description,
                video_path=f"videos/{out_path.name}" if keep_video else None,
                raw_response=raw,
            )
        )
    return result


def _fmt_usd(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "unknown"


def build_run_cost(result: VideoDenseResult, vlm: VLM) -> RunCost:
    generation = estimate_cost(vlm.model, vlm.usage)
    return RunCost(generation=generation, judge=None, num_steps=len(result.clips))


def resolve_run_dir(config: VideoDenseConfig) -> Path:
    if config.output_dir is not None:
        return Path(config.output_dir)
    stem = Path(config.record_path).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    return DEFAULT_OUTPUT_DIR / f"{stem}_{config.provider}_{timestamp}"


def write_results(result: VideoDenseResult, vlm: VLM, run_dir: Path) -> Path:
    """Write results.json into the run directory."""
    config = result.config
    payload = {
        "run": {
            "record": str(config.record_path),
            "provider": config.provider,
            "model": vlm.model,
            "video_cameras": list(config.video_cameras),
            "rotate_180_cameras": sorted(config.rotate_180_cameras),
            "clip_seconds": config.clip_seconds,
            "fps": config.fps,
            "clip_frames": config.clip_frames,
            "trajectory_length": result.trajectory_length,
            "num_clips": len(result.clips),
            "metadata": result.metadata,
        },
        "clips": [asdict(clip) for clip in result.clips],
    }
    if config.estimate_cost:
        run_cost = build_run_cost(result, vlm)
        gen = run_cost.generation
        payload["run"]["cost"] = {
            "generation_input_tokens": gen.usage.input_tokens,
            "generation_output_tokens": gen.usage.output_tokens,
            "generation_cost_usd": gen.total,
            "estimated_cost_total_usd": run_cost.total,
            "estimated_cost_per_clip_usd": run_cost.per_step,
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", help="Path to a .tfrecord file.")
    parser.add_argument(
        "--provider",
        default="gemini",
        help=f"VLM provider. Available: {', '.join(available_providers())}. "
        "Must implement generate_video (gemini, dummy).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for the provider (defaults to the provider's default).",
    )
    parser.add_argument(
        "--video-cameras",
        nargs="+",
        metavar="CAMERA",
        default=None,
        help=f"Cameras to stack top-to-bottom, in order. Default: {' '.join(DEFAULT_VIDEO_CAMERAS)}.",
    )
    parser.add_argument(
        "--rotate-180",
        nargs="*",
        metavar="CAMERA",
        default=None,
        help="Cameras to rotate 180 degrees before stacking. "
        f"Default: {' '.join(DEFAULT_ROTATE_180_CAMERAS)}.",
    )
    parser.add_argument(
        "--example-index",
        type=int,
        default=0,
        help="Which example within the tfrecord to use.",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=DEFAULT_CLIP_SECONDS,
        help="Duration of each clip in seconds (DROID default fps is 15).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DROID_FPS,
        help="Trajectory frame rate (also the encoded video's fps).",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Only describe the first N clips (useful for quick runs).",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Start clipping from this trajectory step instead of 0.",
    )
    parser.add_argument(
        "--language-instruction",
        default=None,
        help="Override the trajectory metadata language instruction.",
    )
    parser.add_argument(
        "--save-videos",
        type=int,
        default=DEFAULT_SAVE_VIDEOS,
        help="Keep the rendered mp4 for the first N clips (others are discarded after the VLM call).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Run output directory (defaults to outputs/dense_description_video/).",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Estimate approximate USD cost from token usage.",
    )
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> VideoDenseConfig:
    return VideoDenseConfig(
        record_path=Path(args.record),
        provider=args.provider,
        model=args.model,
        video_cameras=tuple(args.video_cameras) if args.video_cameras else DEFAULT_VIDEO_CAMERAS,
        rotate_180_cameras=frozenset(args.rotate_180) if args.rotate_180 is not None else DEFAULT_ROTATE_180_CAMERAS,
        example_index=args.example_index,
        clip_seconds=args.clip_seconds,
        fps=args.fps,
        max_clips=args.max_clips,
        start_step=args.start_step,
        language_instruction=args.language_instruction,
        save_videos=args.save_videos,
        output_dir=Path(args.output) if args.output else None,
        estimate_cost=args.estimate_cost,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config_from_args(args)
    vlm = build_vlm(config.provider, model=config.model)
    run_dir = resolve_run_dir(config)
    print(
        f"Generating video dense descriptions with {vlm} "
        f"(clips of {config.clip_seconds:g}s ≈ {config.clip_frames} frames, "
        f"cameras {'/'.join(config.video_cameras)} stacked top-to-bottom, "
        f"rotated 180: {', '.join(config.rotate_180_cameras) or 'none'}) ..."
    )
    result = generate_dense_video_descriptions(config, run_dir, vlm=vlm)
    out_path = write_results(result, vlm, run_dir)
    print(f"Wrote {len(result.clips)} clip descriptions to {out_path}")
    kept = sum(1 for c in result.clips if c.video_path)
    print(f"Saved {kept} sample video(s) under {run_dir / 'videos'}")
    if config.estimate_cost:
        run_cost = build_run_cost(result, vlm)
        print(
            f"Estimated cost: ${_fmt_usd(run_cost.total)} total "
            f"(${_fmt_usd(run_cost.per_step)} per clip across {len(result.clips)} clips)"
        )


if __name__ == "__main__":
    main()
