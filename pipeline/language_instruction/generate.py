"""Generate candidate language instructions for a DROID trajectory.

Walks a tfrecord's trajectory at a configurable step interval, sends the camera
frame(s) at each sampled step to a (swappable) VLM, and writes the proposed
instructions to a text file.

The in-memory result is returned as structured `StepInstructions`, so a future
verification pass can consume `(step, instructions)` pairs directly without
re-parsing the text output.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .prompts import INSTRUCTION_PROMPT, build_prompt, parse_instructions
from .trajectory import DEFAULT_CAMERAS, Trajectory, load_trajectory
from .vlm import VLM, available_providers, build_vlm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "language_instructions"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "outputs" / "language_instruction_images"

@dataclass
class GenerationConfig:
    record_path: Path
    provider: str = "gemini"
    model: str | None = None
    step_interval: int = 25
    num_instructions: int = 3
    cameras: tuple[str, ...] = DEFAULT_CAMERAS
    max_steps: int | None = None
    prompt_template: str = INSTRUCTION_PROMPT
    output_path: Path | None = None
    example_index: int = 0
    save_images: bool = False
    image_dir: Path | None = None

@dataclass
class StepInstructions:
    """Instructions proposed for a single trajectory step."""
    step: int
    instructions: list[str]
    raw_response: str = ""
    image_paths: dict[str, str] = field(default_factory=dict)

@dataclass
class GenerationResult:
    config: GenerationConfig
    trajectory_length: int
    metadata: dict
    steps: list[StepInstructions] = field(default_factory=list)

def generate_instructions(config: GenerationConfig, vlm: VLM | None = None) -> GenerationResult:
    """Run the generation pipeline for one trajectory."""
    trajectory: Trajectory = load_trajectory(
        config.record_path, config.cameras, config.example_index
    )
    if vlm is None:
        vlm = build_vlm(config.provider, model=config.model)

    result = GenerationResult(
        config=config,
        trajectory_length=trajectory.length,
        metadata=trajectory.metadata,
    )

    image_dir = resolve_image_dir(config) if config.save_images else None
    original_instructions = original_task_instructions(trajectory.metadata)
    previous_instructions: list[str] = []

    for step in trajectory.steps(config.step_interval, config.max_steps):
        frame = trajectory.frame(step, config.cameras)
        images = list(frame.values())
        prompt = build_prompt(
            step=step,
            total=trajectory.length,
            num_instructions=config.num_instructions,
            original_instructions=original_instructions,
            previous_instructions=previous_instructions,
            template=config.prompt_template,
        )
        raw_response = vlm.generate(prompt, images)
        step_instructions = parse_instructions(raw_response)

        image_paths: dict[str, str] = {}
        if image_dir is not None:
            image_paths = save_frame(frame, step, image_dir)

        result.steps.append(
            StepInstructions(
                step=step,
                instructions=step_instructions,
                raw_response=raw_response,
                image_paths=image_paths,
            )
        )

        for instruction in step_instructions:
            if instruction not in previous_instructions:
                previous_instructions.append(instruction)

    return result

ORIGINAL_INSTRUCTION_KEYS = (
    "language_instruction1",
    "language_instruction2",
    "language_instruction3",
)

def original_task_instructions(metadata: dict) -> list[str]:
    """Deduplicated original task instructions recorded in the trajectory."""
    instructions: list[str] = []
    for key in ORIGINAL_INSTRUCTION_KEYS:
        value = metadata.get(key)
        if value and value not in instructions:
            instructions.append(value)
    return instructions

def resolve_image_dir(config: GenerationConfig) -> Path:
    """Directory to store queried-step frames (per-record subfolder by default)."""
    if config.image_dir is not None:
        return Path(config.image_dir)
    return DEFAULT_IMAGE_DIR / Path(config.record_path).stem

def save_frame(frame: dict[str, bytes], step: int, image_dir: Path) -> dict[str, str]:
    """Save each camera's JPEG for a step; return {camera: path}."""
    image_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for camera, jpeg in frame.items():
        path = image_dir / f"step{step:04d}_{camera}.jpeg"
        path.write_bytes(jpeg)
        saved[camera] = str(path)
    return saved

def resolve_output_path(config: GenerationConfig, vlm: VLM) -> Path:
    if config.output_path is not None:
        return Path(config.output_path)
    stem = Path(config.record_path).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{stem}_{config.provider}_{timestamp}.txt"
    return DEFAULT_OUTPUT_DIR / filename

def write_txt(result: GenerationResult, vlm: VLM, output_path: Path) -> Path:
    """Write a human-readable summary of the generated instructions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = result.config
    lines: list[str] = [
        f"record: {config.record_path}",
        f"provider: {config.provider}",
        f"model: {vlm.model}",
        f"step_interval: {config.step_interval}",
        f"num_instructions: {config.num_instructions}",
        f"cameras: {', '.join(config.cameras)}",
        f"trajectory_length: {result.trajectory_length}",
    ]
    if result.metadata:
        lines.append("metadata:")
        for key, value in result.metadata.items():
            lines.append(f"  {key}: {value}")
    lines.append("=" * 60)

    for step_result in result.steps:
        lines.append(f"[step {step_result.step}]")
        for instruction in step_result.instructions:
            lines.append(f"  - {instruction}")
        for camera, path in step_result.image_paths.items():
            lines.append(f"  (image) {camera}: {path}")
        lines.append("")

    output_path.write_text("\n".join(lines))
    return output_path

def build_config_from_args(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        record_path=Path(args.record),
        provider=args.provider,
        model=args.model,
        step_interval=args.step_interval,
        num_instructions=args.num_instructions,
        cameras=tuple(args.cameras),
        max_steps=args.max_steps,
        output_path=Path(args.output) if args.output else None,
        example_index=args.example_index,
        save_images=args.save_images,
        image_dir=Path(args.image_dir) if args.image_dir else None,
    )

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=str, help="Path to a .tfrecord file.")
    parser.add_argument(
        "--provider",
        default="gemini",
        help=f"VLM provider. Available: {', '.join(available_providers())}.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for the provider (defaults to the provider's default).",
    )
    parser.add_argument(
        "--step-interval",
        type=int,
        default=25,
        help="Sample (and prompt) every N steps of the trajectory.",
    )
    parser.add_argument(
        "--num-instructions",
        type=int,
        default=3,
        help="How many candidate instructions to request per step.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        help="Which camera image features to send to the VLM.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Only consider steps up to this index (useful for quick runs).",
    )
    parser.add_argument(
        "--example-index",
        type=int,
        default=0,
        help="Which example within the tfrecord to use.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .txt path (defaults to outputs/language_instructions/).",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save the camera frame(s) at each queried step.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Where to save queried-step frames "
        "(defaults to outputs/language_instruction_images/<record>/).",
    )
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config_from_args(args)
    vlm = build_vlm(config.provider, model=config.model)
    print(f"Generating instructions with {vlm} ...")
    result = generate_instructions(config, vlm=vlm)
    output_path = resolve_output_path(config, vlm)
    write_txt(result, vlm, output_path)
    print(
        f"Wrote {sum(len(s.instructions) for s in result.steps)} instructions "
        f"across {len(result.steps)} steps to {output_path}"
    )
    if config.save_images:
        num_images = sum(len(s.image_paths) for s in result.steps)
        print(f"Saved {num_images} step images to {resolve_image_dir(config)}")

if __name__ == "__main__":
    main()
