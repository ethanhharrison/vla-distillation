"""Generate candidate language instructions for a DROID trajectory."""

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

from .filter import ScoredInstruction, build_judge, score_instructions
from .pricing import RunCost, estimate_cost
from .prompts import INSTRUCTION_PROMPT, JUDGE_PROMPT, build_prompt, parse_instructions
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
    judge: bool = False
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_threshold: int = 3
    judge_prompt_template: str = JUDGE_PROMPT
    estimate_cost: bool = False

@dataclass
class StepInstructions:
    """Instructions proposed for a single trajectory step."""
    step: int
    instructions: list[str]
    raw_response: str = ""
    image_paths: dict[str, str] = field(default_factory=dict)
    scored: list[ScoredInstruction] = field(default_factory=list)
    judge_raw_response: str = ""

@dataclass
class GenerationResult:
    config: GenerationConfig
    trajectory_length: int
    metadata: dict
    steps: list[StepInstructions] = field(default_factory=list)

def generate_instructions(config: GenerationConfig, vlm: VLM | None = None, judge: VLM | None = None) -> GenerationResult:
    """Run the generation pipeline for one trajectory."""
    trajectory: Trajectory = load_trajectory(
        config.record_path, config.cameras, config.example_index
    )
    if vlm is None:
        vlm = build_vlm(config.provider, model=config.model)
    if config.judge and judge is None:
        judge = build_judge(config.judge_provider or config.provider, config.judge_model)

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
        proposed_instructions = parse_instructions(raw_response)

        scored: list[ScoredInstruction] = []
        judge_raw = ""
        if config.judge and judge is not None and proposed_instructions:
            accepted_instructions, scored, judge_raw = score_instructions(
                judge=judge,
                generation_prompt=prompt,
                images=images,
                instructions=proposed_instructions,
                step=step,
                total=trajectory.length,
                threshold=config.judge_threshold,
                template=config.judge_prompt_template,
            )
        else:
            accepted_instructions = proposed_instructions

        image_paths: dict[str, str] = {}
        if image_dir is not None:
            image_paths = save_frame(frame, step, image_dir)

        result.steps.append(
            StepInstructions(
                step=step,
                instructions=accepted_instructions,
                raw_response=raw_response,
                image_paths=image_paths,
                scored=scored,
                judge_raw_response=judge_raw,
            )
        )

        for instruction in accepted_instructions:
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

def build_run_cost(result: GenerationResult, vlm: VLM, judge: VLM | None = None) -> RunCost:
    """Estimate the run's cost from the accumulated VLM/judge token usage."""
    generation = estimate_cost(vlm.model, vlm.usage)
    judge_cost = estimate_cost(judge.model, judge.usage) if judge is not None else None
    return RunCost(
        generation=generation,
        judge=judge_cost,
        num_steps=len(result.steps),
    )

def _fmt_usd(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "unknown"

def cost_report_lines(run_cost: RunCost) -> list[str]:
    """Header lines describing the estimated cost (parsed back by the viewers)."""
    gen = run_cost.generation
    lines = [
        f"generation_input_tokens: {gen.usage.input_tokens}",
        f"generation_output_tokens: {gen.usage.output_tokens}",
        f"generation_cost_usd: {_fmt_usd(gen.total)}",
    ]
    if run_cost.judge is not None:
        judge = run_cost.judge
        lines.append(f"judge_input_tokens: {judge.usage.input_tokens}")
        lines.append(f"judge_output_tokens: {judge.usage.output_tokens}")
        lines.append(f"judge_cost_usd: {_fmt_usd(judge.total)}")
    lines.append(f"estimated_cost_total_usd: {_fmt_usd(run_cost.total)}")
    lines.append(f"estimated_cost_per_step_usd: {_fmt_usd(run_cost.per_step)}")
    return lines

def resolve_output_path(config: GenerationConfig, vlm: VLM) -> Path:
    if config.output_path is not None:
        return Path(config.output_path)
    stem = Path(config.record_path).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    filename = f"{stem}_{config.provider}_{timestamp}.txt"
    return DEFAULT_OUTPUT_DIR / filename

def write_txt(result: GenerationResult, vlm: VLM, output_path: Path, judge: VLM | None = None) -> Path:
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
    if config.judge:
        judge_provider = config.judge_provider or config.provider
        judge_model = judge.model if judge is not None else (config.judge_model or "?")
        lines.append(f"judge_provider: {judge_provider}")
        lines.append(f"judge_model: {judge_model}")
        lines.append(f"judge_threshold: {config.judge_threshold}")
    if config.estimate_cost:
        lines.extend(cost_report_lines(build_run_cost(result, vlm, judge)))
    if result.metadata:
        lines.append("metadata:")
        for key, value in result.metadata.items():
            lines.append(f"  {key}: {value}")
    lines.append("=" * 60)

    for step_result in result.steps:
        lines.append(f"[step {step_result.step}]")
        scores = {s.instruction: s.score for s in step_result.scored}
        for instruction in step_result.instructions:
            score = scores.get(instruction)
            suffix = f" | score: {score}" if score is not None else ""
            lines.append(f"  - {instruction}{suffix}")
        for scored in step_result.scored:
            if not scored.accepted:
                score = scored.score if scored.score is not None else "?"
                lines.append(f"  (rejected) {scored.instruction} | score: {score}")
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
        judge=args.judge,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_threshold=args.judge_threshold,
        estimate_cost=args.estimate_cost,
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
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Score each candidate with a VLM judge and drop low-scoring ones.",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="VLM provider for the judge (defaults to --provider). "
        f"Available: {', '.join(available_providers())}.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model name for the judge (defaults to the judge provider's default).",
    )
    parser.add_argument(
        "--judge-threshold",
        type=int,
        default=3,
        help="Minimum judge score (1-5) required to keep an instruction.",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Estimate the approximate USD cost from token usage (generation "
        "plus judge) and record it (total and per-step) in the run file.",
    )
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config_from_args(args)
    vlm = build_vlm(config.provider, model=config.model)
    judge = build_judge(config.judge_provider or config.provider, config.judge_model) if config.judge else None
    if judge is not None:
        print(f"Generating instructions with {vlm}, judging with {judge} ...")
    else:
        print(f"Generating instructions with {vlm} ...")
    result = generate_instructions(config, vlm=vlm, judge=judge)
    output_path = resolve_output_path(config, vlm)
    write_txt(result, vlm, output_path, judge=judge)
    num_accepted = sum(len(s.instructions) for s in result.steps)
    print(
        f"Wrote {num_accepted} instructions "
        f"across {len(result.steps)} steps to {output_path}"
    )
    if judge is not None:
        num_proposed = sum(len(s.scored) for s in result.steps)
        num_rejected = num_proposed - num_accepted
        print(
            f"Judge kept {num_accepted}/{num_proposed} candidates "
            f"(threshold {config.judge_threshold}); dropped {num_rejected}."
        )
    if config.estimate_cost:
        run_cost = build_run_cost(result, vlm, judge)
        print(
            f"Estimated cost: ${_fmt_usd(run_cost.total)} total "
            f"(${_fmt_usd(run_cost.per_step)} per step across {len(result.steps)} steps)"
        )
    if config.save_images:
        num_images = sum(len(s.image_paths) for s in result.steps)
        print(f"Saved {num_images} step images to {resolve_image_dir(config)}")

if __name__ == "__main__":
    main()
