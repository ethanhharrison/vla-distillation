from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import yaml
from dotenv import load_dotenv

load_dotenv()

from .filter import build_judge
from .generate import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_OUTPUT_DIR,
    GenerationConfig,
    GenerationResult,
    StepInstructions,
    build_run_cost,
    generate_instructions,
    write_txt,
)
from .pricing import RunCost
from .prompts import DEFAULT_TEMPLATE, JUDGE_PROMPT, resolve_instruction_template
from .trajectory import DEFAULT_CAMERAS
from .vlm import VLM, build_vlm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CALL_OPTION_KEYS = frozenset(
    {
        "provider",
        "model",
        "step_interval",
        "num_instructions",
        "cameras",
        "max_steps",
        "prompt_template",
        "judge",
        "judge_provider",
        "judge_model",
        "judge_threshold",
        "name",
    }
)

@dataclass
class PipelineCallSpec:
    name: str
    generation: GenerationConfig

@dataclass
class PipelineConfig:
    config_path: Path
    record_path: Path
    calls: list[PipelineCallSpec]
    example_index: int = 0
    save_images: bool = False
    image_dir: Path | None = None
    estimate_cost: bool = False
    output_path: Path | None = None
    postprocess: dict[str, Any] = field(default_factory=dict)
    save_per_call: bool = True

@dataclass
class CallResult:
    name: str
    generation: GenerationResult
    vlm: VLM
    judge: VLM | None
    steps: list[StepInstructions]

@dataclass
class PipelineResult:
    config: PipelineConfig
    calls: list[CallResult]
    merged_by_step: dict[int, list[str]]
    provenance: dict[int, dict[str, list[str]]] = field(default_factory=dict)
    trajectory_length: int = 0
    metadata: dict = field(default_factory=dict)
    image_paths_by_step: dict[int, dict[str, str]] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Postprocess hook (stub; filter / paraphrase later)
# ---------------------------------------------------------------------------

class PostProcessor(Protocol):
    def process(self,
        instructions: list[str],
        *,
        step: int,
        call_name: str,
    ) -> list[str]:
        ...

class NoOpPostProcessor:
    def process(
        self,
        instructions: list[str],
        *,
        step: int,
        call_name: str,
    ) -> list[str]:
        return list(instructions)

def build_postprocessor(postprocess: dict[str, Any] | None) -> PostProcessor:
    """Build the postprocessor from the YAML `postprocess` block.

    Currently only supports a disabled / no-op path. When `enabled` is true,
    raise so callers know the real implementation is not wired yet.
    """
    cfg = postprocess or {}
    if cfg.get("enabled", False):
        raise NotImplementedError(
            "LLM filter/paraphrase postprocess is not implemented yet. "
            "Set postprocess.enabled: false (or omit postprocess) for now."
        )
    return NoOpPostProcessor()

# ---------------------------------------------------------------------------
# Load YAML
# ---------------------------------------------------------------------------

def as_path(value: Any, field_name: str) -> Path:
    if value is None or value == "":
        raise ValueError(f"Config is missing required field {field_name!r}")
    return Path(str(value))

def optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))

def resolve_cameras(value: Any) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_CAMERAS
    if isinstance(value, str):
        return (value,)
    return tuple(str(c) for c in value)

def generation_from_merged(
    merged: dict[str, Any],
    *,
    record_path: Path,
    example_index: int,
    save_images: bool,
    image_dir: Path | None,
    estimate_cost: bool,
) -> GenerationConfig:
    unknown = set(merged) - CALL_OPTION_KEYS
    if unknown:
        raise ValueError(f"Unknown call option(s): {', '.join(sorted(unknown))}. Allowed: {', '.join(sorted(CALL_OPTION_KEYS))}")

    template_name, template_text = resolve_instruction_template(str(merged.get("prompt_template", DEFAULT_TEMPLATE)))
    cameras = resolve_cameras(merged.get("cameras"))
    max_steps_raw = merged.get("max_steps")
    max_steps = int(max_steps_raw) if max_steps_raw is not None else None
    return GenerationConfig(
        record_path=record_path,
        provider=str(merged.get("provider", "gemini")),
        model=merged.get("model"),
        step_interval=int(merged.get("step_interval", 25)),
        num_instructions=int(merged.get("num_instructions", 3)),
        cameras=cameras,
        max_steps=max_steps,
        prompt_template=template_text,
        prompt_template_name=template_name,
        example_index=example_index,
        save_images=save_images,
        image_dir=image_dir,
        judge=bool(merged.get("judge", False)),
        judge_provider=merged.get("judge_provider"),
        judge_model=merged.get("judge_model"),
        judge_threshold=int(merged.get("judge_threshold", 3)),
        judge_prompt_template=JUDGE_PROMPT,
        estimate_cost=estimate_cost,
    )

def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Parse a YAML multi-call config into a PipelineConfig."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with config_path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Pipeline config must be a mapping, got {type(raw).__name__}")

    record_path = as_path(raw.get("record"), "record")
    if not record_path.is_absolute():
        record_path = (PROJECT_ROOT / record_path).resolve()

    example_index = int(raw.get("example_index", 0))
    save_images = bool(raw.get("save_images", False))
    image_dir = optional_path(raw.get("image_dir"))
    if image_dir is not None and not image_dir.is_absolute():
        image_dir = (PROJECT_ROOT / image_dir).resolve()
    estimate_cost = bool(raw.get("estimate_cost", False))
    output_path = optional_path(raw.get("output"))
    if output_path is not None and not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()

    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise TypeError("`defaults` must be a mapping")
    unknown_defaults = set(defaults) - CALL_OPTION_KEYS
    if unknown_defaults:
        raise ValueError(f"Unknown defaults option(s): {', '.join(sorted(unknown_defaults))}")

    calls_raw = raw.get("calls")
    if not calls_raw or not isinstance(calls_raw, list):
        raise ValueError("Config must include a non-empty `calls` list")

    postprocess = raw.get("postprocess") or {}
    if not isinstance(postprocess, dict):
        raise TypeError("`postprocess` must be a mapping")

    save_per_call = bool(raw.get("save_per_call", True))

    calls: list[PipelineCallSpec] = []
    used_names: set[str] = set()
    for idx, call_raw in enumerate(calls_raw):
        if not isinstance(call_raw, dict):
            raise TypeError(f"calls[{idx}] must be a mapping")
        merged = {**defaults, **call_raw}
        name = str(merged.get("name") or f"call_{idx}")
        if name in used_names:
            raise ValueError(f"Duplicate call name {name!r}")
        used_names.add(name)
        generation = generation_from_merged(
            merged,
            record_path=record_path,
            example_index=example_index,
            save_images=save_images,
            image_dir=image_dir,
            estimate_cost=estimate_cost,
        )
        # Per-call image dir under record folder with call name when saving images
        # and no explicit top-level image_dir — keeps call frames from clobbering.
        if save_images and image_dir is None:
            generation.image_dir = DEFAULT_IMAGE_DIR / record_path.stem / name
        elif save_images and image_dir is not None:
            generation.image_dir = image_dir / name

        calls.append(PipelineCallSpec(name=name, generation=generation))

    return PipelineConfig(
        config_path=config_path,
        record_path=record_path,
        calls=calls,
        example_index=example_index,
        save_images=save_images,
        image_dir=image_dir,
        estimate_cost=estimate_cost,
        output_path=output_path,
        postprocess=postprocess,
        save_per_call=save_per_call,
    )

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_call_results(calls: list[CallResult]) -> tuple[
    dict[int, list[str]],
    dict[int, dict[str, list[str]]],
    dict[int, dict[str, str]],
]:
    merged: dict[int, list[str]] = {}
    provenance: dict[int, dict[str, list[str]]] = {}
    images: dict[int, dict[str, str]] = {}

    for call in calls:
        for step_result in call.steps:
            step = step_result.step
            if step not in merged:
                merged[step] = []
                provenance[step] = {}
            if step not in images:
                images[step] = {}
            images[step].update(step_result.image_paths)

            for instruction in step_result.instructions:
                if instruction not in provenance[step]:
                    merged[step].append(instruction)
                    provenance[step][instruction] = [call.name]
                else:
                    if call.name not in provenance[step][instruction]:
                        provenance[step][instruction].append(call.name)
    return (
        dict(sorted(merged.items())),
        dict(sorted(provenance.items())),
        dict(sorted(images.items())),
    )

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_pipeline(
    config: PipelineConfig,
    postprocessor: PostProcessor | None = None,
) -> PipelineResult:
    if postprocessor is None:
        postprocessor = build_postprocessor(config.postprocess)

    call_results: list[CallResult] = []
    trajectory_length = 0
    metadata: dict = {}

    for call_spec in config.calls:
        gen_cfg = call_spec.generation
        print(
            f"[{call_spec.name}] Generating with provider={gen_cfg.provider!r} "
            f"template={gen_cfg.prompt_template_name!r} ..."
        )
        vlm = build_vlm(gen_cfg.provider, model=gen_cfg.model)
        judge = (
            build_judge(
                gen_cfg.judge_provider or gen_cfg.provider,
                gen_cfg.judge_model,
            )
            if gen_cfg.judge
            else None
        )
        if judge is not None:
            print(f"[{call_spec.name}] Judging with {judge} ...")

        generation = generate_instructions(gen_cfg, vlm=vlm, judge=judge)
        trajectory_length = generation.trajectory_length
        metadata = generation.metadata

        # Apply postprocess to each step's verified instructions.
        processed_steps: list[StepInstructions] = []
        for step in generation.steps:
            new_instructions = postprocessor.process(
                step.instructions,
                step=step.step,
                call_name=call_spec.name,
            )
            processed_steps.append(
                StepInstructions(
                    step=step.step,
                    instructions=new_instructions,
                    raw_response=step.raw_response,
                    image_paths=step.image_paths,
                    scored=step.scored,
                    judge_raw_response=step.judge_raw_response,
                )
            )

        # Patch generation.steps so per-call write_txt reflects postprocess.
        generation.steps = processed_steps
        call_results.append(
            CallResult(
                name=call_spec.name,
                generation=generation,
                vlm=vlm,
                judge=judge,
                steps=processed_steps,
            )
        )
        n_acc = sum(len(s.instructions) for s in processed_steps)
        print(
            f"[{call_spec.name}] {n_acc} instructions across "
            f"{len(processed_steps)} steps"
        )

    merged, provenance, image_paths = merge_call_results(call_results)
    return PipelineResult(
        config=config,
        calls=call_results,
        merged_by_step=merged,
        provenance=provenance,
        trajectory_length=trajectory_length,
        metadata=metadata,
        image_paths_by_step=image_paths,
    )

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def resolve_pipeline_output_dir(config: PipelineConfig) -> Path:
    if config.output_path is not None:
        path = Path(config.output_path)
        # If user passed a .txt file, use its parent for per-call dumps and the
        # file itself for the merged write; if a directory, write into it.
        if path.suffix.lower() == ".txt":
            return path.parent
        return path
    stem = config.record_path.stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    return DEFAULT_OUTPUT_DIR / f"{stem}_pipeline_{timestamp}"

def resolve_pipeline_merged_path(config: PipelineConfig, out_dir: Path) -> Path:
    if config.output_path is not None and Path(config.output_path).suffix.lower() == ".txt":
        return Path(config.output_path)
    return out_dir / "merged.txt"

def fmt_usd(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "unknown"

def write_pipeline_txt(result: PipelineResult, output_path: Path) -> Path:
    """Write the merged multi-call run summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = result.config
    lines: list[str] = [
        f"config: {config.config_path}",
        f"record: {config.record_path}",
        f"num_calls: {len(result.calls)}",
        f"calls: {', '.join(c.name for c in result.calls)}",
        f"trajectory_length: {result.trajectory_length}",
        f"save_images: {config.save_images}",
    ]

    for call in result.calls:
        gen = call.generation.config
        lines.append(f"call[{call.name}].provider: {gen.provider}")
        lines.append(f"call[{call.name}].model: {call.vlm.model}")
        lines.append(f"call[{call.name}].prompt_template: {gen.prompt_template_name}")
        lines.append(f"call[{call.name}].step_interval: {gen.step_interval}")
        lines.append(f"call[{call.name}].num_instructions: {gen.num_instructions}")
        if gen.max_steps is not None:
            lines.append(f"call[{call.name}].max_steps: {gen.max_steps}")
        if gen.judge:
            judge_provider = gen.judge_provider or gen.provider
            judge_model = (
                call.judge.model if call.judge is not None else (gen.judge_model or "?")
            )
            lines.append(f"call[{call.name}].judge_provider: {judge_provider}")
            lines.append(f"call[{call.name}].judge_model: {judge_model}")
            lines.append(f"call[{call.name}].judge_threshold: {gen.judge_threshold}")
        n_steps = len(call.steps)
        n_acc = sum(len(s.instructions) for s in call.steps)
        lines.append(f"call[{call.name}].steps: {n_steps}")
        lines.append(f"call[{call.name}].accepted_instructions: {n_acc}")

    if config.estimate_cost:
        total_run: list[RunCost] = []
        for call in result.calls:
            run_cost = build_run_cost(call.generation, call.vlm, call.judge)
            total_run.append(run_cost)
            lines.append(
                f"call[{call.name}].estimated_cost_usd: {fmt_usd(run_cost.total)}"
            )
        # Sum totals when all known; otherwise unknown
        totals = [c.total for c in total_run]
        if all(t is not None for t in totals):
            grand = sum(t or 0.0 for t in totals)
            lines.append(f"estimated_cost_total_usd: {fmt_usd(grand)}")
            n_merged_steps = len(result.merged_by_step) or 1
            lines.append(f"estimated_cost_per_merged_step_usd: {fmt_usd(grand / n_merged_steps)}")
        else:
            lines.append("estimated_cost_total_usd: unknown")

    if result.metadata:
        lines.append("metadata:")
        for key, value in result.metadata.items():
            lines.append(f"  {key}: {value}")
    lines.append("=" * 60)

    for step, instructions in result.merged_by_step.items():
        lines.append(f"[step {step}]")
        step_prov = result.provenance.get(step, {})
        for instruction in instructions:
            sources = step_prov.get(instruction, [])
            suffix = f" | from: {', '.join(sources)}" if sources else ""
            lines.append(f"  - {instruction}{suffix}")
        for camera, path in result.image_paths_by_step.get(step, {}).items():
            lines.append(f"  (image) {camera}: {path}")
        lines.append("")

    output_path.write_text("\n".join(lines))
    return output_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to a multi-call pipeline YAML config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output path from the YAML (directory or merged .txt).",
    )
    parser.add_argument(
        "--no-per-call",
        action="store_true",
        help="Skip writing per-call .txt dumps (merged output only).",
    )
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_pipeline_config(args.config)
    if args.output:
        config.output_path = Path(args.output)
    if args.no_per_call:
        config.save_per_call = False

    print(
        f"Running multi-call pipeline from {config.config_path} "
        f"({len(config.calls)} call(s) on {config.record_path}) ..."
    )
    result = run_pipeline(config)

    out_dir = resolve_pipeline_output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = resolve_pipeline_merged_path(config, out_dir)
    write_pipeline_txt(result, merged_path)

    if config.save_per_call:
        for call in result.calls:
            call_path = out_dir / f"call_{call.name}.txt"
            # Point generation config output so write_txt header is coherent.
            call.generation.config.output_path = call_path
            write_txt(call.generation, call.vlm, call_path, judge=call.judge)
            print(f"Wrote per-call dump: {call_path}")

    total_merged = sum(len(v) for v in result.merged_by_step.values())
    print(
        f"Merged {total_merged} unique instructions across "
        f"{len(result.merged_by_step)} steps -> {merged_path}"
    )
    if config.estimate_cost:
        print("(See merged header for per-call / total estimated costs.)")

if __name__ == "__main__":
    main()
