"""Run a matrix of few-shot dense-description ablations across a few query
clips, writing one self-contained batch directory (one subfolder per
trial x query, each with its own report.html) plus an index linking them.

Named trials (override any via --trials to run a subset):
  baseline                 - current defaults (instruction shown, no contrast)
  no_query_instruction     - query's language instruction hidden from the model
  contrast_1.5             - all images contrast-boosted 1.5x before sending
  prompt_terse             - minimal-instructions prompt variant
  prompt_motion_emphasis   - instructions that push for motion-only sentences
  prompt_compare_explicit  - instructions that ask for an explicit start/end compare first

Usage:
    python scripts/batch_fewshot_ablations.py \
        datasets/droid/success/success-00285.tfrecord
    python scripts/batch_fewshot_ablations.py ... --trials baseline,contrast_1.5
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

import summarize_dense_description_fewshot as viz

from pipeline.dense_description.generate_fewshot import (
    EXAMPLES_YAML,
    FewshotConfig,
    generate,
    load_labeled_examples,
)
from pipeline.language_instruction.vlm import build_vlm

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dense_description_fewshot_ablations"

TRIALS: dict[str, dict] = {
    "baseline": {},
    "no_query_instruction": {"show_query_instruction": False},
    "contrast_1.5": {"contrast_factor": 1.5},
    "prompt_terse": {"prompt_template": "terse"},
    "prompt_motion_emphasis": {"prompt_template": "motion_emphasis"},
    "prompt_compare_explicit": {"prompt_template": "compare_explicit"},
}

TRIAL_BLURBS: dict[str, str] = {
    "baseline": "current defaults: instruction shown, no contrast boost, default prompt",
    "no_query_instruction": "the query clip's language instruction is withheld from the model (few-shot examples still show theirs)",
    "contrast_1.5": "every image (examples + query) gets a 1.5x contrast boost before being sent",
    "prompt_terse": "same examples, but the task/format instructions are cut to a bare skeleton",
    "prompt_motion_emphasis": "instructions explicitly forbid static-scene sentences, demand motion in every line",
    "prompt_compare_explicit": "instructions ask the model to explicitly compare start vs end per camera before answering",
}

# (query_label, example_index, start_step, clip_seconds)
DEFAULT_QUERIES: list[tuple[str, int, int, float]] = [
    ("2s", 12, 60, 2.0),
    ("4s", 13, 120, 4.0),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", help="Path to a .tfrecord file.")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument(
        "--trials", default=None,
        help=f"Comma-separated subset of trial names to run. Default: all ({', '.join(TRIALS)}).",
    )
    parser.add_argument(
        "--queries", default=None,
        help="Comma-separated EXAMPLE_INDEX:START_STEP:SECONDS specs to use as query clips "
        "(default: the 2s switch clip and the 4s wipe clip already in fewshot_examples.yaml).",
    )
    parser.add_argument("--output", default=None, help="Batch directory (defaults under outputs/dense_description_fewshot_ablations/).")
    return parser.parse_args(argv)


def resolve_queries(spec: str | None) -> list[tuple[str, int, int, float]]:
    if spec is None:
        return DEFAULT_QUERIES
    queries = []
    for chunk in spec.split(","):
        idx_s, start_s, seconds_s = chunk.split(":")
        queries.append((f"{seconds_s}s_ex{idx_s}", int(idx_s), int(start_s), float(seconds_s)))
    return queries


def run_one(
    record_path: Path,
    provider: str,
    trial_name: str,
    overrides: dict,
    example_index: int,
    start_step: int,
    clip_seconds: float,
    out_dir: Path,
    examples_by_id: dict[str, dict],
) -> dict:
    config = FewshotConfig(
        record_path=record_path,
        provider=provider,
        example_index=example_index,
        start_step=start_step,
        clip_seconds=clip_seconds,
        **overrides,
    )
    vlm = build_vlm(config.provider, model=config.model)
    result = generate(config, vlm=vlm)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "result.json"
    json_path.write_text(json.dumps(result, indent=2))
    html_path = out_dir / "report.html"
    html_path.write_text(viz.render_html(result, examples_by_id, json_path))

    return {
        "dir": out_dir.name,
        "trial": trial_name,
        "instruction": result["run"]["language_instruction"],
        "gripper_preview": result["shared"].get("gripper", "")[:140],
    }


def write_index(batch_dir: Path, record_path: Path, queries: list, entries: list[dict]) -> Path:
    query_labels = [q[0] for q in queries]
    rows_by_trial: dict[str, list[dict]] = {}
    for e in entries:
        rows_by_trial.setdefault(e["trial"], []).append(e)

    sections = []
    for trial_name, rows in rows_by_trial.items():
        cells = "".join(
            f'<td><a href="{html.escape(r["dir"])}/report.html">{html.escape(r["dir"])}</a>'
            f'<div class="preview">{html.escape(r["gripper_preview"])}…</div></td>'
            for r in rows
        )
        sections.append(f"""
        <tr>
          <th>{html.escape(trial_name)}<div class="blurb">{html.escape(TRIAL_BLURBS.get(trial_name, ''))}</div></th>
          {cells}
        </tr>
        """)

    header_cells = "".join(f"<th>{html.escape(q)}</th>" for q in query_labels)
    out = batch_dir / "index.html"
    out.write_text(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Few-shot dense-description ablations</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px;
         margin: 40px auto; padding: 0 20px; line-height: 1.5; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
  th, td {{ border: 1px solid #8883; padding: 10px; text-align: left; vertical-align: top; }}
  th {{ background: #8881; width: 220px; }}
  .blurb {{ font-weight: 400; font-size: 12px; color: #888; margin-top: 4px; }}
  .preview {{ font-size: 12px; color: #888; margin-top: 4px; }}
  a {{ font-weight: 600; }}
</style></head><body>
  <h1>Few-shot dense-description ablations</h1>
  <p>{html.escape(str(record_path))}</p>
  <table>
    <tr><th>trial</th>{header_cells}</tr>
    {''.join(sections)}
  </table>
</body></html>""")
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    record_path = Path(args.record)
    trial_names = args.trials.split(",") if args.trials else list(TRIALS)
    queries = resolve_queries(args.queries)

    batch_dir = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"{record_path.stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    examples_by_id = {ex["id"]: ex for ex in load_labeled_examples(EXAMPLES_YAML)}

    entries = []
    for trial_name in trial_names:
        overrides = TRIALS[trial_name]
        for query_label, example_index, start_step, clip_seconds in queries:
            name = f"{trial_name}_{query_label}"
            print(f"=== {name} ===")
            entries.append(
                run_one(
                    record_path, args.provider, trial_name, overrides,
                    example_index, start_step, clip_seconds,
                    batch_dir / name, examples_by_id,
                )
            )

    index_path = write_index(batch_dir, record_path, queries, entries)
    print(f"\nWrote {len(entries)} reports to {batch_dir}")
    print(f"Open {index_path}")


if __name__ == "__main__":
    main()
