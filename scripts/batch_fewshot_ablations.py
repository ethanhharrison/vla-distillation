"""Run a matrix of few-shot dense-description ablations across a few query
clips, writing one self-contained batch directory (one subfolder per
trial x query, each with its own report.html) plus an index linking them.

Named trials (override any via --trials to run a subset):
  baseline                 - current defaults (instruction shown, no contrast, stills)
  no_query_instruction     - query's language instruction hidden from the model
  contrast_1.5             - all images contrast-boosted 1.5x before sending
  prompt_terse             - minimal-instructions prompt variant + hidden instruction
  prompt_motion_emphasis   - motion-only-sentences prompt variant + hidden instruction
  prompt_compare_explicit  - explicit start/end compare prompt variant + hidden instruction
  current_ideal            - hidden instruction + 1.5x contrast (4s query clips only)
  video                    - stacked-view clip video instead of stills, for
                             both few-shot examples and the query (native on
                             Gemini; a sampled frame burst on OpenAI, which
                             has no native video input)

Each result is graded by a VLM judge against the correctness rubric in
pipeline/dense_description/rubric.py (pass only if every criterion passes);
the verdict is saved as "judge" in result.json and shown in the reports.

Usage:
    python scripts/batch_fewshot_ablations.py \
        datasets/droid/success/success-00285.tfrecord \
        --provider openai --model gpt-6-astra --reasoning-effort xhigh
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
    build_vlm_for_config,
    generate,
    load_labeled_examples,
)
from pipeline.dense_description.rubric import (
    DEFAULT_JUDGE_EFFORT,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROVIDER,
    evidence_from_trajectory,
    judge_description,
)
from pipeline.language_instruction.trajectory import load_trajectory
from pipeline.language_instruction.vlm import build_vlm

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dense_description_fewshot_ablations"

TRIALS: dict[str, dict] = {
    "baseline": {},
    "no_query_instruction": {"show_query_instruction": False},
    "contrast_1.5": {"contrast_factor": 1.5},
    "prompt_terse": {"prompt_template": "terse", "show_query_instruction": False},
    "prompt_motion_emphasis": {"prompt_template": "motion_emphasis", "show_query_instruction": False},
    "prompt_compare_explicit": {"prompt_template": "compare_explicit", "show_query_instruction": False},
    "current_ideal": {"show_query_instruction": False, "contrast_factor": 1.5},
    "video": {"input_mode": "video"},
}

# Trials restricted to a subset of query labels (default: all queries below).
TRIAL_QUERY_SUBSET: dict[str, list[str]] = {
    "current_ideal": ["4s_a", "4s_b"],  # 4 seconds is part of this recipe
}

TRIAL_BLURBS: dict[str, str] = {
    "baseline": "current defaults: instruction shown, no contrast boost, default prompt, stills",
    "no_query_instruction": "the query clip's language instruction is withheld from the model (few-shot examples still show theirs)",
    "contrast_1.5": "every image (examples + query) gets a 1.5x contrast boost before being sent",
    "prompt_terse": "task/format instructions cut to a bare skeleton, query instruction hidden",
    "prompt_motion_emphasis": "instructions forbid static-scene sentences, demand motion in every line; query instruction hidden",
    "prompt_compare_explicit": "instructions ask for an explicit start/end compare before answering; query instruction hidden",
    "current_ideal": "combines the best-performing knobs so far: hidden instruction + 1.5x contrast, on 4s clips only",
    "video": "the clip's own stacked-view video (examples + query) instead of just start/end stills",
}

# Default query record: droid_100 (public RLDS, genuinely different scenes -
# different labs/kitchens/objects per episode), not success-00285 (a single
# lab's tfrecord shard, where every episode is the same physical kitchen).
DEFAULT_QUERY_RECORD = "datasets/droid/droid_100/1.0.0"

# (query_label, example_index, start_step, clip_seconds)
DEFAULT_QUERIES: list[tuple[str, int, int, float]] = [
    ("2s_a", 0, 30, 2.0),   # RAIL kitchen - "Put the marker in the pot"
    ("2s_b", 2, 30, 2.0),   # TRI kitchen - "Put one green sachet in the grey bowl."
    ("4s_a", 1, 60, 4.0),   # RPL kitchen - "Put the candy bar on the left side of the first shelf"
    ("4s_b", 6, 60, 4.0),   # IRIS lab - "Take the pen out of the bowl and place it on the table"
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "record", nargs="?", default=DEFAULT_QUERY_RECORD,
        help=f"Path to a .tfrecord file OR an RLDS dataset directory for the query clips "
        f"(default: {DEFAULT_QUERY_RECORD}). Few-shot examples always load from their own "
        "record in fewshot_examples.yaml, regardless of this.",
    )
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None, help="Passed through for providers that support it (e.g. OpenAI).")
    parser.add_argument(
        "--trials", default=None,
        help=f"Comma-separated subset of trial names to run. Default: all ({', '.join(TRIALS)}).",
    )
    parser.add_argument(
        "--queries", default=None,
        help="Comma-separated LABEL:EXAMPLE_INDEX:START_STEP:SECONDS specs to use as query clips "
        "(default: two 2s clips and two 4s clips).",
    )
    parser.add_argument("--output", default=None, help="Batch directory (defaults under outputs/dense_description_fewshot_ablations/).")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help="Grades each result against pipeline/dense_description/rubric.py.")
    parser.add_argument("--judge-effort", default=DEFAULT_JUDGE_EFFORT)
    parser.add_argument("--no-judge", action="store_true", help="Skip rubric grading (not recommended).")
    return parser.parse_args(argv)


def resolve_queries(spec: str | None) -> list[tuple[str, int, int, float]]:
    if spec is None:
        return DEFAULT_QUERIES
    queries = []
    for chunk in spec.split(","):
        label, idx_s, start_s, seconds_s = chunk.split(":")
        queries.append((label, int(idx_s), int(start_s), float(seconds_s)))
    return queries


def run_one(
    record_path: Path,
    provider: str,
    model: str | None,
    reasoning_effort: str | None,
    trial_name: str,
    overrides: dict,
    example_index: int,
    start_step: int,
    clip_seconds: float,
    out_dir: Path,
    examples_by_id: dict[str, dict],
    judge: tuple[str, str] | None = None,
) -> dict:
    config = FewshotConfig(
        record_path=record_path,
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
        example_index=example_index,
        start_step=start_step,
        clip_seconds=clip_seconds,
        **overrides,
    )
    vlm = build_vlm_for_config(config)
    result = generate(config, vlm=vlm)
    if judge:
        result["judge"] = judge_result(result, *judge)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "result.json"
    json_path.write_text(json.dumps(result, indent=2))
    html_path = out_dir / "report.html"
    html_path.write_text(viz.render_html(result, examples_by_id, json_path))

    verdict = result.get("judge")
    return {
        "verdict": None if verdict is None else ("PASS" if verdict["pass"] else "FAIL: " + ", ".join(verdict["failed_criteria"])),
        "dir": out_dir.name,
        "trial": trial_name,
        "query_label": None,  # filled by caller
        "instruction": result["run"]["language_instruction"],
        "gripper_preview": result["shared"].get("gripper", "")[:140],
        "cost": result["run"].get("cost", {}),
    }


def judge_result(result: dict, judge_model: str, judge_effort: str) -> dict:
    """Grade a generate() result against the correctness rubric; the judge
    also sees intermediate frames and the (possibly hidden) instruction."""
    run = result["run"]
    cameras = run["cameras"]
    stills = viz.load_clip_images(
        run["record"], run["example_index"], run["start_step"], run["end_step"],
        tuple(cameras), frozenset(run["rotate_180_cameras"]), run["contrast_factor"],
    )
    trajectory = load_trajectory(Path(run["record"]), tuple(cameras), run["example_index"])
    evidence = evidence_from_trajectory(trajectory, run["start_step"], run["end_step"], cameras, stills)
    judge_vlm = build_vlm(DEFAULT_JUDGE_PROVIDER, model=judge_model, reasoning_effort=judge_effort)
    verdict = judge_description(judge_vlm, result["raw_response"], run["language_instruction"], evidence)
    verdict["judge_effort"] = judge_effort
    return verdict


def write_index(batch_dir: Path, record_path: Path, query_labels: list[str], entries: list[dict]) -> Path:
    rows_by_trial: dict[str, dict[str, dict]] = {}
    total_cost = 0.0
    total_known = True
    for e in entries:
        rows_by_trial.setdefault(e["trial"], {})[e["query_label"]] = e
        cost = (e.get("cost") or {}).get("total_cost_usd")
        if cost is None:
            total_known = False
        else:
            total_cost += cost

    sections = []
    for trial_name in TRIALS:
        rows = rows_by_trial.get(trial_name)
        if not rows:
            continue
        row_cost = sum((r.get("cost") or {}).get("total_cost_usd") or 0.0 for r in rows.values())
        cells = []
        for label in query_labels:
            r = rows.get(label)
            if r is None:
                cells.append("<td class='skipped'>—</td>")
            else:
                cost = (r.get("cost") or {}).get("total_cost_usd")
                cost_str = f"${cost:.4f}" if cost is not None else "unknown"
                cells.append(
                    f'<td><a href="{html.escape(r["dir"])}/report.html">{html.escape(r["dir"])}</a>'
                    f'<div class="preview">{html.escape(r["gripper_preview"])}…</div>'
                    + (f'<div class="verdict {"ok" if r["verdict"] == "PASS" else "bad"}">{html.escape(r["verdict"])}</div>'
                       if r.get("verdict") else "")
                    + f'<div class="cost">{cost_str}</div></td>'
                )
        sections.append(f"""
        <tr>
          <th>{html.escape(trial_name)}<div class="blurb">{html.escape(TRIAL_BLURBS.get(trial_name, ''))}</div>
            <div class="cost">row total: ${row_cost:.4f}</div></th>
          {''.join(cells)}
        </tr>
        """)

    header_cells = "".join(f"<th>{html.escape(q)}</th>" for q in query_labels)
    total_line = f"${total_cost:.4f}" + ("" if total_known else " (some unknown, excluded)")
    out = batch_dir / "index.html"
    out.write_text(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Few-shot dense-description ablations</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1100px;
         margin: 40px auto; padding: 0 20px; line-height: 1.5; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
  th, td {{ border: 1px solid #8883; padding: 10px; text-align: left; vertical-align: top; }}
  th {{ background: #8881; width: 220px; }}
  td.skipped {{ color: #888; text-align: center; }}
  .blurb {{ font-weight: 400; font-size: 12px; color: #888; margin-top: 4px; }}
  .preview {{ font-size: 12px; color: #888; margin-top: 4px; }}
  .cost {{ font-size: 12px; color: #2e7d32; margin-top: 4px; font-weight: 600; }}
  .verdict {{ font-size: 12px; font-weight: 700; margin-top: 4px; }}
  .verdict.ok {{ color: #2e7d32; }} .verdict.bad {{ color: #c62828; }}
  a {{ font-weight: 600; }}
</style></head><body>
  <h1>Few-shot dense-description ablations</h1>
  <p>{html.escape(str(record_path))}</p>
  <p class="cost">Total cost: {total_line}</p>
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
    query_labels = [q[0] for q in queries]

    batch_dir = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"{record_path.stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    examples_by_id = {ex["id"]: ex for ex in load_labeled_examples(EXAMPLES_YAML)}

    entries = []
    for trial_name in trial_names:
        overrides = TRIALS[trial_name]
        allowed_labels = TRIAL_QUERY_SUBSET.get(trial_name)
        for query_label, example_index, start_step, clip_seconds in queries:
            if allowed_labels is not None and query_label not in allowed_labels:
                continue
            name = f"{trial_name}_{query_label}"
            print(f"=== {name} ===")
            entry = run_one(
                record_path, args.provider, args.model, args.reasoning_effort,
                trial_name, overrides, example_index, start_step, clip_seconds,
                batch_dir / name, examples_by_id,
                None if args.no_judge else (args.judge_model, args.judge_effort),
            )
            entry["query_label"] = query_label
            entries.append(entry)

    index_path = write_index(batch_dir, record_path, query_labels, entries)
    costs = [(e.get("cost") or {}).get("total_cost_usd") for e in entries]
    if all(c is not None for c in costs):
        print(f"\nTotal cost: ${sum(costs):.4f} across {len(entries)} calls")
    else:
        print(f"\nTotal cost: unknown for some calls (unpriced model) across {len(entries)} calls")
    print(f"Wrote {len(entries)} reports to {batch_dir}")
    print(f"Open {index_path}")


if __name__ == "__main__":
    main()
