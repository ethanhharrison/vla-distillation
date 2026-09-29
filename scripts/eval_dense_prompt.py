"""Score a dense-description prompt on the train or test clip set.

Generates a description for every clip in the chosen split of
pipeline/dense_description/eval_sets/dense_eval_v1.yaml, grades each with the
correctness rubric judge, and writes per-clip reports plus an index and a
summary.json (pass rate, failures per rubric criterion, by event type and
scene, cost). Iterate on the prompt against `train`; run `test` only to check
a finished prompt, so the test score stays an honest estimate.

Defaults are the current best recipe: gpt-6-astra xhigh, start/end stills,
1.5x contrast, query instruction hidden.

Usage:
    # a named template from prompts.py
    python scripts/eval_dense_prompt.py --split train --prompt-template default
    # a prompt you're editing (must contain {camera_labels})
    python scripts/eval_dense_prompt.py --split train --prompt-file prompts/v2.txt --name v2
    # cheaper while iterating: fewer clips, cheaper judge
    python scripts/eval_dense_prompt.py --split train --prompt-file prompts/v2.txt --limit 10 --judge-effort low
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # to import the pipeline package

import yaml

import summarize_dense_description_fewshot as viz

from pipeline.dense_description.generate_fewshot import (
    EXAMPLES_YAML,
    FewshotConfig,
    build_vlm_for_config,
    generate,
    load_labeled_examples,
)
from pipeline.dense_description.prompts import FEWSHOT_INSTRUCTION_TEMPLATES
from pipeline.dense_description.rubric import (
    DEFAULT_JUDGE_EFFORT,
    DEFAULT_JUDGE_MODEL,
    RUBRIC,
    judge_fewshot_result,
)
from pipeline.language_instruction.pricing import Usage, estimate_cost

MANIFEST = PROJECT_ROOT / "pipeline" / "dense_description" / "eval_sets" / "dense_eval_v1.yaml"
DATA_DIR = PROJECT_ROOT / "datasets" / "dense_eval"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dense_prompt_evals"


def load_clips(split: str, manifest: Path = MANIFEST) -> list[dict]:
    clips = [c for c in yaml.safe_load(manifest.read_text())["clips"] if c["split"] == split]
    return sorted(clips, key=lambda c: c["example_index"])


def register_prompt_file(path: Path, name: str) -> str:
    """Make a prompt text file usable as a --prompt-template; returns its name."""
    text = path.read_text()
    if "{camera_labels}" not in text:
        raise ValueError(f"{path} must contain the {{camera_labels}} placeholder (see prompts.py)")
    FEWSHOT_INSTRUCTION_TEMPLATES[name] = text
    return name


def run_clip(clip: dict, args: argparse.Namespace, template: str, out_dir: Path, examples_by_id: dict) -> dict:
    config = FewshotConfig(
        record_path=DATA_DIR / f"{clip['split']}.tfrecord",
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        prompt_template=template,
        contrast_factor=args.contrast,
        show_query_instruction=args.show_instruction,
        example_index=clip["example_index"],
        start_step=clip["start_step"],
        clip_seconds=clip["clip_seconds"],
    )
    result = generate(config, vlm=build_vlm_for_config(config))
    result["clip"] = {k: clip[k] for k in ("id", "split", "scene", "org", "event", "source")}
    if not args.no_judge:
        result["judge"] = judge_fewshot_result(result, args.judge_model, args.judge_effort)
    clip_dir = out_dir / clip["id"]
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "result.json").write_text(json.dumps(result, indent=2))
    (clip_dir / "report.html").write_text(viz.render_html(result, examples_by_id, clip_dir / "result.json"))
    return result


def summarize(results: list[dict]) -> dict:
    judged = [r for r in results if r.get("judge")]
    fails = Counter(cid for r in judged for cid in r["judge"]["failed_criteria"])

    def rate(group_key: str) -> dict:
        out = {}
        for key in sorted({r["clip"][group_key] for r in judged}):
            grp = [r for r in judged if r["clip"][group_key] == key]
            out[key] = f'{sum(r["judge"]["pass"] for r in grp)}/{len(grp)}'
        return out

    gen_cost = sum((r["run"].get("cost") or {}).get("total_cost_usd") or 0.0 for r in results)
    judge_cost = 0.0
    for r in judged:
        u = r["judge"].get("judge_usage") or {}
        est = estimate_cost(r["judge"]["judge_model"], Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))
        judge_cost += est.total or 0.0
    return {
        "clips": len(results),
        "judged": len(judged),
        "pass": sum(r["judge"]["pass"] for r in judged),
        "pass_rate": round(sum(r["judge"]["pass"] for r in judged) / len(judged), 3) if judged else None,
        "failures_by_criterion": {c.id: fails.get(c.id, 0) for c in RUBRIC},
        "pass_by_event": rate("event"),
        "pass_by_scene": rate("scene"),
        "generation_cost_usd": round(gen_cost, 4),
        "judge_cost_usd_approx": round(judge_cost, 4),
    }


def write_index(out_dir: Path, results: list[dict], summary: dict, meta: dict) -> Path:
    rows = []
    for r in sorted(results, key=lambda r: r["clip"]["id"]):
        v = r.get("judge")
        status = "—" if v is None else ("PASS" if v["pass"] else "FAIL")
        cls = "" if v is None else ("ok" if v["pass"] else "bad")
        why = "" if not v or v["pass"] else "<br>".join(
            f'<b>{html.escape(cid)}</b>: {html.escape(v["criteria"][cid]["reason"])}' for cid in v["failed_criteria"]
        )
        rows.append(
            f'<tr class="{cls}"><td><a href="{r["clip"]["id"]}/report.html">{r["clip"]["id"]}</a></td>'
            f'<td>{html.escape(r["clip"]["scene"])}</td><td>{r["clip"]["event"]}</td>'
            f'<td>{html.escape(r["run"]["language_instruction"])}</td><td class="status">{status}</td>'
            f'<td class="why">{why}</td></tr>'
        )
    crit = "".join(f"<li>{k}: {n}</li>" for k, n in summary["failures_by_criterion"].items())
    meta_rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in meta.items())
    out = out_dir / "index.html"
    out.write_text(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Dense prompt eval</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1300px; margin: 32px auto;
         padding: 0 16px; line-height: 1.45; font-size: 14px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #8883; padding: 6px 8px; text-align: left; vertical-align: top; }}
  th {{ background: #8881; }} tr.ok td.status {{ color: #2e7d32; font-weight: 700; }}
  tr.bad td.status {{ color: #c62828; font-weight: 700; }} tr.bad {{ background: #c6282810; }}
  td.why {{ font-size: 12px; }} .big {{ font-size: 28px; font-weight: 700; }}
</style></head><body>
  <h1>Dense prompt eval — {html.escape(meta["split"])} split</h1>
  <p class="big">{summary["pass"]}/{summary["judged"]} pass</p>
  <table>{meta_rows}</table>
  <p><b>Failures by criterion:</b></p><ul>{crit}</ul>
  <p><b>Pass by event:</b> {html.escape(json.dumps(summary["pass_by_event"]))}<br>
     <b>Pass by scene:</b> {html.escape(json.dumps(summary["pass_by_scene"]))}</p>
  <table><tr><th>clip</th><th>scene</th><th>event</th><th>instruction (hidden from model)</th><th>verdict</th><th>why</th></tr>
  {"".join(rows)}</table>
</body></html>""")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt-template", default="default",
                        help=f"Named template in prompts.py: {', '.join(FEWSHOT_INSTRUCTION_TEMPLATES)}.")
    prompt.add_argument("--prompt-file", default=None, help="Text file with the instructions (must contain {camera_labels}).")
    parser.add_argument("--name", default=None, help="Label for this prompt version (used in the output dir name).")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--contrast", type=float, default=1.5)
    parser.add_argument("--show-instruction", action="store_true", help="Show the query's instruction (default: hidden).")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-effort", default=DEFAULT_JUDGE_EFFORT)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N clips of the split.")
    parser.add_argument("--clips", default=None, help="Comma-separated clip ids to run (e.g. to re-check failures).")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for split in ("train", "test"):
        if not (DATA_DIR / f"{split}.tfrecord").exists():
            sys.exit(f"Missing {DATA_DIR / f'{split}.tfrecord'} - run: python scripts/build_dense_eval_sets.py extract")
    if args.prompt_file:
        name = args.name or Path(args.prompt_file).stem
        template = register_prompt_file(Path(args.prompt_file), name)
    else:
        name = args.name or args.prompt_template
        template = args.prompt_template

    clips = load_clips(args.split)
    if args.clips:
        wanted = set(args.clips.split(","))
        clips = [c for c in clips if c["id"] in wanted]
    if args.limit:
        clips = clips[: args.limit]

    out_dir = Path(args.output) if args.output else (
        DEFAULT_OUTPUT_DIR / f"{args.split}_{name}_{args.model}_{args.reasoning_effort}_{datetime.now():%Y%m%d-%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(FEWSHOT_INSTRUCTION_TEMPLATES[template])
    examples_by_id = {ex["id"]: ex for ex in load_labeled_examples(EXAMPLES_YAML)}

    results = []
    print(f"{len(clips)} {args.split} clips -> {out_dir}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_clip, c, args, template, out_dir, examples_by_id): c["id"] for c in clips}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                v = r.get("judge")
                print(f'{futures[fut]}: {"—" if v is None else ("PASS" if v["pass"] else "FAIL " + ",".join(v["failed_criteria"]))}',
                      flush=True)
                results.append(r)
            except Exception as ex:  # noqa: BLE001 - keep going; one bad clip shouldn't sink the run
                print(f"{futures[fut]}: ERROR {ex}", flush=True)

    summary = summarize(results)
    meta = {"split": args.split, "prompt": name, "model": args.model, "reasoning_effort": args.reasoning_effort,
            "contrast": args.contrast, "instruction shown": args.show_instruction,
            "judge": "off" if args.no_judge else f"{args.judge_model} {args.judge_effort}",
            "generation cost": f'${summary["generation_cost_usd"]:.2f}',
            "judge cost (approx)": f'${summary["judge_cost_usd_approx"]:.2f}'}
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=2))
    index = write_index(out_dir, results, summary, meta)
    print(json.dumps(summary, indent=2))
    print(f"Open {index}")


if __name__ == "__main__":
    main()
