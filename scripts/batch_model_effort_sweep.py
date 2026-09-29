"""Re-run the query clips of earlier few-shot ablation batches across several
(model, reasoning_effort, prompt_template) settings and record token usage,
latency and cost per call, to find the cheapest setting that stays accurate.

Each query is taken from a prior run directory (result.json + report.html, as
written by batch_fewshot_ablations.py). The query images are read straight
from that report, so they are byte-identical to what the original model saw
(including rotation / contrast); the few-shot example turns are rebuilt from
fewshot_examples.yaml with the same contrast factor.

Every generated description is graded against the correctness rubric in
pipeline/dense_description/rubric.py by a VLM judge (default gpt-6-astra,
xhigh) that also sees intermediate clip frames and the task instruction;
the verdict is stored under "judge" in each run's JSON. A description
passes only if every rubric criterion passes.

Usage:
    python scripts/batch_model_effort_sweep.py \
        ../ablations_gpt6/current_ideal_4s_a ../ablations_gpt6/current_ideal_4s_b \
        --output outputs/model_effort_sweep --repeats 3
"""

from __future__ import annotations

import argparse
import base64
import functools
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # to import the pipeline package

from dotenv import load_dotenv

load_dotenv()

from pipeline.dense_description.generate_fewshot import EXAMPLES_YAML, example_turns, load_labeled_examples
from pipeline.dense_description.prompts import fewshot_clip_header, fewshot_instructions, parse_fewshot_response
from pipeline.dense_description.rubric import (
    DEFAULT_JUDGE_EFFORT,
    DEFAULT_JUDGE_MODEL,
    evidence_from_video,
    judge_description,
)
from pipeline.language_instruction.vlm import ChatTurn, build_vlm

# (model, reasoning_effort)
DEFAULT_SETTINGS: list[tuple[str, str]] = [
    ("gpt-6-astra", "xhigh"),
    ("gpt-6-astra", "low"),
    ("gpt-6-sol", "xhigh"),
    ("gpt-6-sol", "low"),
    ("gpt-6-sol", "none"),
    ("gpt-6-luna", "xhigh"),
    ("gpt-6-luna", "low"),
    ("gpt-6-luna", "none"),
]
DEFAULT_TEMPLATES = ["default", "compare_explicit"]

# USD per 1M tokens: (input, cached input, output). Reasoning tokens bill as output.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-6-astra": (10.0, 1.0, 50.0),
    "gpt-6-sol": (2.0, 0.2, 10.0),
    "gpt-6-luna": (0.10, 0.01, 0.50),
}

_IMG_RE = re.compile(r'<img src="data:image/jpeg;base64,([^"]+)" alt="([^"]+)">')
_VIDEO_RE = re.compile(r'<video[^>]*src="data:video/mp4;base64,([^"]+)"')


def load_query(run_dir: Path) -> dict:
    """The prior run's config plus the exact query images it sent (the last
    len(cameras)*2 images in its report: start cameras, then end cameras)."""
    result = json.loads((run_dir / "result.json").read_text())
    run = result["run"]
    report = (run_dir / "report.html").read_text()
    images = _IMG_RE.findall(report)
    n = 2 * len(run["cameras"])
    query = images[-n:]
    expected = [f"{t}/{c}" for t in ("start", "end") for c in run["cameras"]]
    if [alt for _, alt in query] != expected:
        raise ValueError(f"{run_dir}: unexpected query image order {[alt for _, alt in query]}")
    decoded = [base64.b64decode(b) for b, _ in query]
    cams = run["cameras"]
    return {
        "name": f"{run_dir.parent.name}/{run_dir.name}",
        "run": run,
        "images": decoded,
        "stills": {"start": dict(zip(cams, decoded[: len(cams)])), "end": dict(zip(cams, decoded[len(cams):]))},
        "video": base64.b64decode(_VIDEO_RE.search(report).group(1)),
        "reference_response": result["raw_response"],
    }


def build_turns(query: dict, template: str, examples: list[dict]) -> list[ChatTurn]:
    run = query["run"]
    cameras = tuple(run["cameras"])
    rotate = frozenset(run["rotate_180_cameras"])
    turns = [ChatTurn(role="user", text=fewshot_instructions(cameras, template))]
    for ex in examples:
        turns += example_turns(ex, cameras, rotate, run["contrast_factor"])
    header = fewshot_clip_header(
        language_instruction=run["language_instruction"],
        start_step=run["start_step"],
        end_step=run["end_step"],
        total=run["trajectory_length"],
        clip_seconds=run["clip_seconds"],
        include_instruction=run["show_query_instruction"],
    ) + " Now write your answer in the format above."
    turns.append(ChatTurn(role="user", text=header, images=query["images"]))
    return turns


def judge_record(record: dict, query: dict, judge_model: str, judge_effort: str) -> dict:
    """Grade `record` against the rubric (pipeline/dense_description/rubric.py)
    and store the verdict under record["judge"]."""
    judge_vlm = build_vlm("openai", model=judge_model, reasoning_effort=judge_effort)
    evidence = evidence_from_video(query["video"], query["run"]["cameras"], query["stills"])
    verdict = judge_description(judge_vlm, record["raw_response"], query["run"]["language_instruction"], evidence)
    verdict["judge_effort"] = judge_effort
    verdict["cost_usd"] = cost_usd(judge_model, verdict["judge_usage"])
    record["judge"] = verdict
    return record


def cost_usd(model: str, usage: dict) -> float:
    inp, cached_inp, out = PRICES[model]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    uncached = usage["prompt_tokens"] - cached
    return (uncached * inp + cached * cached_inp + usage["completion_tokens"] * out) / 1e6


def run_one(
    query: dict, template: str, model: str, effort: str, turns: list[ChatTurn], out_path: Path,
    judge: tuple[str, str] | None,
) -> dict:
    vlm = build_vlm("openai", model=model, reasoning_effort=effort)
    t0 = time.time()
    raw = vlm.generate_chat(turns)
    latency = time.time() - t0
    usage = vlm.last_usage
    record = {
        "query": query["name"],
        "model": model,
        "reasoning_effort": effort,
        "prompt_template": template,
        "latency_s": round(latency, 2),
        "usage": usage,
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cost_usd": cost_usd(model, usage),
        "raw_response": raw,
        **parse_fewshot_response(raw, tuple(query["run"]["cameras"])),
    }
    if judge:
        judge_record(record, query, *judge)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="Prior run directories (each with result.json + report.html).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--templates", default=",".join(DEFAULT_TEMPLATES))
    parser.add_argument(
        "--settings", default=None,
        help="Comma-separated MODEL:EFFORT pairs. Default: " + ",".join(f"{m}:{e}" for m, e in DEFAULT_SETTINGS),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-effort", default=DEFAULT_JUDGE_EFFORT)
    parser.add_argument("--no-judge", action="store_true", help="Skip rubric grading (not recommended).")
    parser.add_argument(
        "--judge-only", action="store_true",
        help="Don't generate; (re-)grade every existing run under OUTPUT/runs with the rubric judge.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.output)
    settings = (
        [tuple(s.split(":")) for s in args.settings.split(",")] if args.settings else DEFAULT_SETTINGS
    )
    templates = args.templates.split(",")
    examples = load_labeled_examples(EXAMPLES_YAML)
    queries = [load_query(Path(d)) for d in args.run_dirs]
    judge = None if args.no_judge else (args.judge_model, args.judge_effort)

    if args.judge_only:
        by_name = {q["name"]: q for q in queries}
        paths = sorted((out_dir / "runs").glob("*.json"))
        print(f"{len(paths)} runs to judge")

        def rejudge(path: Path) -> dict:
            record = json.loads(path.read_text())
            judge_record(record, by_name[record["query"]], args.judge_model, args.judge_effort)
            path.write_text(json.dumps(record, indent=2))
            return record

        _run_pool(args.workers, {path.name: functools.partial(rejudge, path) for path in paths})
        return

    jobs = []
    for q in queries:
        (out_dir / "queries").mkdir(parents=True, exist_ok=True)
        (out_dir / "queries" / f"{q['name'].replace('/', '__')}.json").write_text(
            json.dumps({"run": q["run"], "reference_response": q["reference_response"]}, indent=2)
        )
        for template in templates:
            turns = build_turns(q, template, examples)
            for model, effort in settings:
                for rep in range(args.repeats):
                    path = out_dir / "runs" / f"{q['name'].replace('/', '__')}__{template}__{model}__{effort}__r{rep}.json"
                    if not path.exists():  # resumable
                        jobs.append((q, template, model, effort, turns, path))

    print(f"{len(jobs)} calls to make")
    _run_pool(args.workers, {job[-1].name: functools.partial(run_one, *job, judge) for job in jobs})


def _run_pool(workers: int, tasks: dict) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                verdict = r.get("judge")
                graded = f"  {'PASS' if verdict['pass'] else 'FAIL ' + ','.join(verdict['failed_criteria'])}" if verdict else ""
                print(f"ok  {futures[fut]}  ${r['cost_usd']:.4f}{graded}", flush=True)
            except Exception as ex:  # noqa: BLE001 - keep the batch going; rerun resumes
                print(f"ERR {futures[fut]}: {ex}", flush=True)


if __name__ == "__main__":
    main()
