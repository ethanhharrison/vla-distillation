"""Render per-run report.html pages (same layout as batch_fewshot_ablations.py)
for a batch_model_effort_sweep.py output directory, plus one index.html per
source ablation batch comparing every (prompt, model, effort) setting.

Query stills and the viewing video are taken from each source run's
report.html (the dataset may not be local); few-shot example stills are
regenerated from fewshot_examples.yaml.

Usage:
    python scripts/render_model_effort_sweep.py outputs/model_effort_sweep \
        --source-root .. [--grades outputs/model_effort_sweep/grades.json]
"""

from __future__ import annotations

import argparse
import base64
import functools
import html
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # to import the pipeline package

import summarize_dense_description_fewshot as viz

from pipeline.dense_description.generate_fewshot import EXAMPLES_YAML, load_labeled_examples

# Few-shot example stills are identical across runs - load each clip once.
viz.load_clip_images = functools.lru_cache(maxsize=None)(viz.load_clip_images)

_IMG_RE = re.compile(r'<img src="data:image/jpeg;base64,([^"]+)" alt="([^"]+)">')
_VIDEO_RE = re.compile(r'<video[^>]*src="data:video/mp4;base64,([^"]+)"')

MODEL_ORDER = ["gpt-6-luna", "gpt-6-sol", "gpt-6-astra"]
EFFORT_ORDER = ["none", "low", "xhigh"]
TEMPLATE_ORDER = ["default", "compare_explicit"]


def load_source_query(run_dir: Path, cameras: list[str]) -> tuple[dict[str, dict[str, bytes]], bytes]:
    text = (run_dir / "report.html").read_text()
    imgs = _IMG_RE.findall(text)[-2 * len(cameras):]
    n = len(cameras)
    images = {
        "start": {cam: base64.b64decode(b) for cam, (b, _) in zip(cameras, imgs[:n])},
        "end": {cam: base64.b64decode(b) for cam, (b, _) in zip(cameras, imgs[n:])},
    }
    return images, base64.b64decode(_VIDEO_RE.search(text).group(1))


def _failure_reason(record: dict, manual: str | None) -> str | None:
    """Rubric judge verdict if the run was judged, else the manual grade."""
    verdict = record.get("judge")
    if verdict is None:
        return manual
    if verdict["pass"]:
        return None
    return "; ".join(f'{cid}: {verdict["criteria"][cid]["reason"]}' for cid in verdict["failed_criteria"])


def setting_key(r: dict) -> tuple:
    return (TEMPLATE_ORDER.index(r["prompt_template"]), MODEL_ORDER.index(r["model"]), EFFORT_ORDER.index(r["reasoning_effort"]))


def write_index(batch_dir: Path, source_batch: str, entries: list[dict], has_grades: bool) -> None:
    queries = sorted({e["query_label"] for e in entries})
    reps = sorted({e["rep"] for e in entries})
    rows: dict[tuple, dict] = defaultdict(dict)
    for e in entries:
        rows[setting_key(e["record"])][(e["query_label"], e["rep"])] = e

    header = "".join(
        f'<th colspan="{len(reps)}">{html.escape(q)}<div class="blurb">'
        f'{html.escape(next(e["instruction"] for e in entries if e["query_label"] == q))}</div></th>'
        for q in queries
    )
    body = []
    for key in sorted(rows):
        cells = rows[key]
        first = next(iter(cells.values()))["record"]
        costs = [c["record"]["cost_usd"] for c in cells.values()]
        lat = [c["record"]["latency_s"] for c in cells.values()]
        n_crit = sum(c["critical"] is not None for c in cells.values())
        tds = []
        for q in queries:
            for rep in reps:
                c = cells.get((q, rep))
                if c is None:
                    tds.append("<td></td>")
                    continue
                cls = "bad" if c["critical"] else ("ok" if has_grades else "")
                note = f'<div class="why">{html.escape(c["critical"])}</div>' if c["critical"] else ""
                tds.append(
                    f'<td class="{cls}"><a href="{html.escape(c["dir"])}/report.html">r{rep}</a>'
                    f'<div class="preview">{html.escape(c["record"]["shared"].get("terminal_gripper_state", "")[:110])}</div>{note}</td>'
                )
        grade = (
            f'<div class="score {"bad" if n_crit else "ok"}">{n_crit}/{len(cells)} fail</div>' if has_grades else ""
        )
        body.append(
            f'<tr><th>{html.escape(first["model"])} · {html.escape(first["reasoning_effort"])}'
            f'<div class="blurb">prompt: {html.escape(first["prompt_template"])}</div>'
            f'<div class="blurb">${statistics.mean(costs):.4f}/call · {statistics.mean(lat):.1f}s</div>{grade}</th>'
            f'{"".join(tds)}</tr>'
        )

    rep_header = "".join(f"<th>r{r}</th>" for _ in queries for r in reps)
    (batch_dir / "index.html").write_text(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Model x effort sweep</title>
<style>
  :root {{ color-scheme: light dark; --ok:#2e7d32; --bad:#c62828; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1500px;
         margin: 40px auto; padding: 0 20px; line-height: 1.45; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 13px; }}
  th, td {{ border: 1px solid #8883; padding: 8px; text-align: left; vertical-align: top; }}
  th {{ background: #8881; }} tbody th {{ width: 200px; }}
  td.ok {{ background: #2e7d3212; }} td.bad {{ background: #c6282822; }}
  .blurb {{ font-weight: 400; font-size: 12px; color: #888; margin-top: 3px; }}
  .preview {{ font-size: 11px; color: #888; margin-top: 3px; }}
  .why {{ font-size: 11px; color: var(--bad); margin-top: 3px; font-weight: 600; }}
  .score {{ font-size: 12px; margin-top: 4px; font-weight: 700; }}
  .score.ok {{ color: var(--ok); }} .score.bad {{ color: var(--bad); }}
  a {{ font-weight: 600; }}
</style></head><body>
  <h1>Model x reasoning-effort sweep</h1>
  <p>Queries from <code>{html.escape(source_batch)}</code> (current ideal: 4s, start/end stills, contrast 1.5,
  instruction hidden). Cell text = model's terminal gripper state.{
      " Red = FAIL under the correctness rubric (pipeline/dense_description/rubric.py): any failed criterion fails the run." if has_grades else ""}</p>
  <table>
    <thead><tr><th rowspan="2">setting</th>{header}</tr><tr>{rep_header}</tr></thead>
    <tbody>{"".join(body)}</tbody>
  </table>
</body></html>""")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dir")
    parser.add_argument("--source-root", default="..", help="Directory holding the source ablation batches.")
    parser.add_argument("--grades", default=None, help="Optional grades.json with {'critical': {run_stem: reason}}.")
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir)
    source_root = Path(args.source_root)
    grades = json.loads(Path(args.grades).read_text())["critical"] if args.grades else None
    examples_by_id = {ex["id"]: ex for ex in load_labeled_examples(EXAMPLES_YAML)}

    queries = {
        p.stem: json.loads(p.read_text())["run"] for p in sorted((sweep_dir / "queries").glob("*.json"))
    }
    source_media = {
        stem: load_source_query(source_root / Path(*stem.split("__")), run["cameras"])
        for stem, run in queries.items()
    }

    by_batch: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((sweep_dir / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        query_stem = record["query"].replace("/", "__")
        source_batch, source_run = record["query"].split("/")
        usage = record["usage"]
        run = {
            **queries[query_stem],
            "model": record["model"],
            "prompt_template": record["prompt_template"],
            "reasoning_effort": record["reasoning_effort"],
            "tokens": f'{usage["prompt_tokens"]} in ({(usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0} cached)'
                      f' / {usage["completion_tokens"]} out ({record["reasoning_tokens"]} reasoning)',
            "latency_s": record["latency_s"],
            "cost_usd": f'{record["cost_usd"]:.4f}',
        }
        result = {"run": run, "raw_response": record["raw_response"],
                  "shared": record["shared"], "per_view": record["per_view"], "judge": record.get("judge")}
        rep = int(path.stem.rsplit("__r", 1)[1])
        query_label = source_run.removeprefix("current_ideal_")
        out_dir = sweep_dir / "reports" / source_batch / (
            f'{record["prompt_template"]}_{record["model"].removeprefix("gpt-6-")}_{record["reasoning_effort"]}'
            f"_{query_label}_r{rep}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        images, video = source_media[query_stem]
        (out_dir / "report.html").write_text(
            viz.render_html(result, examples_by_id, out_dir / "result.json", query_images=images, query_video=video)
        )
        by_batch[source_batch].append({
            "dir": out_dir.name, "record": record, "rep": rep, "query_label": query_label,
            "instruction": run["language_instruction"],
            "critical": _failure_reason(record, (grades or {}).get(path.stem)),
        })
        print(f"wrote {out_dir}")

    for source_batch, entries in by_batch.items():
        write_index(sweep_dir / "reports" / source_batch, source_batch, entries,
                    grades is not None or any("judge" in e["record"] for e in entries))
        print(f"Open {sweep_dir / 'reports' / source_batch / 'index.html'}")


if __name__ == "__main__":
    main()
