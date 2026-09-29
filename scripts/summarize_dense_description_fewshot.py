"""Visualize a generate_fewshot.py run: the few-shot examples used as context
(your hand-written ground truth) alongside the query clip and the model's
Shared + Per-view answer. The model only ever sees start/end stills (no
video) - but the query clip's video is also rendered here purely for your
own viewing, so you can judge the model's answer against the actual motion.

Usage:
    python scripts/summarize_dense_description_fewshot.py \
        outputs/dense_description_fewshot_runs/success-00285_ex9_s0150_gemini_20260913-200858.json
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # to import the pipeline package

from pipeline.dense_description.generate_fewshot import EXAMPLES_YAML, clip_images, load_labeled_examples
from pipeline.dense_description.video import render_clip_video
from pipeline.language_instruction.trajectory import load_trajectory

RUNS_DIR = PROJECT_ROOT / "outputs" / "dense_description_fewshot_runs"

SHARED_LABELS = {
    "gripper": "Gripper",
    "scene_change": "Scene change",
    "terminal_gripper_state": "Terminal gripper state",
}


def _fmt_cost(cost: dict | None) -> str:
    if not cost or cost.get("total_cost_usd") is None:
        return "unknown (unpriced model)"
    return (
        f"${cost['total_cost_usd']:.4f} "
        f"({cost['input_tokens']} in / {cost['output_tokens']} out tokens)"
    )


def resolve_run(target: str) -> Path:
    p = Path(target)
    if p.is_file():
        return p
    matches = sorted(RUNS_DIR.glob(f"*{target}*.json"), key=lambda q: q.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No fewshot run found for {target!r} under {RUNS_DIR}")
    return matches[0]


def _img(jpeg: bytes, alt: str) -> str:
    enc = base64.b64encode(jpeg).decode("ascii")
    return f'<img src="data:image/jpeg;base64,{enc}" alt="{html.escape(alt)}">'


def _frame_rows(images_by_cam_time: dict[str, dict[str, bytes]], cameras: list[str]) -> str:
    """images_by_cam_time: {"start": {cam: jpeg}, "end": {cam: jpeg}}."""
    rows = []
    for label in ("start", "end"):
        figs = "".join(
            f'<figure>{_img(images_by_cam_time[label][cam], f"{label}/{cam}")}'
            f'<figcaption>{html.escape(cam)}</figcaption></figure>'
            for cam in cameras
            if cam in images_by_cam_time[label]
        )
        rows.append(f'<div class="frame-row"><h5>{label}</h5><div class="frames">{figs}</div></div>')
    return "".join(rows)


def _shared_per_view_block(shared: dict, per_view: dict, cameras: list[str], title: str) -> str:
    shared_rows = "".join(
        f'<tr><th>{html.escape(label)}</th><td>{html.escape(str(shared.get(key, "")))}</td></tr>'
        for key, label in SHARED_LABELS.items()
    )
    per_view_rows = "".join(
        f'<tr><th>{html.escape(cam)}</th><td>{html.escape(str(per_view.get(cam, "")))}</td></tr>'
        for cam in cameras
    )
    return f"""
    <div class="answer">
      <h5>{html.escape(title)}</h5>
      <table class="shared">{shared_rows}</table>
      <table class="perview">{per_view_rows}</table>
    </div>
    """


def _judge_block(verdict: dict | None) -> str:
    """Rubric verdict (pipeline/dense_description/rubric.py) for the query, if graded."""
    if not verdict:
        return ""
    from pipeline.dense_description.rubric import RUBRIC

    rows = "".join(
        f'<tr class="{"ok" if verdict["criteria"][c.id]["pass"] else "bad"}">'
        f'<th>{"✓" if verdict["criteria"][c.id]["pass"] else "✗"} {html.escape(c.name)}</th>'
        f'<td>{html.escape(verdict["criteria"][c.id]["reason"])}</td></tr>'
        for c in RUBRIC if c.id in verdict["criteria"]
    )
    objects = ", ".join(
        f'<span class="{"ok" if o.get("correct") else "bad"}">{html.escape(str(o.get("mentioned")))}'
        + ("" if o.get("correct") else f' → {html.escape(str(o.get("actual")))}') + "</span>"
        for o in verdict.get("objects", [])
    )
    status = "PASS" if verdict["pass"] else "FAIL"
    return f"""
    <div class="judge {'ok' if verdict['pass'] else 'bad'}">
      <h5>Rubric verdict: <span class="verdict">{status}</span>
        <span class="sub">judge {html.escape(str(verdict.get('judge_model')))} · {html.escape(str(verdict.get('judge_effort', '')))}</span></h5>
      <p><b>What actually happens (judge):</b> {html.escape(verdict.get('ground_truth', ''))}</p>
      <table>{rows}</table>
      <p class="objects"><b>Objects named:</b> {objects}</p>
    </div>
    """


def load_clip_images(
    record: str,
    example_index: int,
    start_step: int,
    end_step: int,
    cameras: tuple[str, ...],
    rotate_180: frozenset[str],
    contrast_factor: float | None,
) -> dict[str, dict[str, bytes]]:
    """Regenerate a clip's start/end stills with the exact same transforms
    (rotation, contrast) the run actually used, so the viz always matches
    what the model was shown - regardless of which trial variant this is."""
    trajectory = load_trajectory(Path(record), cameras, example_index)
    flat = clip_images(trajectory, start_step, end_step, cameras, rotate_180, contrast_factor)
    n = len(cameras)
    return {
        "start": dict(zip(cameras, flat[:n])),
        "end": dict(zip(cameras, flat[n : 2 * n])),
    }


def render_clip_video_bytes(
    record: str,
    example_index: int,
    start_step: int,
    end_step: int,
    cameras: tuple[str, ...],
    rotate_180: frozenset[str],
    fps: float = 15.0,
) -> bytes:
    """Render any clip (example or query) as a stacked-view mp4."""
    trajectory = load_trajectory(Path(record), cameras, example_index)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "clip.mp4"
        return render_clip_video(
            trajectory, start_step, end_step, cameras, fps, out_path, rotate_180=rotate_180,
        )


def _video_block(video_bytes: bytes, caption: str) -> str:
    b64 = base64.b64encode(video_bytes).decode("ascii")
    return (
        f'<div class="frame-row"><h5>{html.escape(caption)}</h5>'
        f'<video controls loop muted playsinline src="data:video/mp4;base64,{b64}"></video></div>'
    )


def render_html(
    result: dict,
    examples_by_id: dict[str, dict],
    source_path: Path,
    query_images: dict[str, dict[str, bytes]] | None = None,
    query_video: bytes | None = None,
) -> str:
    """`query_images` / `query_video` override loading the query clip from
    its record (e.g. when re-rendering a run whose dataset isn't local)."""
    run = result["run"]
    cameras = run["cameras"]
    rotate_180 = frozenset(run.get("rotate_180_cameras", []))
    contrast_factor = run.get("contrast_factor")
    input_mode = run.get("input_mode", "stills")
    is_video = input_mode == "video"
    provider = run.get("provider", "")

    example_sections = []
    for ex_id in run["fewshot_example_ids"]:
        ex = examples_by_id.get(ex_id, {})
        if is_video:
            video_bytes = render_clip_video_bytes(
                ex["record"], ex["example_index"], ex["start_step"], ex["end_step"], tuple(cameras), rotate_180,
            )
            caption = "video sent to the model" + (" (as a sampled frame burst — see below)" if provider == "openai" else "")
            media = _video_block(video_bytes, caption)
        else:
            images = load_clip_images(
                ex["record"], ex["example_index"], ex["start_step"], ex["end_step"],
                tuple(cameras), rotate_180, contrast_factor,
            )
            media = _frame_rows(images, cameras)
        example_sections.append(f"""
        <section class="ex fewshot">
          <h3>{html.escape(ex_id)} <span class="tag">few-shot example</span>
            <span class="sub">example {ex.get('example_index')} · steps {ex.get('start_step')}-{ex.get('end_step')}
            ({ex.get('clip_seconds')}s)</span></h3>
          <p class="instruction">"{html.escape(str(ex.get('language_instruction', '')))}"</p>
          <div class="body">
            <div class="col">{media}</div>
            <div class="col">{_shared_per_view_block(ex.get('shared') or {}, ex.get('per_view') or {}, cameras, "Hand-written (ground truth)")}</div>
          </div>
        </section>
        """)

    if query_video is None:
        query_video = render_clip_video_bytes(
            run["record"], run["example_index"], run["start_step"], run["end_step"], tuple(cameras), rotate_180,
        )
    if is_video:
        query_caption = "video sent to the model" + (
            " (as a sampled frame burst — this API has no native video input)" if provider == "openai"
            else " (native video input)"
        )
        query_media = _video_block(query_video, query_caption)
    else:
        if query_images is None:
            query_images = load_clip_images(
                run["record"], run["example_index"], run["start_step"], run["end_step"],
                tuple(cameras), rotate_180, contrast_factor,
            )
        query_media = _frame_rows(query_images, cameras) + _video_block(
            query_video, "video (viewing only — the model saw only the stills above)"
        )
    instruction_note = (
        "shown to the model" if run.get("show_query_instruction", True)
        else "NOT shown to the model — for your reference only"
    )
    query_section = f"""
    <section class="ex query">
      <h3>Query <span class="tag">held out — not in few-shot pool</span>
        <span class="sub">example {run['example_index']} · steps {run['start_step']}-{run['end_step']}
        ({run['clip_seconds']:g}s)</span></h3>
      <p class="instruction">"{html.escape(run['language_instruction'])}" <span class="sub">({instruction_note})</span></p>
      <div class="body">
        <div class="col">{query_media}</div>
        <div class="col">{_shared_per_view_block(result['shared'], result['per_view'], cameras, "Model output")}
          {_judge_block(result.get('judge'))}</div>
      </div>
    </section>
    """

    meta_rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in {
            "record": run.get("record"),
            "provider": run.get("provider"),
            "model": run.get("model"),
            "reasoning_effort": run.get("reasoning_effort") or "n/a",
            "input mode": input_mode,
            "cameras (top-to-bottom)": " / ".join(cameras),
            "rotated 180": ", ".join(run.get("rotate_180_cameras", [])) or "none",
            "contrast factor": contrast_factor if contrast_factor is not None else "off (1.0)",
            "prompt template": run.get("prompt_template", "default"),
            "query instruction shown to model": run.get("show_query_instruction", True),
            "few-shot examples": ", ".join(run["fewshot_example_ids"]),
            **({"cost": _fmt_cost(run["cost"])} if "cost" in run else {}),
            # Only present for runs from batch_model_effort_sweep.py.
            **{k: run[k] for k in ("reasoning_effort", "tokens", "latency_s", "cost_usd") if k in run},
        }.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Few-shot dense descriptions</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
         max-width: 1300px; padding: 24px; line-height: 1.45; }}
  h1 {{ margin-bottom: 2px; }} .subtitle {{ color:#888; margin-top:0; }}
  .card {{ border:1px solid #8883; border-radius:10px; padding:12px 18px; margin:14px 0; }}
  table {{ border-collapse:collapse; margin-bottom:10px; width:100%; }}
  th {{ text-align:left; padding:3px 10px 3px 0; color:#888; vertical-align:top; width:34%; font-weight:600; }}
  td {{ padding:3px 0; vertical-align:top; }}
  .ex {{ border-top:2px solid #8884; padding-top:14px; margin-top:30px; border-radius:8px; }}
  .ex.query {{ background:#8e24aa11; padding:14px; border-top:none; border:2px solid #8e24aa55; }}
  .ex h3 {{ margin:4px 0; }} .sub {{ color:#888; font-weight:400; font-size:13px; }}
  .tag {{ font-size:11px; text-transform:uppercase; letter-spacing:.03em; background:#8882;
         border-radius:5px; padding:2px 7px; margin:0 8px; }}
  .ex.query .tag {{ background:#8e24aa33; color:#8e24aa; }}
  .instruction {{ color:#8e24aa; margin:2px 0 12px; }}
  .body {{ display:grid; grid-template-columns: minmax(260px, 420px) 1fr; gap:20px; align-items:start; }}
  .frame-row {{ margin-bottom:10px; }} .frame-row h5 {{ margin:0 0 6px; color:#888; font-size:12px;
    text-transform:uppercase; letter-spacing:.03em; }}
  .frames {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .frames figure {{ margin:0; width:120px; }} .frames img {{ width:100%; border-radius:8px; display:block; }}
  .frames figcaption {{ font-size:10px; color:#888; text-align:center; }}
  video {{ width:100%; max-width:320px; border-radius:8px; display:block; background:#000; margin-top:4px; }}
  .answer h5 {{ margin:0 0 8px; color:#888; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }}
  .answer table.perview {{ margin-top:2px; }}
  .judge {{ border:2px solid #8884; border-radius:8px; padding:10px 14px; margin-top:14px; }}
  .judge.ok {{ border-color:#2e7d3288; }} .judge.bad {{ border-color:#c6282888; }}
  .judge h5 {{ margin:0 0 6px; font-size:13px; text-transform:uppercase; letter-spacing:.03em; }}
  .judge.ok .verdict, tr.ok th, span.ok {{ color:#2e7d32; }} .judge.bad .verdict, tr.bad th, span.bad {{ color:#c62828; }}
  .judge p {{ margin:4px 0 8px; font-size:13px; }} .judge table th {{ width:38%; }}
  @media (max-width: 800px) {{ .body {{ grid-template-columns: 1fr; }} }}
</style></head><body>
  <h1>Few-shot dense descriptions</h1>
  <p class="subtitle">Hand-written examples teach the model your Shared + Per-view format; the query clip is held out.</p>
  <div class="card"><h2>Run</h2><table>{meta_rows}</table></div>

  {''.join(example_sections)}
  {query_section}
  <footer class="subtitle">Rendered from {html.escape(str(source_path))} on
    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</footer>
</body></html>"""  # noqa: DTZ005


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="Path to a generate_fewshot.py results .json, or a name under outputs/dense_description_fewshot_runs/.")
    p.add_argument("--output", default=None, help="Output .html (defaults next to the run json).")
    p.add_argument("--open", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_path = resolve_run(args.run)
    result = json.loads(run_path.read_text())

    examples_by_id = {ex["id"]: ex for ex in load_labeled_examples(EXAMPLES_YAML)}

    out = Path(args.output) if args.output else run_path.with_suffix(".html")
    out.write_text(render_html(result, examples_by_id, run_path))
    print(f"Wrote {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
