"""Visualize a dense_description video-input run as an HTML report.

Reads a run's `results.json` (produced by
`pipeline.dense_description.generate_video`) and renders a self-contained
report: one section per clip with its description, and — for the clips whose
mp4 was kept — an embedded, playable video. Writes `report.html` into the run
directory itself (not a shared visualizations dir), so the whole run —
descriptions, sample videos, and report — is one folder to `scp -r` to a local
machine.

Usage:
    python scripts/summarize_dense_description_videos.py <run-dir | results.json | run-name> [--open]
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import webbrowser
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "outputs" / "dense_description_video"


def resolve_run(target: str) -> Path:
    """Accept a run dir, a results.json path, or a run name under RUNS_DIR."""
    p = Path(target)
    if p.is_file() and p.name == "results.json":
        return p
    if p.is_dir() and (p / "results.json").is_file():
        return p / "results.json"
    cand = RUNS_DIR / target / "results.json"
    if cand.is_file():
        return cand
    matches = sorted(
        RUNS_DIR.glob(f"*{target}*/results.json"),
        key=lambda q: q.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No results.json found for {target!r} under {RUNS_DIR}")


def _video_block(run_dir: Path, video_path: str | None) -> str:
    if not video_path:
        return '<div class="missing">(video not kept for this clip)</div>'
    path = run_dir / video_path
    if not path.is_file():
        return f'<div class="missing">missing: {html.escape(video_path)}</div>'
    enc = base64.b64encode(path.read_bytes()).decode("ascii")
    size_kb = path.stat().st_size / 1024
    return (
        f'<video controls loop muted playsinline src="data:video/mp4;base64,{enc}"></video>'
        f'<div class="videometa">{html.escape(path.name)} — {size_kb:.0f} KB</div>'
    )


def render_html(payload: dict, run_dir: Path, source_path: Path) -> str:
    run = payload["run"]
    clips = payload["clips"]

    meta_rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in {
            "record": run.get("record"),
            "provider": run.get("provider"),
            "model": run.get("model"),
            "video_cameras (top/bottom)": " / ".join(run.get("video_cameras", [])),
            "clip_seconds": run.get("clip_seconds"),
            "fps": run.get("fps"),
            "trajectory_length": run.get("trajectory_length"),
            "num_clips": run.get("num_clips"),
        }.items()
    )

    cost = run.get("cost")
    cost_card = ""
    if cost:
        cost_rows = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in cost.items()
        )
        cost_card = f'<div class="card"><h2>Estimated cost</h2><table>{cost_rows}</table></div>'

    clip_sections = []
    for clip in clips:
        clip_sections.append(f"""
        <section class="clip">
          <h3>clip {clip['clip_index']} <span class="sub">steps {clip['start_step']}-{clip['end_step']}</span></h3>
          <p class="instruction">“{html.escape(clip['language_instruction'])}”</p>
          <div class="body">
            <div class="videocol">{_video_block(run_dir, clip.get('video_path'))}</div>
            <div class="desccol"><p class="description-body">{html.escape(clip['description'])}</p></div>
          </div>
        </section>
        """)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Dense descriptions (video) — {html.escape(Path(run.get('record', source_path.name)).stem)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
         max-width: 1200px; padding: 24px; line-height: 1.45; }}
  h1 {{ margin-bottom: 2px; }} .subtitle {{ color:#888; margin-top:0; }}
  .card {{ border:1px solid #8883; border-radius:10px; padding:12px 18px; margin:14px 0; }}
  table {{ border-collapse:collapse; }} th {{ text-align:left; padding-right:16px; color:#888; vertical-align:top; }}
  .clip {{ border-top:2px solid #8884; padding-top:12px; margin-top:26px; }}
  .clip h3 {{ margin:4px 0; }} .clip .sub {{ color:#888; font-weight:400; font-size:14px; }}
  .instruction {{ color:#8e24aa; margin:2px 0 10px; }}
  .body {{ display:grid; grid-template-columns: minmax(240px, 420px) 1fr; gap:20px; align-items:start; }}
  video {{ width:100%; border-radius:10px; display:block; background:#000; }}
  .videometa {{ font-size:12px; color:#888; margin-top:4px; }}
  .description-body {{ white-space:pre-wrap; }}
  .missing {{ background:#8881; border-radius:8px; padding:16px; color:#c33; font-size:13px; text-align:center; }}
  @media (max-width: 720px) {{ .body {{ grid-template-columns: 1fr; }} }}
</style></head><body>
  <h1>Dense descriptions — video input</h1>
  <p class="subtitle">{html.escape(str(run.get('record', '')))}</p>

  <div class="card"><h2>Run</h2><table>{meta_rows}</table></div>
  {cost_card}

  {''.join(clip_sections)}
  <footer class="subtitle">Rendered from {html.escape(str(source_path))} on
    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</footer>
</body></html>"""  # noqa: DTZ005


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="Run dir, results.json path, or a run name under outputs/dense_description_video/.")
    p.add_argument("--open", action="store_true", help="Open in the default browser.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results_path = resolve_run(args.run)
    run_dir = results_path.parent
    payload = json.loads(results_path.read_text())

    out = run_dir / "report.html"
    out.write_text(render_html(payload, run_dir, results_path))
    print(f"Wrote report ({len(payload['clips'])} clips) to {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
