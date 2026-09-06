"""Compare frame-based vs video-based dense descriptions on matching clips.

Pairs clips (by start/end step) between a `pipeline.dense_description.generate`
run (start/end stills per camera) and a `pipeline.dense_description.generate_video`
run (a single stacked-view clip video), and renders both side by side with their
descriptions — so it's easy to see whether the video input changes what gets
described. Reuses each mode's own parsing/rendering helpers rather than
reimplementing them.

Usage:
    python scripts/compare_dense_description_modes.py \
        outputs/dense_descriptions/success-00285_gemini_20260906-161649.txt \
        outputs/dense_description_video/success-00285_gemini_20260906-160138
"""

from __future__ import annotations

import argparse
import html
import webbrowser
from datetime import datetime
from pathlib import Path

from summarize_dense_description_videos import _video_block
from summarize_dense_description_videos import resolve_run as resolve_video_run
from summarize_dense_descriptions import (
    _frame_row,
    _render_description,
    parse_run,
    resolve_run_file,
)

import json


def render_html(frame_run: dict, video_payload: dict, video_dir: Path, frame_path: Path, video_path: Path) -> str:
    frame_clips = {c["start_step"]: c for c in frame_run["clips"]}
    video_clips = {c["start_step"]: c for c in video_payload["clips"]}
    shared_starts = sorted(set(frame_clips) & set(video_clips))

    frame_info = frame_run["info"]
    video_run = video_payload["run"]

    meta_rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in {
            "record": frame_info.get("record", video_run.get("record")),
            "model": frame_info.get("model", video_run.get("model")),
            "frame cameras": frame_info.get("cameras"),
            "video cameras (top/bottom)": " / ".join(video_run.get("video_cameras", [])),
            "matched clips": len(shared_starts),
        }.items()
    )

    sections = []
    for start in shared_starts:
        fc, vc = frame_clips[start], video_clips[start]
        sections.append(f"""
        <section class="clip">
          <h3>steps {fc['start_step']}-{fc['end_step']}</h3>
          <p class="instruction">“{html.escape(fc['language'])}”</p>
          <div class="body">
            <div class="col">
              <h4>Frames (start/end stills)</h4>
              {_frame_row("start", fc["start_images"])}
              {_frame_row("end", fc["end_images"])}
              <div class="desc">{_render_description(fc["description"])}</div>
            </div>
            <div class="col">
              <h4>Video (continuous, {video_run.get('fps')} fps)</h4>
              {_video_block(video_dir, vc.get("video_path"))}
              <div class="desc"><p class="description-body">{html.escape(vc["description"])}</p></div>
            </div>
          </div>
        </section>
        """)

    if not shared_starts:
        sections.append(
            '<p class="missing">No clips share a start step between the two runs '
            "— re-run both over the same trajectory/clip-seconds so they line up.</p>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Dense descriptions — frames vs video</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
         max-width: 1400px; padding: 24px; line-height: 1.45; }}
  h1 {{ margin-bottom: 2px; }} .subtitle {{ color:#888; margin-top:0; }}
  .card {{ border:1px solid #8883; border-radius:10px; padding:12px 18px; margin:14px 0; }}
  table {{ border-collapse:collapse; }} th {{ text-align:left; padding-right:16px; color:#888; vertical-align:top; }}
  .clip {{ border-top:2px solid #8884; padding-top:12px; margin-top:26px; }}
  .clip h3 {{ margin:4px 0; }}
  .instruction {{ color:#8e24aa; margin:2px 0 10px; }}
  .body {{ display:grid; grid-template-columns: 1fr 1fr; gap:24px; align-items:start; }}
  .col h4 {{ margin:0 0 8px; color:#888; font-size:13px; text-transform:uppercase; letter-spacing:.03em; }}
  .frame-row {{ margin-bottom:10px; }} .frame-row h4 {{ display:none; }}
  .frames {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .frames figure {{ margin:0; width:160px; }} .frames img {{ width:100%; border-radius:8px; display:block; }}
  .frames figcaption {{ font-size:11px; color:#888; text-align:center; }}
  video {{ width:100%; border-radius:10px; display:block; background:#000; }}
  .videometa {{ font-size:12px; color:#888; margin-top:4px; }}
  .desc {{ margin-top:10px; background:#8881; border-radius:8px; padding:10px 12px; }}
  .description-body {{ white-space:pre-wrap; margin:0; }}
  .cam-section h5 {{ margin:6px 0 2px; font-size:12px; color:#888; }}
  .missing {{ background:#8881; border-radius:8px; padding:16px; color:#c33; text-align:center; }}
  @media (max-width: 800px) {{ .body {{ grid-template-columns: 1fr; }} }}
</style></head><body>
  <h1>Dense descriptions — frames vs video</h1>
  <p class="subtitle">Same clip(s), same model, different input format.</p>
  <div class="card"><h2>Run</h2><table>{meta_rows}</table></div>

  {''.join(sections)}
  <footer class="subtitle">Frames: {html.escape(str(frame_path))} · Video: {html.escape(str(video_path))}
    · rendered {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</footer>
</body></html>"""  # noqa: DTZ005


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames", help="Frame-based run: .txt path or a name under outputs/dense_descriptions/.")
    p.add_argument("video", help="Video-based run: dir, results.json, or a name under outputs/dense_description_video/.")
    p.add_argument("--output", default=None, help="Output .html (defaults to comparison.html inside the video run dir).")
    p.add_argument("--open", action="store_true", help="Open in the default browser.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frame_path = resolve_run_file(args.frames)
    frame_run = parse_run(frame_path)
    video_results_path = resolve_video_run(args.video)
    video_dir = video_results_path.parent
    video_payload = json.loads(video_results_path.read_text())

    out = Path(args.output) if args.output else video_dir / "comparison.html"
    out.write_text(render_html(frame_run, video_payload, video_dir, frame_path, video_results_path))
    print(f"Wrote comparison to {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
