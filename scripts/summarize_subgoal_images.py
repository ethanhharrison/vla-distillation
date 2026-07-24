"""Visualize a Stage B (subgoal_image) run as an HTML contact sheet.

Reads a run's `results.json` (produced by `pipeline.subgoal_image.generate`) and
renders a self-contained, debug-first HTML report. For each example (source
step) it shows the three real source cameras once, then — stacked underneath for
easy side-by-side comparison — one row per variant (backend x prompt template x
subgoal source), each showing the subgoal frame per camera at BOTH native
resolution and downscaled to 224x224 (the policy resolution), with the
perceptual-hash delta vs source, the per-sample cost, the prompt-template id, and
any error surfaced (failures are shown, not hidden).

Usage:
    python scripts/summarize_subgoal_images.py <run-dir | results.json | run-name> [--open]
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
RUNS_DIR = PROJECT_ROOT / "outputs" / "subgoal_images"
DEFAULT_VIZ_DIR = PROJECT_ROOT / "outputs" / "visualizations"


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
    matches = sorted(RUNS_DIR.glob(f"*{target}*/results.json"),
                     key=lambda q: q.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No results.json found for {target!r} under {RUNS_DIR}")


def _img(img_dir: Path, name: str | None, cls: str) -> str:
    if not name:
        return f'<div class="missing {cls}">—</div>'
    path = img_dir / name
    if not path.is_file():
        return f'<div class="missing {cls}">missing: {html.escape(name)}</div>'
    # Re-encode to JPEG q85 (native pixel dimensions preserved) before embedding:
    # the models return large PNGs, so base64-embedding them verbatim bloats the
    # self-contained HTML ~10x. This keeps native resolution but a viewable size.
    data = _jpeg_bytes(path)
    enc = base64.b64encode(data).decode("ascii")
    return f'<img class="{cls}" src="data:image/jpeg;base64,{enc}" alt="{html.escape(name)}">'


def _jpeg_bytes(path: Path, quality: int = 85) -> bytes:
    import io

    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def _phash_badge(phash: dict | None) -> str:
    if not phash:
        return ""
    norm = phash["norm"]
    # traffic-light band: too-weak (<0.06) grey, sane mid green, too-strong (>0.45) red
    if norm < 0.06:
        cls, tip = "phash weak", "very small change (edit may be too weak)"
    elif norm > 0.45:
        cls, tip = "phash strong", "large change (may be a different scene)"
    else:
        cls, tip = "phash mid", "mid-band change"
    return (f'<span class="{cls}" title="{tip}">Δphash {phash["bits"]}/'
            f'{phash["hash_size"] ** 2} = {norm:.3f}</span>')


BACKEND_NAMES = {
    "gemini_image": "Gemini",
    "openai_image": "OpenAI",
    "real_future": "Real future",
    "dummy_image": "Dummy",
}


def _variant_title(s: dict) -> str:
    """Clean, single-line title, e.g. 'Gemini: gemini-2.5-flash-image · default'."""
    disp = BACKEND_NAMES.get(s["backend"], s["backend"])
    if s["backend"] == "real_future":
        k = s.get("k")
        return f"{html.escape(disp)}" + (f" <span class='dim'>(k={k})</span>" if k is not None else "")
    if s["backend"] == "dummy_image":
        return html.escape(disp)
    title = f"{html.escape(disp)}: <span class='model'>{html.escape(str(s.get('model')))}</span>"
    if s.get("prompt_template"):
        title += (f" · <span class='tpl'>{html.escape(str(s.get('prompt_template')))}</span>"
                  f" <code class='dim'>{html.escape(str(s.get('prompt_template_id')))}</code>")
    return title


def render_html(results: dict, img_dir: Path, source_path: Path) -> str:
    run = results["run"]
    samples = results["samples"]
    cameras = run["cameras"]

    # group samples by example, preserving order
    by_example: dict[str, list[dict]] = {}
    order: list[str] = []
    for s in samples:
        by_example.setdefault(s["example_id"], []).append(s)
        if s["example_id"] not in order:
            order.append(s["example_id"])

    # phash roll-up
    deltas_by_backend: dict[str, list[float]] = {}
    n_errors = 0
    for s in samples:
        for r in s["cameras"].values():
            if r.get("error"):
                n_errors += 1
            elif r.get("phash"):
                deltas_by_backend.setdefault(s["backend"], []).append(r["phash"]["norm"])

    # header: templates
    tpl_rows = "".join(
        f'<details><summary><code>{html.escape(t["id"])}</code> '
        f'<strong>{html.escape(t["name"])}</strong></summary>'
        f'<pre>{html.escape(t["text"])}</pre></details>'
        for t in run.get("prompt_templates", [])
    ) or "<em>none (free backends only)</em>"

    phash_summary = "".join(
        f"<li><strong>{html.escape(b)}</strong>: n={len(d)}, "
        f"min/med/max = {min(d):.3f} / {sorted(d)[len(d)//2]:.3f} / {max(d):.3f}</li>"
        for b, d in sorted(deltas_by_backend.items())
    ) or "<li><em>no images</em></li>"

    example_sections = []
    for ex_id in order:
        group = by_example[ex_id]
        first = group[0]
        instruction = first.get("instruction", "")
        step = first.get("step_index")

        # source row (shown once; identical across variants of this example)
        src_cells = "".join(
            f'<div class="cell"><div class="camlabel">{html.escape(cam)}</div>'
            f'{_img(img_dir, first["cameras"].get(cam, {}).get("source"), "native")}</div>'
            for cam in cameras
        )

        variant_rows = []
        for s in group:
            cells = []
            for cam in cameras:
                r = s["cameras"].get(cam, {})
                if r.get("error"):
                    body = f'<div class="err">ERROR: {html.escape(str(r["error"]))}</div>'
                else:
                    ns = r.get("native_size") or []
                    ns_txt = f'{ns[0]}×{ns[1]}' if len(ns) == 2 else "?"
                    body = (
                        f'<figure>{_img(img_dir, r.get("subgoal"), "native")}'
                        f'<figcaption>{ns_txt} {_phash_badge(r.get("phash"))}</figcaption>'
                        f'</figure>'
                    )
                cells.append(f'<div class="cell">{body}</div>')
            variant_rows.append(
                f'<div class="variant v-{html.escape(s["subgoal_source"])}">'
                f'<div class="vlabel">{_variant_title(s)}</div>'
                f'<div class="row">{"".join(cells)}</div></div>'
            )

        example_sections.append(f"""
        <section class="example">
          <h3>{html.escape(ex_id)} <span class="sub">step {step} — “{html.escape(instruction)}”</span></h3>
          <div class="srcblock"><div class="srclabel">source</div><div class="row">{src_cells}</div></div>
          {"".join(variant_rows)}
        </section>
        """)

    meta_rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in {
            "episode_id": run.get("episode_id"),
            "dataset": run.get("dataset"),
            "backends": ", ".join(run.get("backends", [])),
            "cameras": ", ".join(cameras),
            "num_samples": len(samples),
            "num_errors": n_errors,
        }.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Subgoal images — {html.escape(run.get('episode_id', source_path.name))}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
         max-width: 1500px; padding: 24px; line-height: 1.45; }}
  h1 {{ margin-bottom: 2px; }} .subtitle {{ color:#888; margin-top:0; }}
  .card {{ border:1px solid #8883; border-radius:10px; padding:12px 18px; margin:14px 0; }}
  table {{ border-collapse:collapse; }} th {{ text-align:left; padding-right:16px; color:#888; vertical-align:top; }}
  details pre {{ white-space:pre-wrap; background:#8881; padding:10px; border-radius:8px; }}
  .example {{ border-top:2px solid #8884; padding-top:10px; margin-top:26px; }}
  .example h3 {{ margin:4px 0; }} .example .sub {{ color:#888; font-weight:400; font-size:14px; }}
  .row {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:14px; }}
  .cell {{ min-width:0; }} .camlabel, .srclabel {{ font-size:12px; color:#888; margin:2px 0; font-weight:600; }}
  .srcblock {{ background:#8881; border-radius:10px; padding:10px 12px; margin:6px 0 12px; }}
  .variant {{ border:1px solid #8883; border-left:4px solid #8886; border-radius:10px;
             padding:10px 12px; margin:12px 0; }}
  .v-edited {{ border-left-color:#8e24aa; }}
  .v-real_future {{ border-left-color:#1565c0; }}
  .v-dummy {{ border-left-color:#9e9e9e; }}
  .vlabel {{ font-size:15px; font-weight:600; margin-bottom:8px; }}
  figure {{ margin:0; }} figcaption {{ font-size:12px; color:#888; text-align:center; margin-top:3px; }}
  img.native {{ width:100%; height:auto; border-radius:8px; display:block; }}
  .model {{ color:#888; font-weight:500; }} .tpl {{ color:#8e24aa; }}
  .dim {{ color:#999; font-weight:400; font-size:0.85em; }}
  .phash {{ font-size:11px; border-radius:6px; padding:1px 6px; font-weight:600; margin-left:4px; }}
  .phash.weak {{ background:#61616122; color:#9e9e9e; }}
  .phash.mid {{ background:#2e7d3222; color:#2e7d32; }}
  .phash.strong {{ background:#c6282822; color:#c62828; }}
  .err {{ background:#c6282818; color:#c62828; border-radius:6px; padding:8px; font-size:12px; }}
  .missing {{ background:#8881; border-radius:6px; padding:8px; color:#c33; font-size:12px; text-align:center; }}
  .missing.native {{ min-height:120px; display:flex; align-items:center; justify-content:center; }}
</style></head><body>
  <h1>Stage B — subgoal images</h1>
  <p class="subtitle">{html.escape(run.get('episode_id',''))} —
     {html.escape(', '.join(run.get('backends', [])))}
     {'· <strong style=color:#c62828>ABORTED ON BUDGET</strong>' if run.get('aborted_on_budget') else ''}</p>

  <div class="card"><h2>Run</h2><table>{meta_rows}</table></div>
  <div class="card"><h2>phash-delta distribution (verify-B signal)</h2>
    <ul>{phash_summary}</ul>
    <p class="subtitle">band: &lt;0.06 too-weak · 0.06–0.45 sane · &gt;0.45 too-strong (heuristic, not yet a gate)</p>
  </div>
  <div class="card"><h2>Prompt templates</h2>{tpl_rows}</div>

  {''.join(example_sections)}
  <footer class="subtitle">Rendered from {html.escape(str(source_path))} on
    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</footer>
</body></html>"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="Run dir, results.json path, or a run name under outputs/subgoal_images/.")
    p.add_argument("--output", default=None, help="Output .html (default: outputs/visualizations/).")
    p.add_argument("--open", action="store_true", help="Open in the default browser.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results_path = resolve_run(args.run)
    img_dir = results_path.parent / "images"
    results = json.loads(results_path.read_text())

    out = Path(args.output) if args.output else (
        DEFAULT_VIZ_DIR / f"subgoal_{results_path.parent.name}.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(results, img_dir, results_path))
    print(f"Wrote contact sheet ({len(results['samples'])} samples) to {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
