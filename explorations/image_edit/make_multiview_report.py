"""HTML contact sheet for a `multiview_probe.py` run.

Built to answer one question by eye: **do the three views agree with each
other?** No pixel metric can answer that — the views are different viewpoints,
so cross-view agreement is not a pixel comparison — which is why the numbers here
deliberately bound only the things they *can* decide (did the canvas layout
survive, how much did each view move, is it still the same scene) and the rest is
laid out for a human to judge.

Reads `results.json` and the PNGs the probe wrote, so it never calls an API and
can be re-run freely after a run.

    ../../.venv/bin/python make_multiview_report.py --run multiview_gemini_image --open
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import webbrowser
from pathlib import Path

import numpy as np
from PIL import Image

from metrics import fmt, mean

HERE = Path(__file__).resolve().parent
CAMERAS = ("exterior_1", "exterior_2", "wrist")


def _uri(path: Path, width: int = 300) -> str:
    im = Image.open(path).convert("RGB")
    if im.width != width:
        im = im.resize((width, max(1, round(im.height * width / im.width))))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _row(sdir: Path, files: dict[str, str], captions: dict[str, str] | None = None) -> str:
    captions = captions or {}
    cells = []
    for cam in CAMERAS:
        if cam not in files:
            continue
        cells.append(
            f'<figure><img src="{_uri(sdir / files[cam])}">'
            f'<figcaption>{html.escape(cam)}{captions.get(cam, "")}</figcaption></figure>'
        )
    return f'<div class="row">{"".join(cells)}</div>'


def _badge(label: str, value: str, kind: str = "") -> str:
    return f' <span class="m {kind}">{html.escape(label)} <b>{html.escape(value)}</b></span>'


def _view_table(views: dict) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in CAMERAS if c in views)
    rows = ""
    for label, key in (("edit vs source", "vs_src"),
                       ("vs real t+k", "vs_real"),
                       ("real motion", "real_motion")):
        cells = "".join(f"<td>{fmt(views[c].get(key))}</td>" for c in CAMERAS if c in views)
        rows += f"<tr><th>{label}</th>{cells}</tr>"
    sizes = "".join(
        f'<td>{"x".join(str(v) for v in views[c].get("native_size", []))}</td>'
        for c in CAMERAS if c in views
    )
    rows += f"<tr><th>returned size</th>{sizes}</tr>"
    return f"<table><tr><th></th>{head}</tr>{rows}</table>"


def _layout_block(chk: dict, canvas: dict, sdir: Path) -> str:
    ok = chk["preserved"]
    verdict = ("layout preserved — every panel came back where it went in"
               if ok else
               f"LAYOUT BROKE — only {chk['panels_matched']}/{chk['n_panels']} panels "
               "came back in their own position")
    cams = list(chk["detail"])
    head = "".join(f"<th>src {html.escape(c)}</th>" for c in cams)
    rows = ""
    for cam in cams:
        d = chk["detail"][cam]
        cells = ""
        for c in cams:
            star = ' class="hit"' if d["best_match"] == c else ""
            cells += f'<td{star}>{fmt(d["diffs"][c])}</td>'
        rows += f"<tr><th>out {html.escape(cam)}</th>{cells}</tr>"
    pair = ""
    if canvas.get("source_file") and canvas.get("output_file"):
        pair = (
            '<div class="pair">'
            f'<figure><img src="{_uri(sdir / canvas["source_file"], 430)}">'
            f'<figcaption>canvas sent — {canvas.get("source_geometry", "")}</figcaption></figure>'
            f'<figure><img src="{_uri(sdir / canvas["output_file"], 430)}">'
            f'<figcaption>canvas returned — '
            f'{"x".join(str(v) for v in canvas.get("output_size", []))} '
            f'(aspect {canvas.get("source_aspect")} &rarr; {canvas.get("output_aspect")})'
            "</figcaption></figure></div>"
        )
    return (
        f'{pair}<p class="{"ok" if ok else "bad"}">{html.escape(verdict)}</p>'
        '<p class="dim">Each output panel vs every source panel; the marked cell is '
        'the closest match. On the diagonal = the grid survived. Off it = the model '
        'reflowed the composite, which a single vs-source number cannot distinguish '
        'from a large edit.</p>'
        f'<table class="assign"><tr><th></th>{head}</tr>{rows}</table>'
    )


def build(run_dir: Path) -> str:
    index = json.loads((run_dir / "results.json").read_text())
    samples = index["samples"]
    k = index.get("k")

    by_sit: dict[str, list[dict]] = {}
    for s in samples:
        by_sit.setdefault(s["situation_id"], []).append(s)

    # ---- summary across situations, per strategy -------------------------- #
    strategies = index.get("strategies", [])

    def agg(strat: str) -> dict | None:
        rows = [s for s in samples if s["strategy"] == strat]
        if not rows:
            return None
        vs_src, vs_real, real, per_cam = [], [], [], {}
        for s in rows:
            for cam, v in s["views"].items():
                vs_src.append(v["vs_src"])
                per_cam.setdefault(cam, []).append(v["vs_src"])
                if v.get("vs_real") is not None:
                    vs_real.append(v["vs_real"])
                if v.get("real_motion") is not None:
                    real.append(v["real_motion"])
        lay = [s["canvas"]["layout_check"] for s in rows
               if s.get("canvas", {}).get("layout_check")]
        return {
            "rows": rows, "vs_src": mean(vs_src), "vs_real": mean(vs_real),
            "real": mean(real), "per_cam": {c: mean(v) for c, v in per_cam.items()},
            "layout": (f"{sum(c['preserved'] for c in lay)}/{len(lay)}" if lay else "n/a"),
            "calls": sum(s["n_calls"] for s in rows),
            "cost": sum(s["cost_usd"] for s in rows),
            "errs": sum(len(s["errors"]) for s in rows),
            "routes": sorted({r for s in rows for r in s.get("routes", [])}),
        }

    stats = {s: a for s in strategies if (a := agg(s))}

    summary_rows = ""
    for strat, a in stats.items():
        floor = stats.get(f"{strat}:noop", {}).get("vs_src")
        above = (a["vs_src"] - floor) if floor is not None else None
        is_noop = strat.endswith(":noop")
        cls = ' class="noop"' if is_noop else ""
        above_cell = ("<td class=\"dim\">(this is the floor)</td>" if is_noop
                      else f"<td><b>{fmt(above)}</b></td>" if above is not None
                      else "<td>n/a</td>")
        summary_rows += (
            f"<tr{cls}><th>{html.escape(strat)}</th>"
            f"<td>{a['calls']}</td>"
            f"<td>{fmt(a['vs_src'])}</td>{above_cell}"
            f"<td>{fmt(a['vs_real'])}</td>"
            f"<td>{fmt(a['real'])}</td><td>{a['layout']}</td>"
            f"<td>${a['cost']:.3f}</td>"
            f"<td>{a['errs'] or ''}</td>"
            f"<td>{html.escape(','.join(a['routes']))}</td></tr>"
        )

    # per-camera edit magnitude, floor-corrected — where view starvation shows up
    cams_seen = [c for c in CAMERAS if any(c in a["per_cam"] for a in stats.values())]
    percam_rows = ""
    for strat, a in stats.items():
        if strat.endswith(":noop"):
            continue
        fl = stats.get(f"{strat}:noop", {}).get("per_cam", {})
        cells = ""
        for c in cams_seen:
            e = a["per_cam"].get(c)
            f0 = fl.get(c)
            cells += (f"<td>{fmt(e)}{f' <span class=dim>&minus;{fmt(f0)}</span>' if f0 is not None else ''}"
                      f"{f' = <b>{fmt(e - f0)}</b>' if (e is not None and f0 is not None) else ''}</td>")
        real_cells = "".join(
            f"<td class=\"dim\">{fmt(mean([v['real_motion'] for s in a['rows'] if c in s['views'] for v in [s['views'][c]] if v.get('real_motion') is not None]))}</td>"
            for c in cams_seen)
        percam_rows += (f"<tr><th>{html.escape(strat)}</th>{cells}</tr>"
                        f"<tr><th class=\"dim\">&nbsp;&nbsp;real motion</th>{real_cells}</tr>")

    # ---- per-situation sections ------------------------------------------ #
    sections = []
    for sid in sorted(by_sit):
        sdir = run_dir / sid
        first = by_sit[sid][0]
        blocks = [
            f'<div class="sit"><h3>{html.escape(sid)} '
            f'<span class="dim">{html.escape(first["instruction"])}</span></h3>',
            '<div class="lbl">source observation (real, t)</div>',
            _row(sdir, {c: f"source__{c}.png" for c in CAMERAS
                        if (sdir / f"source__{c}.png").exists()}),
        ]
        real_files = {c: f"real__{c}.png" for c in CAMERAS
                      if (sdir / f"real__{c}.png").exists()}
        if real_files:
            blocks += [f'<div class="lbl real">real future at t+{k} — what actually '
                       'happened (the yardstick)</div>', _row(sdir, real_files)]

        for s in by_sit[sid]:
            cls = ("noop" if s["strategy"].endswith(":noop")
                   else "canvas" if s["strategy"].startswith("canvas")
                   else "multi" if s["strategy"] == "multiturn" else "indep")
            badges = _badge("cost", f"${s['cost_usd']:.3f}")
            if s.get("cached"):
                badges += _badge("cached", "yes", "c")
            if s.get("latency_s"):
                badges += _badge("latency", f"{s['latency_s']}s")
            badges += _badge("calls", str(s["n_calls"]))
            if s.get("routes"):
                badges += _badge("route", s["routes"][0], "r")
            blocks.append(f'<div class="cond {cls}"><h4>{html.escape(s["strategy"])}'
                          f'{badges}</h4>')
            for e in s["errors"]:
                blocks.append(f'<p class="err">{html.escape(e)}</p>')
            chk = s.get("canvas", {}).get("layout_check")
            if chk:
                blocks.append(_layout_block(chk, s["canvas"], sdir))
            if s["views"]:
                caps = {c: f' <span class="dim">turn {v["turn"]}</span>'
                        for c, v in s["views"].items() if "turn" in v}
                blocks.append('<div class="lbl">subgoal views</div>')
                blocks.append(_row(sdir, {c: v["file"] for c, v in s["views"].items()}, caps))
                blocks.append(_view_table(s["views"]))
            blocks.append("</div>")
        blocks.append("</div>")
        sections.append("".join(blocks))

    cost = index.get("cost") or {}
    oa = index.get("openai") or {}
    subtitle = " · ".join(filter(None, [
        str(index.get("backend")), str(index.get("model")),
        f"quality={oa.get('quality')}" if oa else "",
        f"template={index.get('prompt_template')}",
        f"spend ${fmt(cost.get('spent_usd'), 4)}",
    ]))

    percam_head = "".join(f"<th>{html.escape(c)}</th>" for c in cams_seen)

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Multi-view probe — {html.escape(str(index.get('model', '')))}</title><style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; max-width:1080px; margin:0 auto; padding:24px; line-height:1.45; }}
  .card {{ border:1px solid #8883; border-radius:10px; padding:12px 18px; margin:14px 0; }}
  .sit {{ border-top:3px solid #8885; margin-top:34px; padding-top:8px; }}
  .sit h3 {{ margin:6px 0; }} .dim {{ color:#888; font-weight:400; font-size:13px; }}
  .lbl {{ font-size:12px; color:#888; font-weight:600; margin:10px 0 4px; }}
  .lbl.real {{ color:#1565c0; }}
  .row {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; max-width:940px; margin-bottom:10px; }}
  .pair {{ display:grid; grid-template-columns:repeat(2,1fr); gap:10px; max-width:940px; margin:6px 0 10px; }}
  figure {{ margin:0; }} figure img {{ width:100%; border-radius:6px; display:block; }}
  figcaption {{ font-size:11px; color:#888; text-align:center; margin-top:2px; }}
  .cond {{ border-left:4px solid #8886; border-radius:8px; padding:6px 14px; margin:14px 0; }}
  .cond.indep {{ border-left-color:#1565c0; }}
  .cond.canvas {{ border-left-color:#ef6c00; }}
  .cond.multi {{ border-left-color:#8e24aa; }}
  .cond.noop {{ border-left-color:#2e7d32; opacity:.82; }}
  tr.noop th, tr.noop td {{ color:#2e7d32; }}
  .cond h4 {{ margin:6px 0; }}
  .err {{ font-size:12px; color:#c62828; }}
  .ok {{ font-size:13px; color:#2e7d32; font-weight:600; margin:6px 0; }}
  .bad {{ font-size:13px; color:#c62828; font-weight:600; margin:6px 0; }}
  table {{ border-collapse:collapse; margin:4px 0 8px; }}
  th,td {{ padding:2px 14px 2px 0; text-align:left; font-size:13px; }} th {{ color:#888; }}
  table.assign td {{ padding:2px 12px 2px 0; color:#888; }}
  table.assign td.hit {{ color:inherit; font-weight:700; }}
  code {{ font-size:12px; background:#8881; padding:0 4px; border-radius:4px; }}
  .m {{ font-size:10px; background:#8881; border-radius:5px; padding:0 4px; font-weight:600; }}
  .m.c {{ background:#2e7d3222; color:#2e7d32; }} .m.r {{ background:#8e24aa22; color:#8e24aa; }}
</style></head><body>
  <h1>Can the hosted editors do all three views at once?</h1>
  <p class="dim">{html.escape(subtitle)}</p>

  <div class="card"><h2>The question</h2>
  <p>Subgoal generation needs three <b>mutually consistent</b> camera views. Today each
  camera gets its own independent API call, so nothing stops an object moving in
  <code>exterior_1</code> and sitting still in <code>exterior_2</code>. Three ways to ask:</p>
  <ul>
    <li><b>independent</b> — one call per camera. What ships today; the control.</li>
    <li><b>canvas</b> — the three views stitched into one image, edited in a single call,
        split back apart. One request, so the views <i>can</i> cohere, and it is 3x
        cheaper per situation. Two layouts are tried, because the provider can only
        return certain aspect ratios and the canvas shape interacts with that.</li>
    <li><b>multiturn</b> — views edited one at a time in the same conversation, each with
        the earlier edits in context. Same cost as independent.</li>
  </ul>
  </div>

  <div class="card"><h2>Summary</h2>
  <table><tr><th>strategy</th><th>calls</th><th>edit vs src</th>
  <th>above own floor</th><th>vs real t+k</th>
  <th>real motion</th><th>layout kept</th><th>spend</th><th>errors</th><th>route</th></tr>
  {summary_rows}</table>
  <p class="dim">Mean absolute pixel difference on 0-255 RGB, the same unit as the
  <code>analyze.py</code> table and the Cosmos&nbsp;3 / Wan tables. <b>Above own
  floor</b> is the column that matters: each strategy&rsquo;s edit magnitude minus
  what that same strategy produces when told to change <i>nothing</i>. A stitched
  composite and a single frame do not pay the same re-render tax, so raw edit
  magnitude cannot be compared across strategies &mdash; this can.</p>
  </div>

  <div class="card"><h2>Per camera, floor-corrected</h2>
  <table><tr><th>strategy</th>{percam_head}</tr>{percam_rows}</table>
  <p class="dim">Read as <code>edit &minus; floor = attributable</code>. This is where
  <b>view starvation</b> shows up: a canvas that buys exterior motion by giving up on
  the wrist panel looks fine in the aggregate and bad here. Compare each column
  against its own real motion (grey row) &mdash; the wrist genuinely moves several
  times more than the exteriors over the same span.</p>
  </div>

  <div class="card"><h2>What these numbers do and do not settle</h2>
  <ul>
    <li><b>They cannot score cross-view consistency.</b> The views are different
        viewpoints, so agreement between them is not a pixel comparison. That is the
        judgement to make by eye from the rows below — does the same object move the
        same way in all three?</li>
    <li><b>They do settle whether the canvas survives the round trip</b> (the assignment
        table per canvas run) and <b>whether any view got starved</b>: compare each view's
        edit-vs-source against its own real motion. A canvas that collapses the wrist view
        shows up there — the failure the Wan harness found with the same layouts.</li>
    <li><b>Edit magnitude is floor-corrected</b> when a run includes <code>--noop</code>:
        each strategy is also asked to change nothing, with the same wording the
        full-run table used, so the tax is measured per strategy rather than borrowed.
        Where a floor is missing the table says <code>n/a</code> rather than showing an
        uncorrected number as if it were comparable.</li>
    <li><b>One draw per cell, few situations.</b> This is a go / no-go probe, not a
        measurement. Nothing here resolves a small effect.</li>
  </ul>
  </div>

  {"".join(sections)}
</body></html>"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Run dir name under results/.")
    p.add_argument("--out", default=None)
    p.add_argument("--open", action="store_true")
    args = p.parse_args(argv)

    run_dir = Path(args.run)
    if not run_dir.exists():
        run_dir = HERE / "results" / args.run
    doc = build(run_dir)
    out = Path(args.out) if args.out else run_dir / "report.html"
    out.write_text(doc)
    print(f"wrote {out}  ({len(doc) / 1e6:.1f} MB)")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
