"""Build the image-edit experiment report — same layout as the Cosmos 3 one.

Reads results/<run>/index.json and the shared situation set. Per situation
(grouped, not scattered):

  <situation> — anchor t, original instruction
    Source observation .......... 3 real cameras in one row
    Real future (the yardstick) . the same 3 cameras at t+k
    A · sanity (original) ....... the edited subgoal, 3 cameras
    B · counterfactual (Gemini) .
    C · null / stress (authored)
    N · no-op floor (if run) .... what the model does when told to change nothing

Where the Cosmos 3 report shows two timestamps from a rollout, there is only one
image here: an editor produces the subgoal directly rather than a trajectory
towards it. And where that report carries an action-MAE column, this one cannot
— an image editor emits no actions, which is exactly the gap that makes it a
Stage-B-only candidate rather than a B+C replacement.

    ../../.venv/bin/python make_report.py --run runs --open
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
from metrics import diff, fmt, load_rgb, mean

HERE = Path(__file__).resolve().parent
CAMERAS = ("exterior_1", "exterior_2", "wrist")   # display order
COND = {
    "A": ("A · sanity (original instruction)", "a"),
    "B": ("B · counterfactual (Gemini-proposed)", "b"),
    "C": ("C · null / stress (authored: impossible tasks)", "c"),
    "MP": ("Targeted test — “move the pot”", "t"),
    "N": ("N · no-op floor (told to change nothing)", "n"),
}
COND_ORDER = ["A", "B", "C", "MP", "N"]


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #

def _b64(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _arr_to_uri(arr: np.ndarray, width: int = 300) -> str:
    from PIL import Image

    im = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if im.width != width:
        im = im.resize((width, round(im.height * width / im.width)))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return _b64(buf.getvalue())


def cam_row(cams: dict[str, np.ndarray], captions: dict[str, str] | None = None) -> str:
    captions = captions or {}
    cells = "".join(
        f'<figure><img src="{_arr_to_uri(cams[c])}">'
        f'<figcaption>{html.escape(c)}{captions.get(c, "")}</figcaption></figure>'
        for c in CAMERAS if c in cams
    )
    return f'<div class="row">{cells}</div>'


def badge(label: str, value: float, kind: str = "") -> str:
    return f' <span class="m {kind}">{html.escape(label)} <b>{value:.1f}</b></span>'


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def build(run_dir: Path, sit_dir: Path) -> tuple[str, dict]:
    index = json.loads((run_dir / "index.json").read_text())
    cams_run = [c for c in CAMERAS if c in index.get("cameras", CAMERAS)]
    k = index.get("k") or index.get("horizon")
    fps = index.get("fps") or 15

    by_sit: dict[str, list[dict]] = {}
    for s in index["samples"]:
        by_sit.setdefault(s["situation_id"], []).append(s)

    sections, latencies = [], []
    real_mags, noop_mags, sens = [], [], []

    for sid in sorted(by_sit):
        sdir = sit_dir / sid
        sit = json.loads((sdir / "situation.json").read_text())
        samples = by_sit[sid]

        src = {c: load_rgb(sdir / "history" / f"{c}_0.png") for c in cams_run
               if (sdir / "history" / f"{c}_0.png").exists()}

        gen: dict[str, dict[str, np.ndarray]] = {}
        for s in samples:
            gen[s["sample_id"]] = {}
            for c in cams_run:
                info = s["cameras"].get(c)
                if info and info.get("image"):
                    if info.get("latency_s") is not None:
                        latencies.append(info["latency_s"])
                    gen[s["sample_id"]][c] = load_rgb(run_dir / s["sample_id"] / info["image"])

        a_sample = next((s for s in samples if s["condition"] == "A"), None)
        a_gen = gen.get(a_sample["sample_id"], {}) if a_sample else {}

        blocks = [f'<div class="lbl">source observation (real cameras, anchor t={sit["anchor"]})</div>'
                  f'{cam_row(src)}']
        real = {c: load_rgb(sdir / "future" / f"{c}_f{k:02d}.png") for c in cams_run
                if (sdir / "future" / f"{c}_f{k:02d}.png").exists()}
        if real:
            caps = {}
            for c, r in real.items():
                if c in src:
                    d = diff(r, src[c])
                    real_mags.append(d)
                    caps[c] = badge("vs src", d)
            blocks.append(f'<div class="lbl real">REAL future +{k} frames ({k / fps:.2f}s) — the yardstick</div>'
                          f'{cam_row(real, caps)}')

        present = [c for c in COND_ORDER if any(s["condition"] == c for s in samples)]
        present += sorted({s["condition"] for s in samples} - set(COND_ORDER))
        for cond in present:
            items = []
            for s in [x for x in samples if x["condition"] == cond]:
                cams_i, caps = {}, {}
                for c in cams_run:
                    img = gen[s["sample_id"]].get(c)
                    if img is None:
                        continue
                    cams_i[c] = img
                    cap = badge("vs src", diff(img, src[c])) if c in src else ""
                    if c in real:
                        cap += badge("vs real", diff(img, real[c]), "r")
                    if cond != "A" and c in a_gen:
                        dv = diff(img, a_gen[c])
                        cap += badge("vs A", dv, "d")
                        if cond == "N":
                            noop_mags.append(diff(img, src[c]) if c in src else float("nan"))
                        else:
                            sens.append({"situation": sid, "condition": cond, "camera": c,
                                         "instruction": s["instruction"], "image_vs_a": dv})
                    caps[c] = cap
                body = cam_row(cams_i, caps) if cams_i else "<em>no image</em>"
                errs = [f'{c}: {html.escape(str((s["cameras"].get(c) or {}).get("error")))}'
                        for c in cams_run if (s["cameras"].get(c) or {}).get("error")]
                err_html = f'<div class="err">{" · ".join(errs)}</div>' if errs else ""
                label = ("“change nothing” control prompt" if cond == "N"
                         else f'“{html.escape(s["instruction"])}”')
                items.append(f'<div class="gen-item"><div class="instr">{label}</div>'
                             f'{err_html}{body}</div>')

            title, cls = COND.get(cond, (cond, "x"))
            blocks.append(f'<div class="cond {cls}"><h4>{html.escape(title)}</h4>{"".join(items)}</div>')

        sections.append(
            f'<section class="sit"><h3>{html.escape(sid)} '
            f'<span class="dim">· anchor t={sit["anchor"]} · original: '
            f'“{html.escape(sit["instruction"])}”</span></h3>{"".join(blocks)}</section>')

    # ---- calibration card ---------------------------------------------------
    real_mag = mean(real_mags)
    noop = mean([v for v in noop_mags if v == v]) if noop_mags else None
    null = (index.get("resample_null") or {}).get("image")

    rows = ""
    for cond in COND_ORDER:
        vals = [r["image_vs_a"] for r in sens if r["condition"] == cond]
        if not vals:
            continue
        title = COND[cond][0].split(" · ")[-1] if cond in COND else cond
        rows += (f"<tr><td><b>{cond}</b> {html.escape(title)}</td>"
                 f"<td>{len(vals)}</td><td>{mean(vals):.1f}</td></tr>")
    all_v = [r["image_vs_a"] for r in sens]
    if all_v:
        ratio = f"{mean(all_v) / null:.0%}" if null else "n/a"
        rows += (f"<tr><td><b>overall</b></td><td>{len(all_v)}</td>"
                 f"<td><b>{mean(all_v):.1f}</b> ({ratio} of the null)</td></tr>")

    floor_line = (
        f'<li><b>no-op floor: {noop:.1f}</b> — how far the image moves when the model is told to '
        f'change nothing. The editor\'s equivalent of a reconstruction floor; treat anything '
        f'below it as zero.</li>' if noop is not None else
        '<li><b>no-op floor: not measured</b> — this run had no condition N, so every '
        '<span class="m">vs src</span> below includes an unmeasured re-render tax. '
        'Re-run with <code>--noop</code> to bound it.</li>'
    )
    null_line = (
        f'<li><b>resample null: {null:.1f}</b> — the same instruction asked twice. An effect '
        f'below this is indistinguishable from re-asking the question.</li>' if null else
        '<li><b>resample null: not measured</b> — sensitivity below is uninterpretable until '
        '<code>resample_null.py</code> has run.</li>'
    )

    meta = "".join(
        f"<tr><td>{html.escape(k_)}</td><td>{html.escape(str(v))}</td></tr>"
        for k_, v in {
            "backend": index.get("backend"), "model": index.get("model"),
            "prompt template": f'{index.get("prompt_template")} ({index.get("prompt_template_id")})',
            "cameras": ", ".join(cams_run), "future frame k": k,
            "openai params": index.get("openai"), "no-op floor run": index.get("noop_floor"),
            "spend (USD)": (index.get("cost") or {}).get("spent_usd"),
            "cached calls": (index.get("cost") or {}).get("num_cached_calls"),
            "aborted on budget": index.get("aborted_on_budget"),
        }.items()
    )

    lat = np.array(latencies) if latencies else np.array([0.0])
    stats = {
        "real_mag": real_mag, "noop": noop,
        "image_vs_a": mean(all_v) if all_v else None, "null": null,
        "n_images": sum(1 for s in index["samples"]
                        for c in s["cameras"].values() if c.get("image")),
    }

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Image edit — {html.escape(str(index.get('model', '')))}</title><style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; max-width:1080px; margin:0 auto; padding:24px; line-height:1.45; }}
  .card {{ border:1px solid #8883; border-radius:10px; padding:12px 18px; margin:14px 0; }}
  .sit {{ border-top:3px solid #8885; margin-top:34px; padding-top:8px; }}
  .sit h3 {{ margin:6px 0; }} .dim {{ color:#888; font-weight:400; font-size:13px; }}
  .lbl {{ font-size:12px; color:#888; font-weight:600; margin:10px 0 4px; }}
  .lbl.real {{ color:#1565c0; }}
  .row {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; max-width:940px; margin-bottom:10px; }}
  figure {{ margin:0; }} figure img {{ width:100%; border-radius:6px; display:block; }}
  figcaption {{ font-size:11px; color:#888; text-align:center; margin-top:2px; }}
  .cond {{ border-left:4px solid #8886; border-radius:8px; padding:6px 14px; margin:14px 0; }}
  .cond.a {{ border-left-color:#1565c0; }} .cond.b {{ border-left-color:#8e24aa; }}
  .cond.c {{ border-left-color:#c62828; }} .cond.t {{ border-left-color:#ef6c00; }}
  .cond.n {{ border-left-color:#2e7d32; }}
  .cond h4 {{ margin:6px 0; }}
  .gen-item {{ margin:8px 0 14px; }} .instr {{ font-size:14px; margin-bottom:4px; }}
  .err {{ font-size:12px; color:#c62828; margin-bottom:4px; }}
  table {{ border-collapse:collapse; }} th,td {{ padding:2px 14px 2px 0; text-align:left; font-size:13px; }} th {{ color:#888; }}
  code {{ font-size:12px; background:#8881; padding:0 4px; border-radius:4px; }}
  .m {{ font-size:10px; background:#8881; border-radius:5px; padding:0 4px; font-weight:600; }}
  .m.r {{ background:#1565c022; color:#1565c0; }} .m.d {{ background:#8e24aa22; color:#8e24aa; }}
</style></head><body>
  <h1>Hosted image edit — subgoal per situation</h1>
  <p class="dim">{html.escape(str(index.get('backend')))} ·
     {html.escape(str(index.get('model')))} ·
     {stats['n_images']} images · latency mean {lat.mean():.1f}s ·
     spend ${fmt((index.get('cost') or {}).get('spent_usd'), 4)}</p>

  <div class="card"><h2>What you're looking at</h2>
    <p>Each <b>situation</b> is one timestep from the same set the Cosmos 3 and DreamZero
    harnesses use, with the same conditions, so these numbers can be read against theirs.
    We show the real 3-camera <b>source observation</b>, the <b>real future</b> at t+{k}
    (what actually happened — the yardstick), then the model's subgoal per condition.
    Unlike the world models there is <b>one image, not a rollout</b>: an editor jumps
    straight to the subgoal, so there is no trajectory to inspect and no action chunk.</p>
    <ul>
      <li><b style="color:#1565c0">A · sanity</b>: the episode's original instruction.</li>
      <li><b style="color:#8e24aa">B · counterfactual</b>: NEW instructions Gemini proposed from this scene.</li>
      <li><b style="color:#c62828">C · null / stress</b>: deliberately impossible instructions we authored.</li>
      <li><b style="color:#2e7d32">N · no-op</b>: told to change nothing — the re-render tax.</li>
    </ul>
    <p class="dim">Badges are mean absolute pixel difference (0–255), computed on the smaller
      of the two grids since the edit comes back much larger than the DROID frame:
      <span class="m">vs src</span> movement from the anchor frame ·
      <span class="m r">vs real</span> distance from what actually happened ·
      <span class="m d">vs A</span> how much swapping the instruction changed the subgoal.</p>
  </div>

  <div class="card"><h2>Calibration — what counts as a real effect</h2>
    <ul>
      {floor_line}
      <li><b>Real motion over the horizon: {fmt(real_mag)}</b> — real future vs source. A
          counterfactual subgoal <em>should not</em> match this; it is the scale, not the target.</li>
      {null_line}
    </ul>
    <table><tr><th>condition</th><th>n</th><th>image vs A</th></tr>{rows}</table>
    <p class="dim">There is no action column: an image editor produces no actions, so unlike
    DreamZero it can only ever be a Stage B candidate, never a combined B+C one.</p>
  </div>

  <div class="card"><h2>Run</h2><table>{meta}</table></div>

  {''.join(sections)}
</body></html>"""
    return doc, stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="runs")
    p.add_argument("--output", default=None)
    p.add_argument("--open", action="store_true")
    args = p.parse_args(argv)

    run_dir = HERE / "results" / args.run
    index = json.loads((run_dir / "index.json").read_text())
    doc, stats = build(run_dir, Path(index["situations_dir"]))
    out = Path(args.output) if args.output else run_dir / "report.html"
    out.write_text(doc)
    print(f"wrote {out}  ({stats['n_images']} images)")
    print(f"  no-op floor {fmt(stats['noop'])} · real motion {fmt(stats['real_mag'])} · "
          f"image vs A {fmt(stats['image_vs_a'])} · resample null {fmt(stats['null'])}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
