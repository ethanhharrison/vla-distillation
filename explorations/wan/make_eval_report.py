"""Per-config A/B/C report — one situation per section, three views per row.

The cosmos3 report's layout, for the Wan eval grid:

  <situation> — anchor, original instruction
    source observation ......... 3 real cameras
    REAL future +k ............. the yardstick
    A · sanity ................. generated, 3 cameras
    B · counterfactual ......... one block per proposed instruction
    C · null / stress .......... one block per impossible instruction

Reads a config directory written by `run_eval.py`, scanning `*/sample.json`
rather than a shared index — jobs run concurrently across GPUs and never write a
common file, so the directory *is* the index.

    ../cosmos3/.venv/bin/python make_eval_report.py --config e_prefix --open
"""

from __future__ import annotations

import argparse
import html
import json
import webbrowser
from pathlib import Path

import numpy as np
from make_report import CAMERAS, DROID_FPS, badge, cam_row, diff, load_frames, load_rgb

HERE = Path(__file__).resolve().parent
COND = {
    "A": ("A · sanity (original instruction)", "a"),
    "B": ("B · counterfactual (Gemini-proposed)", "b"),
    "C": ("C · null / stress (authored: impossible tasks)", "c"),
    "MP": ("Targeted test — “move the pot”", "t"),
}
COND_ORDER = ["A", "B", "C", "MP"]


def load_samples(run_dir: Path) -> list[dict]:
    out = []
    for p in sorted(run_dir.glob("*/sample.json")):
        s = json.loads(p.read_text())
        s["_dir"] = p.parent
        out.append(s)
    return out


def build(run_dir: Path, k: int | None = None) -> tuple[str, dict]:
    cfg = json.loads((run_dir / "config.json").read_text())
    sit_dir = Path(cfg["situations_dir"])
    fps = cfg.get("fps", 16.0)
    k = k or cfg.get("k_droid", 32)
    samples = [s for s in load_samples(run_dir) if s["sample_id"] != "_seed_null"]
    if not samples:
        raise SystemExit(f"no samples in {run_dir}")

    by_sit: dict[str, list[dict]] = {}
    for s in samples:
        by_sit.setdefault(s["situation_id"], []).append(s)

    sections, floors, real_mags, sens = [], [], [], []
    for sid in sorted(by_sit):
        sdir = sit_dir / sid
        sit = json.loads((sdir / "situation.json").read_text())
        group = by_sit[sid]

        src = {c: load_rgb(sdir / "history" / f"{c}_0.png") for c in CAMERAS
               if (sdir / "history" / f"{c}_0.png").exists()}
        real = {c: load_rgb(sdir / "future" / f"{c}_f{k:02d}.png") for c in CAMERAS
                if (sdir / "future" / f"{c}_f{k:02d}.png").exists()}

        frames: dict[str, dict[str, list]] = {}
        for s in group:
            frames[s["sample_id"]] = {
                c: load_frames(s["_dir"] / rec["video"])
                for c, rec in s.get("camera_records", {}).items()
            }

        a = next((s for s in group if s["condition"] == "A"), None)
        a_frames = frames.get(a["sample_id"], {}) if a else {}
        for c, fr in a_frames.items():
            if fr and c in src:
                floors.append(diff(fr[0], src[c]))

        blocks = [f'<div class="lbl">source observation — anchor t={sit["anchor"]}</div>'
                  f'{cam_row(src)}']
        if real:
            caps = {}
            for c, r in real.items():
                if c in src:
                    d = diff(r, src[c])
                    real_mags.append(d)
                    caps[c] = badge("vs src", d)
            blocks.append(f'<div class="lbl real">REAL future +{k} ({k / DROID_FPS:.2f}s)'
                          f' — the yardstick</div>{cam_row(real, caps)}')

        present = [c for c in COND_ORDER if any(s["condition"] == c for s in group)]
        present += sorted({s["condition"] for s in group} - set(COND_ORDER))
        for cond in present:
            items = []
            for s in [x for x in group if x["condition"] == cond]:
                cams_i, caps = {}, {}
                for c in CAMERAS:
                    fr = frames[s["sample_id"]].get(c) or []
                    if not fr:
                        continue
                    j = min(round(k / DROID_FPS * fps), len(fr) - 1)
                    cams_i[c] = fr[j]
                    cap = badge("vs src", diff(fr[j], src[c])) if c in src else ""
                    if c in real:
                        cap += badge("vs real", diff(fr[j], real[c]), "r")
                    af = a_frames.get(c) or []
                    if cond != "A" and j < len(af):
                        dv = diff(fr[j], af[j])
                        cap += badge("vs A", dv, "d")
                        sens.append({"condition": cond, "camera": c, "value": dv})
                    caps[c] = cap
                body = cam_row(cams_i, caps) if cams_i else "<em>no clips</em>"
                err = "<div class='err'>generation failed</div>" if (s["_dir"] / "error.txt").exists() else ""
                items.append(f'<div class="gen-item"><div class="instr">'
                             f'“{html.escape(s["instruction"])}”</div>{err}{body}</div>')
            title, cls = COND.get(cond, (cond, "x"))
            blocks.append(f'<div class="cond {cls}"><h4>{html.escape(title)}</h4>'
                          f'{"".join(items)}</div>')

        sections.append(
            f'<section class="sit"><h3>{html.escape(sid)} '
            f'<span class="dim">· anchor t={sit["anchor"]} · original: '
            f'“{html.escape(sit["instruction"])}”</span></h3>{"".join(blocks)}</section>')

    # --- calibration ---------------------------------------------------------
    null = None
    np_dir = run_dir / "_seed_null"
    if (np_dir / "sample.json").exists() and a_frames:
        nrec = json.loads((np_dir / "sample.json").read_text())
        vals = []
        first = sorted(by_sit)[0]
        base = next((s for s in by_sit[first] if s["condition"] == "A"), None)
        if base:
            for c, rec in nrec.get("camera_records", {}).items():
                nf = load_frames(np_dir / rec["video"])
                bf = load_frames(base["_dir"] / f"generated_{c}.mp4")
                if nf and bf:
                    j = min(round(k / DROID_FPS * fps), len(nf) - 1, len(bf) - 1)
                    vals.append(diff(nf[j], bf[j]))
        null = float(np.mean(vals)) if vals else None

    def m(xs):
        return float(np.mean(xs)) if xs else float("nan")

    rows = ""
    for cond in COND_ORDER:
        vals = [x["value"] for x in sens if x["condition"] == cond]
        if vals:
            ratio = f"{m(vals) / null:.0%}" if null else "n/a"
            rows += (f"<tr><td><b>{cond}</b></td><td>{len(vals)}</td>"
                     f"<td>{m(vals):.1f}</td><td>{ratio}</td></tr>")
    allv = [x["value"] for x in sens]
    if allv:
        ratio = f"{m(allv) / null:.0%}" if null else "n/a"
        rows += (f"<tr><td><b>overall</b></td><td>{len(allv)}</td>"
                 f"<td><b>{m(allv):.1f}</b></td><td><b>{ratio}</b></td></tr>")
    null_line = (f"<li><b>seed null: {null:.1f}</b> — same instruction, seed "
                 f"{cfg.get('seed')} vs {cfg.get('seed', 0) + 1}. An instruction effect "
                 f"below this is indistinguishable from re-rolling the dice.</li>"
                 if null else
                 "<li><b>seed null: not measured</b> — sensitivity below is uninterpretable "
                 "until the <code>--seed-null</code> job has run.</li>")
    meta = "".join(f"<tr><td>{html.escape(str(kk))}</td><td>{html.escape(str(vv))}</td></tr>"
                   for kk, vv in cfg.items())

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Wan eval — {html.escape(run_dir.name)}</title><style>
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
  .cond h4 {{ margin:6px 0; }} .gen-item {{ margin:8px 0 14px; }}
  .instr {{ font-size:14px; margin-bottom:4px; }}
  .err {{ font-size:12px; color:#c62828; }}
  table {{ border-collapse:collapse; }} th,td {{ padding:3px 16px 3px 0; text-align:left; font-size:13px; }} th {{ color:#888; }}
  .m {{ font-size:10px; background:#8881; border-radius:5px; padding:0 4px; font-weight:600; }}
  .m.r {{ background:#1565c022; color:#1565c0; }} .m.d {{ background:#8e24aa22; color:#8e24aa; }}
</style></head><body>
  <h1>Wan eval — {html.escape(run_dir.name)}</h1>
  <p class="dim">{html.escape(str(cfg.get("prompt_template")))} ·
     {html.escape(str(cfg.get("view_mode")))} · cfg {cfg.get("guidance_scale")} ·
     {len(samples)} samples · {len(by_sit)} situations</p>

  <div class="card"><h2>What you're looking at</h2>
    <p>Each <b>situation</b> is one DROID scene at its first frame. Every row is one moment
    with the three cameras across it. We show the real source, the real future at +{k}
    (what actually happened), then the model's rollout per instruction condition.</p>
    <p class="dim">Badges: <span class="m">vs src</span> movement from the anchor ·
    <span class="m r">vs real</span> distance from what actually happened ·
    <span class="m d">vs A</span> how much swapping the instruction changed the rollout.</p>
  </div>

  <div class="card"><h2>Calibration — what counts as a real effect</h2>
    <ul>
      <li><b>VAE reconstruction floor: {m(floors):.1f}</b> — generated frame 0 vs the real
          source. The model cannot beat it; treat anything below as zero.</li>
      <li><b>Real motion over the horizon: {m(real_mags):.1f}</b> — averaged over cameras,
          which the wrist dominates. Read each camera against its own column.</li>
      {null_line}
    </ul>
    <table><tr><th>condition</th><th>n</th><th>vs A</th><th>% of null</th></tr>{rows}</table>
  </div>

  <div class="card"><h2>Config</h2><table>{meta}</table></div>
  {''.join(sections)}
</body></html>"""
    return doc, {"situations": len(by_sit), "samples": len(samples),
                 "floor": m(floors), "real": m(real_mags),
                 "sens": m(allv) if allv else float("nan"), "null": null}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--open", action="store_true")
    args = p.parse_args(argv)

    run_dir = HERE / "results" / args.config
    doc, st = build(run_dir, args.k)
    out = Path(args.output) if args.output else run_dir / "report.html"
    out.write_text(doc)
    nullstr = "n/a" if st["null"] is None else f"{st['null']:.1f}"
    print(f"wrote {out}  ({st['samples']} samples, {st['situations']} situations)")
    print(f"  floor {st['floor']:.1f} · real motion {st['real']:.1f} · "
          f"sens {st['sens']:.1f} · seed null {nullstr}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
