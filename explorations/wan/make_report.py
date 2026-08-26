"""Build the Wan rollout report — three views side by side, per timestamp.

Same layout as `explorations/cosmos3/make_report.py`: a row is one moment in
time, the three columns are the three cameras. That is the layout that makes
cross-view agreement judgeable — if the marker has moved in exterior_1 but not
in exterior_2, you see it in one glance, which is exactly the question the
concat-vs-independent comparison is asking.

Organised by *run* rather than by situation, because here the scene is fixed and
the varying axis is prompt x view mode.

The 16 fps model vs 15 Hz data mismatch is handled once: real DROID frame k is
generated frame round(k / 15 * 16), and both numbers appear in the caption.
Comparing frame k to frame k would quietly compare the wrong moments.

    ../cosmos3/.venv/bin/python make_report.py --runs v_prefix_cat v_prefix_ind --open
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

HERE = Path(__file__).resolve().parent
DROID_FPS = 15.0
#: Column order. The wrist moves ~9x more than the exteriors over this horizon,
#: so it sits last rather than mixed in.
CAMERAS = ("exterior_1", "exterior_2", "wrist")
#: Midpoint and end of the horizon, mirroring the cosmos3 report's two picks.
DEFAULT_PICKS = (16, 32)


def _b64(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _uri(arr: np.ndarray, width: int = 300) -> str:
    from PIL import Image

    im = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if im.width != width:
        im = im.resize((width, round(im.height * width / im.width)))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return _b64(buf.getvalue())


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def load_frames(path: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    try:
        return [np.asarray(f) for f in imageio.mimread(path, memtest=False)]
    except Exception:
        return []


def diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference on the smaller grid (see image_edit/metrics.py)."""
    from PIL import Image

    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    to = lambda x: (x if x.shape[:2] == (h, w) else np.asarray(  # noqa: E731
        Image.fromarray(x.astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS)))
    return float(np.abs(to(a).astype(float) - to(b).astype(float)).mean())


def badge(label: str, value: float, kind: str = "") -> str:
    return f' <span class="m {kind}">{html.escape(label)} <b>{value:.1f}</b></span>'


def cam_row(cams: dict[str, np.ndarray], captions: dict[str, str] | None = None) -> str:
    """One moment in time, the three cameras across."""
    captions = captions or {}
    cells = "".join(
        f'<figure><img src="{_uri(cams[c])}">'
        f'<figcaption>{html.escape(c)}{captions.get(c, "")}</figcaption></figure>'
        for c in CAMERAS if c in cams
    )
    return f'<div class="row">{cells}</div>'


def load_run(d: Path) -> dict | None:
    """Read a run in either layout: multi-camera `run.json`, or single `smoke.json`."""
    if (d / "run.json").exists():
        meta = json.loads((d / "run.json").read_text())
        cams = {}
        for cam, rec in meta.get("camera_records", {}).items():
            frames = load_frames(d / rec["video"])
            if frames:
                cams[cam] = frames
        return {"dir": d, "meta": meta, "cams": cams} if cams else None
    if (d / "smoke.json").exists():
        meta = json.loads((d / "smoke.json").read_text())
        frames = load_frames(d / "generated.mp4")
        if not frames:
            return None
        meta.setdefault("camera_records", {})
        return {"dir": d, "meta": meta, "cams": {meta["camera"]: frames}}
    return None


def build(run_dirs: list[Path], picks: tuple[int, ...]) -> tuple[str, dict]:
    runs = [r for r in (load_run(d) for d in run_dirs) if r is not None]
    if not runs:
        raise SystemExit("no readable runs")

    ref = runs[0]["meta"]
    sdir = Path(ref["situations_dir"]) / ref["situation"]
    sit = json.loads((sdir / "situation.json").read_text())
    all_cams = [c for c in CAMERAS if any(c in r["cams"] for r in runs)]

    src = {c: load_rgb(sdir / "history" / f"{c}_0.png") for c in all_cams
           if (sdir / "history" / f"{c}_0.png").exists()}
    real = {k: {c: load_rgb(sdir / "future" / f"{c}_f{k:02d}.png")
                for c in all_cams if (sdir / "future" / f"{c}_f{k:02d}.png").exists()}
            for k in picks}

    # --- reference: source, then the real future at each pick ---------------
    head = [f'<div class="lbl">source observation — anchor t={sit["anchor"]}</div>'
            f'{cam_row(src)}']
    for k in picks:
        if not real[k]:
            continue
        caps = {c: badge("vs src", diff(real[k][c], src[c])) for c in real[k] if c in src}
        head.append(f'<div class="lbl real">REAL future +{k} frames ({k / DROID_FPS:.2f}s)'
                    f' — the yardstick</div>{cam_row(real[k], caps)}')

    # --- one section per run, one cam_row per pick --------------------------
    sections = []
    for i, r in enumerate(runs):
        m = r["meta"]
        fps = m.get("fps", 16.0)
        mode = m.get("view_mode", "independent")

        blocks = ""
        for k in picks:
            cams_k, caps = {}, {}
            for c in all_cams:
                frames = r["cams"].get(c)
                if not frames:
                    continue
                j = min(round(k / DROID_FPS * fps), len(frames) - 1)
                cams_k[c] = frames[j]
                cap = badge("vs src", diff(frames[j], src[c])) if c in src else ""
                if c in real.get(k, {}):
                    cap += badge("vs real", diff(frames[j], real[k][c]), "r")
                caps[c] = cap
            if cams_k:
                j = min(round(k / DROID_FPS * fps), len(next(iter(r["cams"].values()))) - 1)
                blocks += (f'<div class="cap">generated f{j} = real t+{k} '
                           f'({k / DROID_FPS:.2f}s)</div>{cam_row(cams_k, caps)}')

        floors = ", ".join(f"{c} {m.get('camera_records', {}).get(c, {}).get('recon_floor', '?')}"
                           for c in all_cams if c in r["cams"])
        stat = (f'<span class="dim">mode <b>{mode}</b> · '
                f'{m.get("total_generate_s", m.get("generate_s"))}s · {m.get("steps")} steps · '
                f'cfg {m.get("guidance_scale")}'
                f'{"/" + str(m["guidance_scale_2"]) if m.get("guidance_scale_2") else ""} · '
                f'seed {m.get("seed")} · floor: {floors}</span>')

        sections.append(
            f'<section class="run {"cat" if mode == "concat" else "ind"}">'
            f'<h3>{html.escape(r["dir"].name)} '
            f'<span class="dim">· {html.escape(m.get("prompt_template", "?"))} · {mode}</span></h3>'
            f'<blockquote>{html.escape(m.get("prompt", m["instruction"]))}</blockquote>'
            f'{stat}{blocks}</section>')

    # --- calibration: per camera, since the views differ wildly -------------
    last = picks[-1]
    rows = ""
    for r in runs:
        recs = r["meta"].get("camera_records", {})
        cells = "".join(
            f"<td>{recs.get(c, {}).get('gen_vs_src', '—')}</td>" for c in all_cams)
        rows += (f'<tr><td>{html.escape(r["dir"].name)}</td>'
                 f'<td>{r["meta"].get("view_mode", "independent")}</td>{cells}</tr>')
    real_cells = "".join(
        f"<td><b>{diff(real[last][c], src[c]):.1f}</b></td>" if c in real.get(last, {}) and c in src
        else "<td>—</td>" for c in all_cams)
    rows += f'<tr class="ref"><td><b>real motion</b></td><td>—</td>{real_cells}</tr>'
    cam_head = "".join(f"<th>{html.escape(c)}</th>" for c in all_cams)
    real_motion = float(np.mean([diff(real[last][c], src[c]) for c in all_cams
                                 if c in real.get(last, {}) and c in src]))
    wrist_note = (
        f' The wrist column is the tell — reality moved '
        f'<b>{diff(real[last]["wrist"], src["wrist"]):.0f}</b> there.'
        if "wrist" in real.get(last, {}) and "wrist" in src else ""
    )

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Wan 2.2 — three views per timestamp</title><style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; max-width:1080px; margin:0 auto; padding:24px; line-height:1.45; }}
  .card {{ border:1px solid #8883; border-radius:10px; padding:12px 18px; margin:14px 0; }}
  .run {{ border-top:3px solid #8885; margin-top:32px; padding:8px 0 8px 14px; border-left:4px solid #8886; border-radius:8px; }}
  .run.ind {{ border-left-color:#c62828; }} .run.cat {{ border-left-color:#2e7d32; }}
  .run h3 {{ margin:6px 0; }} .dim {{ color:#888; font-weight:400; font-size:13px; }}
  blockquote {{ margin:6px 0; padding:8px 12px; background:#8881; border-radius:6px; font-size:13px; }}
  .lbl {{ font-size:12px; color:#888; font-weight:600; margin:12px 0 4px; }}
  .lbl.real {{ color:#1565c0; }}
  .cap {{ font-size:11px; color:#888; margin:10px 0 2px; }}
  .row {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; max-width:940px; margin-bottom:8px; }}
  figure {{ margin:0; }} figure img {{ width:100%; border-radius:6px; display:block; }}
  figcaption {{ font-size:11px; color:#888; text-align:center; margin-top:2px; }}
  table {{ border-collapse:collapse; }} th,td {{ padding:3px 16px 3px 0; text-align:left; font-size:13px; }} th {{ color:#888; }}
  tr.ref td {{ border-top:1px solid #8884; color:#1565c0; }}
  .m {{ font-size:10px; background:#8881; border-radius:5px; padding:0 4px; font-weight:600; }}
  .m.r {{ background:#1565c022; color:#1565c0; }}
</style></head><body>
  <h1>Wan2.2-I2V-A14B — three views per timestamp</h1>
  <p class="dim">{html.escape(ref["situation"])} · original task
     “{html.escape(sit["instruction"])}” · model {html.escape(str(ref.get("model")))} ·
     {len(runs)} runs · seed {ref.get("seed")}</p>

  <div class="card"><h2>What you're looking at</h2>
    <p>One DROID scene, one seed. Every row is <b>one moment in time with the three cameras
    across it</b>, so cross-view agreement is judgeable at a glance: if an object has moved
    in <code>exterior_1</code> but not in <code>exterior_2</code>, the row shows it.</p>
    <ul>
      <li><b style="color:#c62828">independent</b> — one generation per camera, same seed.
      Identical initial noise, but the rollouts share no state, so nothing makes them agree.</li>
      <li><b style="color:#2e7d32">concat</b> — one generation of the stitched three-view
      canvas, split back apart. One latent covers all three views, so they <i>can</i> cohere —
      and it is 3x cheaper. But the canvas is off-distribution for a general model, and each
      view gets a third of the pixels.</li>
    </ul>
    <p class="dim">Model runs at <b>16 fps</b>, DROID is <b>15 Hz</b>, so real frame <i>k</i>
    is generated frame <code>round(k/15*16)</code> — captions give both.
    Badges: <span class="m">vs src</span> movement from the anchor ·
    <span class="m r">vs real</span> distance from what actually happened.</p>
  </div>

  <div class="card"><h2>Calibration — read each camera against its own column</h2>
    <p class="dim">Cells are <span class="m">vs src</span> at t+{last}. The blue row is what
    reality did over the same span. The <b>wrist really moves ~9x more than the exteriors</b>
    here, so a pooled number across cameras is meaningless.</p>
    <table><tr><th>run</th><th>mode</th>{cam_head}</tr>{rows}</table>
    <p class="dim">Two readings fit a low concat number equally well and the pictures above
    are what separate them: the canvas genuinely constrains the model to a coherent scene, or
    it simply generates very little and scores well against a low-motion scene by accident.{wrist_note}</p>
  </div>

  {''.join(head)}
  {''.join(sections)}
</body></html>"""
    return doc, {"runs": len(runs), "real_motion": real_motion,
                 "images": (len(head) + len(runs) * len(picks)) * len(all_cams)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--picks", nargs="+", type=int, default=list(DEFAULT_PICKS),
                   help="Real DROID frame indices to show (default: 16 32).")
    p.add_argument("--output", default=None)
    p.add_argument("--open", action="store_true")
    args = p.parse_args(argv)

    dirs = [HERE / "results" / r for r in args.runs]
    missing = [d for d in dirs
               if not ((d / "run.json").exists() or (d / "smoke.json").exists())]
    if missing:
        raise SystemExit("no run.json or smoke.json in: "
                         + ", ".join(str(d) for d in missing))

    doc, stats = build(dirs, tuple(args.picks))
    out = Path(args.output) if args.output else HERE / "results" / "report.html"
    out.write_text(doc)
    print(f"wrote {out}  ({stats['runs']} runs, ~{stats['images']} images, "
          f"real motion {stats['real_motion']:.1f})")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
