"""Build the DreamZero experiment report: a clear, grouped HTML contact sheet.

Runs in the MAIN venv (imageio + PIL). Reads results/runs/index.json and the
situation set. Layout, per situation (grouped, not scattered):

  <situation> — anchor t, original instruction
    Source observation .......... 3 real cameras in one row
    A · sanity (original) ....... instruction -> generated (split into 3 cams), 1-2 timestamps
    B · counterfactual (Gemini) .
    C · null / stress (authored).
    (+ any extra conditions, e.g. a targeted "move the pot" test)

The model's generated video is a 2x2 tiling of the DROID views; we SPLIT it back
into the 3 cameras using the confirmed layout (dreamzero_cotrain.py):
  top row (2x-wide) = wrist ; bottom-left = exterior_1 ; bottom-right = exterior_2.
Condition A also shows numeric joint-MAE vs the logged future (no plots).

    python explorations/dreamzero/make_report.py --situations <sit_dir>
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAMERAS = ("exterior_1", "exterior_2", "wrist")   # display order (source + generated)
# condition key -> (title, css-class); display in this order, skip if absent
COND = {
    "A": ("A · sanity (original instruction)", "a"),
    "B": ("B · counterfactual (Gemini-proposed)", "b"),
    "C": ("C · null / stress (authored: impossible tasks)", "c"),
    "MP": ("Targeted test — “move the pot”", "t"),
}
COND_ORDER = ["A", "B", "C", "MP"]


def _b64(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _arr_to_uri(arr: np.ndarray, width: int = 300) -> str:
    from PIL import Image
    im = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if im.width != width:
        im = im.resize((width, round(im.height * width / im.width)))
    buf = io.BytesIO(); im.save(buf, "PNG"); return _b64(buf.getvalue())


def _png_uri(path: Path, width: int = 300) -> str:
    import imageio.v2 as imageio
    return _arr_to_uri(np.asarray(imageio.imread(path)), width)


def split_grid(frame: np.ndarray) -> dict[str, np.ndarray]:
    """Split the model's 2x2 DROID grid into the 3 camera views.

    Layout (confirmed from groot .../transform/dreamzero_cotrain.py):
      top row (full width, wrist stretched 2x) = wrist
      bottom-left = exterior_1 ; bottom-right = exterior_2
    """
    H, W, _ = frame.shape
    h, w = H // 2, W // 2
    from PIL import Image
    wrist_wide = frame[:h, :]                        # (h, 2w) — undo the 2x width repeat
    wrist = np.asarray(Image.fromarray(wrist_wide.astype(np.uint8)).resize((w, h)))
    return {"exterior_1": frame[h:, :w], "exterior_2": frame[h:, w:], "wrist": wrist}


def gen_timestamps(mp4: Path, which=("mid", "last")):
    """Yield (label, {cam: array}) for 1-2 sampled generated frames, each split."""
    import imageio.v2 as imageio
    try:
        frames = list(imageio.mimread(mp4, memtest=False))
    except Exception:
        return
    if not frames:
        return
    n = len(frames)
    picks = {"mid": n // 2, "last": n - 1}
    for w in which:
        i = picks[w]
        yield f"frame {i + 1}/{n}", split_grid(np.asarray(frames[i]))


def cam_row(cam_imgs: dict[str, np.ndarray]) -> str:
    return "".join(
        f'<figure>{f"<img src={_arr_to_uri(cam_imgs[c])!r}>"}<figcaption>{c}</figcaption></figure>'
        for c in CAMERAS if c in cam_imgs
    )


def source_row(sdir: Path) -> str:
    cells = "".join(
        f'<figure><img src="{_png_uri(sdir / "history" / f"{c}_0.png")}"><figcaption>{c}</figcaption></figure>'
        for c in CAMERAS if (sdir / "history" / f"{c}_0.png").exists()
    )
    return f'<div class="row">{cells}</div>'


def joint_mae(pred, gt_joint, gt_grip) -> dict:
    H = min(pred.shape[0], gt_joint.shape[0])
    return {"joint_mae": float(np.mean(np.abs(pred[:H, :7] - gt_joint[:H]))),
            "gripper_mae": float(np.mean(np.abs(pred[:H, 7] - gt_grip[:H, 0])))}


def build(args):
    runs = Path(args.runs)
    sit_dir = Path(args.situations)
    index = json.loads((runs / "index.json").read_text())
    by_sit: dict[str, list[dict]] = {}
    for s in index["samples"]:
        by_sit.setdefault(s["situation_id"], []).append(s)

    calib, latencies, sections = [], [], []
    for sid, samples in by_sit.items():
        sdir = sit_dir / sid
        sit = json.loads((sdir / "situation.json").read_text())
        cond_blocks = []
        present = [c for c in COND_ORDER if any(s["condition"] == c for s in samples)]
        for cond in present:
            rows = []
            for s in [s for s in samples if s["condition"] == cond]:
                latencies.append(s.get("latency_s", 0.0))
                gen = runs / s["sample_id"] / (s.get("video") or "generated.mp4")
                ts_html = ""
                for lbl, cams in gen_timestamps(gen):
                    ts_html += f'<div class="cap">{lbl} — split into cameras</div><div class="row gen">{cam_row(cams)}</div>'
                if not ts_html:
                    ts_html = "<em>no video</em>"
                note = ""
                if cond == "A":
                    gt = np.load(sdir / "gt.npz")
                    m = joint_mae(np.load(runs / s["sample_id"] / "action_chunk.npy"),
                                  gt["future_joint"], gt["future_gripper"])
                    calib.append({"situation": sid, **m})
                    note = f' · <span class="cal">joint MAE {m["joint_mae"]:.3f} rad</span>'
                rows.append(
                    f'<div class="gen-item"><div class="instr">“{html.escape(s["instruction"])}”'
                    f'<span class="dim"> · action {s["action_shape"]} '
                    f'[{s["action_min"]:.2f},{s["action_max"]:.2f}] · {s["latency_s"]}s{note}</span></div>{ts_html}</div>')
            title, cls = COND[cond]
            cond_blocks.append(f'<div class="cond {cls}"><h4>{html.escape(title)}</h4>{"".join(rows)}</div>')

        sections.append(
            f'<section class="sit"><h3>{html.escape(sid)} '
            f'<span class="dim">· anchor t={sit["anchor"]} · original: “{html.escape(sit["instruction"])}”</span></h3>'
            f'<div class="lbl">source observation (real cameras)</div>{source_row(sdir)}'
            f'{"".join(cond_blocks)}</section>')

    calib_html = "<em>no A samples</em>"
    if calib:
        rows = "".join(f"<tr><td>{c['situation']}</td><td>{c['joint_mae']:.4f}</td>"
                       f"<td>{c['gripper_mae']:.4f}</td></tr>" for c in calib)
        mj = np.mean([c["joint_mae"] for c in calib])
        calib_html = (f"<table><tr><th>situation</th><th>joint MAE (rad)</th><th>gripper MAE</th></tr>{rows}"
                      f"<tr><td><b>mean</b></td><td><b>{mj:.4f}</b></td><td></td></tr></table>")
    lat = np.array(latencies) if latencies else np.array([0.0])

    out = Path(args.out) if args.out else (runs / "report.html")
    out.write_text(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>DreamZero — {html.escape(index.get('episode_id',''))}</title><style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; max-width:1080px; margin:0 auto; padding:24px; line-height:1.45; }}
  .card {{ border:1px solid #8883; border-radius:10px; padding:12px 18px; margin:14px 0; }}
  .sit {{ border-top:3px solid #8885; margin-top:34px; padding-top:8px; }}
  .sit h3 {{ margin:6px 0; }} .dim {{ color:#888; font-weight:400; font-size:13px; }}
  .lbl {{ font-size:12px; color:#888; font-weight:600; margin:10px 0 4px; }}
  .row {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; max-width:940px; }}
  .row.gen {{ margin-bottom:10px; }}
  figure {{ margin:0; }} figure img {{ width:100%; border-radius:6px; display:block; }}
  figcaption {{ font-size:11px; color:#888; text-align:center; margin-top:2px; }}
  .cond {{ border-left:4px solid #8886; border-radius:8px; padding:6px 14px; margin:14px 0; }}
  .cond.a {{ border-left-color:#1565c0; }} .cond.b {{ border-left-color:#8e24aa; }}
  .cond.c {{ border-left-color:#c62828; }} .cond.t {{ border-left-color:#ef6c00; }}
  .cond h4 {{ margin:6px 0; }}
  .gen-item {{ margin:8px 0 14px; }} .instr {{ font-size:14px; margin-bottom:4px; }}
  .cap {{ font-size:11px; color:#888; margin:4px 0 2px; }}
  .cal {{ color:#2e7d32; font-weight:600; }}
  table {{ border-collapse:collapse; }} th,td {{ padding:2px 14px 2px 0; text-align:left; font-size:13px; }} th {{ color:#888; }}
</style></head><body>
  <h1>DreamZero — subgoal-video + action, per situation</h1>
  <p class="dim">episode {html.escape(index.get('episode_id',''))} · {len(index['samples'])} generations · latency mean {lat.mean():.1f}s</p>
  <div class="card"><h2>What you're looking at</h2>
    <p>Each <b>situation</b> is one timestep. We show the real 3-camera <b>source observation</b>
    (one row), then the model's generated output per instruction condition. The model emits a 2×2
    multi-view video; we <b>split it back into the 3 cameras</b> (top=wrist, bottom-left=exterior_1,
    bottom-right=exterior_2) and show 1–2 timestamps.</p>
    <ul>
      <li><b style="color:#1565c0">A · sanity</b>: the episode's original instruction.</li>
      <li><b style="color:#8e24aa">B · counterfactual</b>: NEW instructions Gemini proposed from this scene.</li>
      <li><b style="color:#c62828">C · null / stress</b>: deliberately impossible instructions we authored.</li>
      <li><b style="color:#ef6c00">Targeted test</b>: the specific prompt “move the pot”.</li>
    </ul></div>
  <div class="card"><h2>Condition A — action calibration vs logged future</h2>{calib_html}
    <p class="dim">predicted (24,8) joint_position+gripper chunk vs the episode's real future over 24 steps.</p></div>
  {''.join(sections)}
</body></html>""")
    print(f"Wrote report ({len(index['samples'])} generations, {len(by_sit)} situations) to {out}")
    if calib:
        print(f"Condition A mean joint MAE: {np.mean([c['joint_mae'] for c in calib]):.4f} rad")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=str(HERE / "results" / "runs"))
    p.add_argument("--situations", required=True)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
