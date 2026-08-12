"""Build the Cosmos 3 experiment report — same layout as the DreamZero one.

Reads results/<run>/index.json and the shared DreamZero situation set. Per
situation (grouped, not scattered):

  <situation> — anchor t, original instruction
    Source observation .......... 3 real cameras in one row
    Real future (the yardstick) . the same 3 cameras at the matched timestamps
    A · sanity (original) ....... generated, 3 cameras per timestamp, 1-2 timestamps
    B · counterfactual (Gemini) .
    C · null / stress (authored).
    (+ any extra conditions, e.g. the targeted "move the pot" test)

Unlike DreamZero — which emits one 2x2 multi-view video that we split back into
3 cameras — Cosmos conditions on a single frame, so each camera is its own
generation. They are laid out in the same 3-up rows, but note they are
independent rollouts and nothing enforces cross-view consistency.

Where DreamZero's report carried a condition-A action calibration against the
logged future, this one cannot: the predicted actions are in a model-normalized
end-effector-delta space with no denormalization stats. The equivalent
calibration here is the seed null — how much the output moves when only the
random seed changes — which upper-bounds what an instruction effect must beat.

    .venv/bin/python make_report.py --run runs_allcams [--open]
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
CAMERAS = ("exterior_1", "exterior_2", "wrist")   # display order
COND = {
    "A": ("A · sanity (original instruction)", "a"),
    "B": ("B · counterfactual (Gemini-proposed)", "b"),
    "C": ("C · null / stress (authored: impossible tasks)", "c"),
    "MP": ("Targeted test — “move the pot”", "t"),
}
COND_ORDER = ["A", "B", "C", "MP"]


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


def _png(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _frames(path: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    try:
        return [np.asarray(f) for f in imageio.mimread(path, memtest=False)]
    except Exception:
        return []


def diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference, after matching a onto b's grid."""
    from PIL import Image

    if a.shape[:2] != b.shape[:2]:
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((b.shape[1], b.shape[0])))
    return float(np.abs(a.astype(float) - b.astype(float)).mean())


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
    # Show the midpoint and the end of whatever horizon this run used.
    chunk = int(index.get("chunk_size", 16))
    PICKS = (chunk // 2, chunk)

    by_sit: dict[str, list[dict]] = {}
    for s in index["samples"]:
        by_sit.setdefault(s["situation_id"], []).append(s)

    sections, latencies = [], []
    floors, real_mags, sens = [], [], []

    for sid in sorted(by_sit):
        sdir = sit_dir / sid
        sit = json.loads((sdir / "situation.json").read_text())
        samples = by_sit[sid]

        src = {c: _png(sdir / "history" / f"{c}_0.png") for c in cams_run
               if (sdir / "history" / f"{c}_0.png").exists()}

        # generated frames, per sample per camera
        gen: dict[str, dict[str, list[np.ndarray]]] = {}
        for s in samples:
            gen[s["sample_id"]] = {}
            for c in cams_run:
                info = s["cameras"].get(c)
                if info:
                    latencies.append(info.get("latency_s", 0.0))
                    gen[s["sample_id"]][c] = _frames(run_dir / s["sample_id"] / info["video"])

        a_sample = next((s for s in samples if s["condition"] == "A"), None)
        a_gen = gen.get(a_sample["sample_id"], {}) if a_sample else {}
        for c, fr in a_gen.items():
            if fr and c in src:
                floors.append(diff(fr[0], src[c]))

        # source + real-future rows
        blocks = [f'<div class="lbl">source observation (real cameras, anchor t={sit["anchor"]})</div>'
                  f'{cam_row(src)}']
        for i in PICKS:
            real = {c: _png(sdir / "future" / f"{c}_f{i:02d}.png") for c in cams_run
                    if (sdir / "future" / f"{c}_f{i:02d}.png").exists()}
            if not real:
                continue
            caps = {}
            for c, r in real.items():
                if c in src:
                    d = diff(r, src[c])
                    real_mags.append(d)
                    caps[c] = badge("vs src", d)
            blocks.append(f'<div class="lbl real">REAL future +{i} frames ({i / 15:.2f}s) — the yardstick</div>'
                          f'{cam_row(real, caps)}')

        # one block per condition
        present = [c for c in COND_ORDER if any(s["condition"] == c for s in samples)]
        present += sorted({s["condition"] for s in samples} - set(COND_ORDER))
        for cond in present:
            items = []
            for s in [x for x in samples if x["condition"] == cond]:
                ts_html = ""
                for i in PICKS:
                    cams_i, caps = {}, {}
                    for c in cams_run:
                        fr = gen[s["sample_id"]].get(c) or []
                        if i >= len(fr):
                            continue
                        cams_i[c] = fr[i]
                        cap = badge("vs src", diff(fr[i], src[c])) if c in src else ""
                        rp = sdir / "future" / f"{c}_f{i:02d}.png"
                        if rp.exists():
                            cap += badge("vs real", diff(fr[i], _png(rp)), "r")
                        af = a_gen.get(c) or []
                        if cond != "A" and i < len(af):
                            dv = diff(fr[i], af[i])
                            cap += badge("vs A", dv, "d")
                            if i == PICKS[-1]:
                                sens.append({"situation": sid, "condition": cond, "camera": c,
                                             "instruction": s["instruction"], "video_vs_a": dv})
                        caps[c] = cap
                    if cams_i:
                        ts_html += (f'<div class="cap">generated frame +{i} ({i / 15:.2f}s) — '
                                    f'one independent rollout per camera</div>{cam_row(cams_i, caps)}')
                if not ts_html:
                    ts_html = "<em>no video</em>"

                note = ""
                if cond != "A" and a_sample:
                    maes = []
                    seen_actions = set()
                    for c in cams_run:
                        name = (s["cameras"].get(c) or {}).get("action")
                        a_name = (a_sample["cameras"].get(c) or {}).get("action")
                        if not name or not a_name or name in seen_actions:
                            continue
                        seen_actions.add(name)   # concat runs share one action file
                        p = run_dir / s["sample_id"] / name
                        q = run_dir / a_sample["sample_id"] / a_name
                        if p.exists() and q.exists():
                            maes.append(float(np.abs(np.load(p) - np.load(q)).mean()))
                    if maes:
                        note = f' · <span class="cal">action MAE vs A {np.mean(maes):.3f}</span>'
                        for row in sens:
                            if row["situation"] == sid and row["condition"] == cond \
                                    and row["instruction"] == s["instruction"]:
                                row["action_mae_vs_a"] = float(np.mean(maes))

                shape = next((s["cameras"][c]["action_shape"] for c in cams_run
                              if s["cameras"].get(c)), None)
                items.append(
                    f'<div class="gen-item"><div class="instr">“{html.escape(s["instruction"])}”'
                    f'<span class="dim"> · action {shape}{note}</span></div>{ts_html}</div>')

            title, cls = COND.get(cond, (cond, "x"))
            blocks.append(f'<div class="cond {cls}"><h4>{html.escape(title)}</h4>{"".join(items)}</div>')

        sections.append(
            f'<section class="sit"><h3>{html.escape(sid)} '
            f'<span class="dim">· anchor t={sit["anchor"]} · original: “{html.escape(sit["instruction"])}”</span></h3>'
            f'{"".join(blocks)}</section>')

    # ---- calibration card ---------------------------------------------------
    floor = float(np.mean(floors)) if floors else float("nan")
    real_mag = float(np.mean(real_mags)) if real_mags else float("nan")
    seed_null = index.get("seed_null") or {}

    rows = ""
    for cond in COND_ORDER:
        vals = [r["video_vs_a"] for r in sens if r["condition"] == cond]
        acts = [r["action_mae_vs_a"] for r in sens if r["condition"] == cond and "action_mae_vs_a" in r]
        if not vals:
            continue
        title = COND[cond][0].split(" · ")[-1] if cond in COND else cond
        rows += (f"<tr><td><b>{cond}</b> {html.escape(title)}</td><td>{len(vals)}</td>"
                 f"<td>{np.mean(vals):.1f}</td>"
                 f"<td>{np.mean(acts):.3f}</td></tr>" if acts else
                 f"<tr><td><b>{cond}</b></td><td>{len(vals)}</td><td>{np.mean(vals):.1f}</td><td>—</td></tr>")
    all_v = [r["video_vs_a"] for r in sens]
    all_a = [r["action_mae_vs_a"] for r in sens if "action_mae_vs_a" in r]
    if all_v:
        rows += (f"<tr><td><b>overall</b></td><td>{len(all_v)}</td>"
                 f"<td><b>{np.mean(all_v):.1f}</b></td>"
                 f"<td><b>{np.mean(all_a):.3f}</b></td></tr>" if all_a else "")

    seed_line = ""
    if seed_null:
        seed_line = (f'<li><b>seed null: video {seed_null.get("video", float("nan")):.1f} · '
                     f'action {seed_null.get("action", float("nan")):.3f}</b> — same instruction, '
                     f'different random seed. An instruction effect below this is indistinguishable '
                     f'from re-rolling the dice.</li>')

    meta = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in {
            "model": index.get("model"), "action mode": index.get("action_mode"),
            "domain_name": index.get("domain_name"), "action space": index.get("action_space"),
            "cameras": ", ".join(cams_run), "fps": index.get("fps"),
            "chunk_size": index.get("chunk_size"), "resolution_tier": index.get("resolution_tier"),
            "inference steps": index.get("num_inference_steps"),
            "guidance_scale": index.get("guidance_scale"), "seed": index.get("seed"),
            "guardrails": index.get("guardrails"), "prompt upsampling": index.get("prompt_upsampling"),
        }.items()
    )

    lat = np.array(latencies) if latencies else np.array([0.0])
    stats = {"floor": floor, "real_mag": real_mag,
             "video_vs_a": float(np.mean(all_v)) if all_v else None,
             "action_vs_a": float(np.mean(all_a)) if all_a else None,
             "n_generations": len(index["samples"]) * len(cams_run)}

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Cosmos 3 — {html.escape(str(index.get('episode_id', '')))}</title><style>
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
  .cond h4 {{ margin:6px 0; }}
  .gen-item {{ margin:8px 0 14px; }} .instr {{ font-size:14px; margin-bottom:4px; }}
  .cap {{ font-size:11px; color:#888; margin:4px 0 2px; }}
  .cal {{ color:#ef6c00; font-weight:600; }}
  table {{ border-collapse:collapse; }} th,td {{ padding:2px 14px 2px 0; text-align:left; font-size:13px; }} th {{ color:#888; }}
  .m {{ font-size:10px; background:#8881; border-radius:5px; padding:0 4px; font-weight:600; }}
  .m.r {{ background:#1565c022; color:#1565c0; }} .m.d {{ background:#8e24aa22; color:#8e24aa; }}
</style></head><body>
  <h1>Cosmos 3 — subgoal video + action, per situation</h1>
  <p class="dim">episode {html.escape(str(index.get('episode_id', '')))} ·
     {len(index['samples'])} samples × {len(cams_run)} cameras =
     {len(index['samples']) * len(cams_run)} generations · latency mean {lat.mean():.1f}s</p>

  <div class="card"><h2>What you're looking at</h2>
    <p>Each <b>situation</b> is one timestep. We show the real 3-camera <b>source observation</b>,
    then the <b>real future</b> at the matched timestamps (what actually happened — the yardstick),
    then the model's generated output per instruction condition. Cosmos conditions on a
    <b>single frame</b>, so unlike DreamZero's 2×2 multi-view video each camera here is an
    <b>independent rollout</b>; they are laid out 3-up for comparison but nothing enforces
    cross-view consistency.</p>
    <ul>
      <li><b style="color:#1565c0">A · sanity</b>: the episode's original instruction.</li>
      <li><b style="color:#8e24aa">B · counterfactual</b>: NEW instructions Gemini proposed from this scene.</li>
      <li><b style="color:#c62828">C · null / stress</b>: deliberately impossible instructions we authored.</li>
      <li><b style="color:#ef6c00">Targeted test</b>: the specific prompt “move the pot”.</li>
    </ul>
    <p class="dim">Badges are mean absolute pixel difference (0–255):
      <span class="m">vs src</span> movement from the anchor frame ·
      <span class="m r">vs real</span> distance from what actually happened ·
      <span class="m d">vs A</span> how much swapping the instruction changed this rollout.</p>
  </div>

  <div class="card"><h2>Calibration — what counts as a real effect</h2>
    <ul>
      <li><b>VAE reconstruction floor: {floor:.1f}</b> — generated frame 0 vs the real source frame.
          The model cannot get closer to reality than this; treat it as zero.</li>
      <li><b>Real motion over the horizon: {real_mag:.1f}</b> — real future vs source.</li>
      {seed_line}
    </ul>
    <table><tr><th>condition</th><th>n</th><th>video vs A</th><th>action MAE vs A</th></tr>{rows}</table>
    <p class="dim">Condition A cannot be calibrated against logged DROID actions the way the DreamZero
    report was: these actions are 10D end-effector-pose deltas in a model-normalized space and the
    checkpoint ships no denormalization statistics.</p>
  </div>

  <div class="card"><h2>Run</h2><table>{meta}</table></div>

  {''.join(sections)}
</body></html>"""
    return doc, stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="runs_allcams")
    p.add_argument("--output", default=None)
    p.add_argument("--open", action="store_true")
    args = p.parse_args(argv)

    run_dir = HERE / "results" / args.run
    index = json.loads((run_dir / "index.json").read_text())
    doc, stats = build(run_dir, Path(index["situations_dir"]))
    out = Path(args.output) if args.output else run_dir / "report.html"
    out.write_text(doc)
    print(f"wrote {out}  ({stats['n_generations']} generations)")
    fmt = lambda v, p: "n/a" if v is None else f"{v:.{p}f}"  # noqa: E731
    print(f"  recon floor {fmt(stats['floor'], 1)} · real motion {fmt(stats['real_mag'], 1)} · "
          f"video vs A {fmt(stats['video_vs_a'], 1)} · action MAE vs A {fmt(stats['action_vs_a'], 3)}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
