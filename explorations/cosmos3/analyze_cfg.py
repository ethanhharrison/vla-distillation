"""Compare runs on fidelity vs instruction sensitivity — the classifier-free-guidance tradeoff.

Higher CFG pushes the sample harder toward the text, which is what makes the
instruction matter at all — but it can also push the sample off the image
manifold, showing up as objects that drift, duplicate or morph. This scores both
sides so the tradeoff is visible instead of assumed.

Per run, over condition A (where a real future exists to compare against):

  recon floor    generated frame 0 vs the real source frame. Pure autoencoder
                 loss; the model cannot beat this.
  fidelity       mean |gen[t] - real[t]| over the whole clip, and at the end.
                 Lower = closer to what actually happened.
  gen jitter     mean |f[t+1] - f[t]| inside the generated clip.
  real jitter    the same statistic on the real clip.
  jitter ratio   gen/real. ~1 = moves like reality. >>1 = erratic, flickering,
                 or hallucinated motion. <<1 = frozen.
  drift          |gen[t] - real[t]| at the end minus at the start: does error
                 accumulate over the rollout?

And, over conditions A vs C, the thing we would be trading away:

  sensitivity    mean |gen_A[H] - gen_C[H]|, i.e. how much the instruction moves
                 the rollout, against that run's own seed null where recorded.

    .venv/bin/python analyze_cfg.py --runs cfgsweep_1 cfgsweep_2 runs_multitraj
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _png(p: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(p).convert("RGB"))


def _frames(p: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    try:
        return [np.asarray(f) for f in imageio.mimread(p, memtest=False)]
    except Exception:
        return []


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    from PIL import Image

    if a.shape[:2] != b.shape[:2]:
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((b.shape[1], b.shape[0])))
    return float(np.abs(a.astype(float) - b.astype(float)).mean())


def _jitter(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return float("nan")
    return float(np.mean([np.abs(frames[i + 1].astype(float) - frames[i].astype(float)).mean()
                          for i in range(len(frames) - 1)]))


def score(run_dir: Path) -> dict:
    index = json.loads((run_dir / "index.json").read_text())
    sit_dir = Path(index["situations_dir"])
    cams = index["cameras"]
    H = int(index["chunk_size"])

    by_sit: dict[str, list[dict]] = {}
    for s in index["samples"]:
        by_sit.setdefault(s["situation_id"], []).append(s)

    floor, fid_mean, fid_end, gj, rj, drift = [], [], [], [], [], []
    sens = {"B": [], "C": []}   # plausible counterfactual vs impossible, kept apart
    for sid, group in by_sit.items():
        sdir = sit_dir / sid
        a = next((s for s in group if s["condition"] == "A"), None)
        if a is None:
            continue
        for cam in cams:
            src_p = sdir / "history" / f"{cam}_0.png"
            if not src_p.exists():
                continue
            src = _png(src_p)
            af = _frames(run_dir / a["sample_id"] / f"generated_{cam}.mp4")
            if not af:
                continue

            real = []
            for t in range(1, min(H, len(af) - 1) + 1):
                rp = sdir / "future" / f"{cam}_f{t:02d}.png"
                if rp.exists():
                    real.append((t, _png(rp)))
            if not real:
                continue

            floor.append(_diff(af[0], src))
            per_t = [_diff(af[t], r) for t, r in real if t < len(af)]
            if per_t:
                fid_mean.append(st.mean(per_t))
                fid_end.append(per_t[-1])
                drift.append(per_t[-1] - per_t[0])
            gj.append(_jitter(af[: len(real) + 1]))
            rj.append(_jitter([src] + [r for _, r in real]))

            for s in group:
                if s["condition"] not in sens:
                    continue
                cf = _frames(run_dir / s["sample_id"] / f"generated_{cam}.mp4")
                if cf and H < len(cf) and H < len(af):
                    sens[s["condition"]].append(_diff(af[H], cf[H]))

    m = lambda xs: st.mean(xs) if xs else float("nan")  # noqa: E731
    return {
        "run": run_dir.name,
        "cfg": index.get("guidance_scale"),
        "n": len(floor),
        "floor": m(floor), "fid_mean": m(fid_mean), "fid_end": m(fid_end),
        "gen_jitter": m(gj), "real_jitter": m(rj),
        "jitter_ratio": m(gj) / m(rj) if rj and m(rj) else float("nan"),
        "drift": m(drift), "sensitivity": m(sens["B"] + sens["C"]),
        "sens_b": m(sens["B"]), "sens_c": m(sens["C"]),
        "seed_null": (index.get("seed_null") or {}).get("video"),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args(argv)

    rows = [score(HERE / "results" / r) for r in args.runs]
    rows.sort(key=lambda r: (r["cfg"] is None, r["cfg"]))

    print(f"{'run':18} {'CFG':>5} {'floor':>6} {'fidelity':>9} {'fid@end':>8} "
          f"{'jitter g/r':>16} {'drift':>7} {'sensB':>6} {'sensC':>6}")
    for r in rows:
        b = "  n/a" if r["sens_b"] != r["sens_b"] else f"{r['sens_b']:6.1f}"
        print(f"{r['run']:18} {r['cfg']:5.1f} {r['floor']:6.1f} {r['fid_mean']:9.1f} "
              f"{r['fid_end']:8.1f} {r['gen_jitter']:6.2f}/{r['real_jitter']:<4.2f}={r['jitter_ratio']:4.2f} "
              f"{r['drift']:7.1f} {b} {r['sens_c']:6.1f}")
    print("\nfidelity = mean |gen - real| over the clip (lower is better)")
    print("jitter ratio ~1 means the rollout moves like reality; >>1 is erratic motion")
    print("sens = |A - C| at the last frame — what you give up by lowering CFG")
    for r in rows:
        if r["seed_null"]:
            print(f"  {r['run']}: seed null {r['seed_null']:.1f} "
                  f"(sensitivity is {r['sensitivity'] / r['seed_null']:.0%} of it)")


if __name__ == "__main__":
    main()
