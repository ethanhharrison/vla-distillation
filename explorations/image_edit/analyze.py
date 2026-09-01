"""Compare image-edit runs on edit magnitude, fidelity and instruction sensitivity.

The counterpart of `explorations/cosmos3/analyze_cfg.py`, reporting the subset of
its columns that survive having a single image instead of a clip.

Per run, over condition A (the only condition with a real answer to compare to):

  noop floor     |N - src| where condition N asked the model to change nothing.
                 The re-render tax: how far the image moves for reasons that
                 have nothing to do with the instruction. Present only for runs
                 built with --noop; treat everything below it as zero.
  edit magnitude |A - src|. How far the edit actually moved the scene.
  real motion    |real[t+k] - src|. What the scene really did over the same
                 span — the yardstick an edit should be measured against, not a
                 target to match, since a counterfactual should NOT match it.
  fidelity       |A - real[t+k]|. Only meaningful for A, where the real future
                 is the right answer.

And, across conditions, the thing the whole exercise is about:

  sensitivity    |A - X| for each counterfactual condition X, against that run's
                 own resample null. Below the null, the instruction moved the
                 output less than asking twice does, and means nothing.

    ../../.venv/bin/python analyze.py --runs gemini_flash_3_1 openai_gpt_image_2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import diff, fmt, load_rgb, mean

HERE = Path(__file__).resolve().parent
#: Conditions that are counterfactuals of A. N is the no-op floor, not a
#: counterfactual, so it is scored separately and never folded into sensitivity.
COUNTERFACTUAL = ("B", "C", "MP")


def score(run_dir: Path) -> dict:
    index = json.loads((run_dir / "index.json").read_text())
    sit_dir = Path(index["situations_dir"])
    cameras = index["cameras"]
    k = index.get("k") or index.get("horizon")

    by_sit: dict[str, list[dict]] = {}
    for s in index["samples"]:
        by_sit.setdefault(s["situation_id"], []).append(s)

    noop, edit_mag, real_motion, fidelity = [], [], [], []
    sens: dict[str, list[float]] = {c: [] for c in COUNTERFACTUAL}
    latencies, phashes = [], []

    for sid, group in by_sit.items():
        sdir = sit_dir / sid
        a = next((s for s in group if s["condition"] == "A"), None)
        if a is None:
            continue
        for cam in cameras:
            src_p = sdir / "history" / f"{cam}_0.png"
            a_info = a["cameras"].get(cam) or {}
            if not src_p.exists() or not a_info.get("image"):
                continue
            src = load_rgb(src_p)
            a_img = load_rgb(run_dir / a["sample_id"] / a_info["image"])

            edit_mag.append(diff(a_img, src))
            if a_info.get("latency_s") is not None:
                latencies.append(a_info["latency_s"])
            if a_info.get("phash_norm") is not None:
                phashes.append(a_info["phash_norm"])

            real_p = sdir / "future" / f"{cam}_f{k:02d}.png"
            if real_p.exists():
                real = load_rgb(real_p)
                real_motion.append(diff(real, src))
                fidelity.append(diff(a_img, real))

            for s in group:
                info = s["cameras"].get(cam) or {}
                if not info.get("image") or s["condition"] == "A":
                    continue
                other = load_rgb(run_dir / s["sample_id"] / info["image"])
                if s["condition"] == "N":
                    noop.append(diff(other, src))
                elif s["condition"] in sens:
                    sens[s["condition"]].append(diff(other, a_img))

    all_sens = [v for c in COUNTERFACTUAL for v in sens[c]]
    null = (index.get("resample_null") or {}).get("image")
    cost = index.get("cost") or {}
    return {
        "run": run_dir.name,
        "model": index.get("model"),
        "n": len(edit_mag),
        "noop": mean(noop) if noop else None,
        "edit_mag": mean(edit_mag),
        "real_motion": mean(real_motion),
        "fidelity": mean(fidelity),
        "sens": mean(all_sens),
        "sens_b": mean(sens["B"]),
        "sens_c": mean(sens["C"]),
        "null": null,
        "ratio": (mean(all_sens) / null) if null else None,
        "phash": mean(phashes),
        "latency": mean(latencies),
        "spent_usd": cost.get("spent_usd"),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args(argv)

    rows = [score(HERE / "results" / r) for r in args.runs]

    header = (f"{'run':22} {'model':26} {'n':>3} {'noop':>5} {'edit':>6} {'real':>6} "
              f"{'fidel':>6} {'sensB':>6} {'sensC':>6} {'null':>6} {'s/null':>7} {'$':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        ratio = "n/a" if r["ratio"] is None else f"{r['ratio']:.0%}"
        spent = "n/a" if r["spent_usd"] is None else f"{r['spent_usd']:.2f}"
        print(f"{r['run'][:22]:22} {str(r['model'])[:26]:26} {r['n']:3d} "
              f"{fmt(r['noop']):>5} {fmt(r['edit_mag']):>6} {fmt(r['real_motion']):>6} "
              f"{fmt(r['fidelity']):>6} {fmt(r['sens_b']):>6} {fmt(r['sens_c']):>6} "
              f"{fmt(r['null']):>6} {ratio:>7} {spent:>7}")

    print("\nall numbers are mean |a-b| on 0-255 RGB, comparable to the cosmos3/dreamzero tables")
    print("noop  = |change-nothing edit - source|: the re-render tax. Below it means nothing.")
    print("edit  = |A - source|, real = |real t+k - source|, fidel = |A - real t+k|")
    print("sens  = |A - counterfactual|; s/null = sens over this run's own resample null")
    for r in rows:
        if r["noop"] is None:
            print(f"  ! {r['run']}: no --noop floor measured; `edit` includes an "
                  f"unmeasured re-render tax.")
        if r["null"] is None:
            print(f"  ! {r['run']}: no resample null recorded; sensitivity is "
                  f"uninterpretable. Run resample_null.py --run {r['run']}.")


if __name__ == "__main__":
    main()
