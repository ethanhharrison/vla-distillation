"""Side-by-side PNG strip comparing runs on one situation — for eyeballing artifacts.

The numeric metrics answer "how far from reality" but say nothing about *how* a
frame is wrong: a duplicated object, a morphing gripper and a uniform colour
shift can all score the same. This lays the real future and each run's rollout
on a grid so the failure mode is visible.

Rows are REAL then one per run; columns are timesteps.

    .venv/bin/python compare_strip.py --runs cfgsweep_1 cfgsweep_2 runs_multitraj \\
        --situation ep003_t0000 --camera exterior_1 --frames 8 16 24 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TILE_W, LABEL_H = 300, 16


def _png(p: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(p).convert("RGB"))


def _frames(p: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    try:
        return [np.asarray(f) for f in imageio.mimread(p, memtest=False)]
    except Exception:
        return []


def main(argv=None) -> None:
    from PIL import Image, ImageDraw

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--situation", required=True)
    p.add_argument("--camera", default="exterior_1")
    p.add_argument("--condition", default="A")
    p.add_argument("--frames", type=int, nargs="+", default=[8, 16, 24, 32])
    p.add_argument("--out", default=None)
    p.add_argument("--scale", type=int, default=2)
    args = p.parse_args(argv)

    rows: list[tuple[str, dict[int, np.ndarray]]] = []

    # real future (plus the source frame in the leading column)
    first_index = json.loads((HERE / "results" / args.runs[0] / "index.json").read_text())
    sit_dir = Path(first_index["situations_dir"]) / args.situation
    src = _png(sit_dir / "history" / f"{args.camera}_0.png")
    real = {}
    for t in args.frames:
        rp = sit_dir / "future" / f"{args.camera}_f{t:02d}.png"
        if rp.exists():
            real[t] = _png(rp)
    rows.append(("REAL future", real))

    for run in args.runs:
        run_dir = HERE / "results" / run
        index = json.loads((run_dir / "index.json").read_text())
        sample = next((s for s in index["samples"]
                       if s["situation_id"] == args.situation and s["condition"] == args.condition), None)
        if sample is None:
            print(f"  {run}: no {args.condition} sample for {args.situation}, skipping")
            continue
        fr = _frames(run_dir / sample["sample_id"] / f"generated_{args.camera}.mp4")
        label = f"{run}  cfg={index.get('guidance_scale')} shift={index.get('flow_shift')}"
        rows.append((label, {t: fr[t] for t in args.frames if t < len(fr)}))

    h = round(src.shape[0] * TILE_W / src.shape[1])
    ncol = len(args.frames) + 1
    img = Image.new("RGB", (TILE_W * ncol, (h + LABEL_H) * len(rows)), (18, 18, 18))
    d = ImageDraw.Draw(img)
    for r, (label, tiles) in enumerate(rows):
        y = r * (h + LABEL_H)
        d.text((4, y + 3), label, fill=(255, 255, 255))
        if r == 0:
            img.paste(Image.fromarray(src.astype(np.uint8)).resize((TILE_W, h)), (0, y + LABEL_H))
            d.text((6, y + LABEL_H + 2), "source t=0", fill=(255, 220, 0))
        for ci, t in enumerate(args.frames, start=1):
            if t in tiles:
                img.paste(Image.fromarray(tiles[t].astype(np.uint8)).resize((TILE_W, h)),
                          (ci * TILE_W, y + LABEL_H))
                d.text((ci * TILE_W + 6, y + LABEL_H + 2), f"+{t}", fill=(255, 220, 0))

    out = Path(args.out) if args.out else HERE / "results" / f"strip_{args.situation}_{args.camera}.png"
    img.resize((img.width * args.scale, img.height * args.scale)).save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
