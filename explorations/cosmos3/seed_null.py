"""Measure the seed null for a finished run and record it in its index.json.

The control that makes every other number readable: re-run condition A of the
first situation with the SAME instruction and a DIFFERENT seed, then measure how
far the output moved. Any instruction effect smaller than this is
indistinguishable from re-rolling the dice.

Reuses the run's own settings out of index.json so the null is measured under
exactly the config it will be compared against.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python seed_null.py --run runs_allcams
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="runs_allcams")
    p.add_argument("--alt-seed", type=int, default=None, help="Defaults to the run's seed + 1.")
    p.add_argument("--frame", type=int, default=16, help="Generated frame index to compare.")
    args = p.parse_args(argv)

    import imageio.v2 as imageio
    from diffusers.utils import load_image

    from run_experiment import VIEWPOINT, build_pipe, policy_call, save_sample

    run_dir = HERE / "results" / args.run
    index = json.loads((run_dir / "index.json").read_text())
    sit_dir = Path(index["situations_dir"])
    cams = index["cameras"]
    alt_seed = args.alt_seed if args.alt_seed is not None else index["seed"] + 1

    a = next(s for s in index["samples"] if s["condition"] == "A")
    sdir = sit_dir / a["situation_id"]
    out = run_dir / "_seed_null"

    # Re-use the run's sampling settings verbatim.
    call_args = argparse.Namespace(
        chunk_size=index["chunk_size"], resolution_tier=index["resolution_tier"],
        fps=index["fps"], steps=index["num_inference_steps"],
        guidance_scale=index["guidance_scale"], seed=alt_seed,
        # Anything that changes the OUTPUT GEOMETRY must be carried over, or the
        # re-run is generated at a different size and cannot be diffed.
        upscale=index.get("upscale", 1),
    )
    # Concat-view runs generate one canvas per instruction and store a single
    # action_concat.npy, so the per-camera path below does not apply to them.
    concat = bool(index.get("concat_view"))

    pipe = build_pipe(index["model_path"], flow_shift=index.get("flow_shift"))
    vids, acts = [], []
    for cam in cams:
        if concat:
            from run_experiment import CANVAS_CAMERAS, build_canvas, split_canvas
            from PIL import Image as _Image
            if cam == cams[0]:   # one canvas call covers all three views
                canvas = build_canvas({c: sdir / "history" / f"{c}_0.png" for c in CANVAS_CAMERAS})
                _frames, action, dt = policy_call(pipe, canvas, a["instruction"], "concat", call_args)
                _split = {c: [_Image.fromarray(split_canvas(np.asarray(f))[c]) for f in _frames]
                          for c in CANVAS_CAMERAS}
            frames = _split[cam]
            save_sample(out, cam, frames, None, index["fps"])
        else:
            image = load_image(str(sdir / "history" / f"{cam}_0.png"))
            frames, action, dt = policy_call(pipe, image, a["instruction"], cam, call_args)
            save_sample(out, cam, frames, action, index["fps"])
        print(f"  {cam}: seed {alt_seed} re-run, {len(frames)}f, {dt:.1f}s")

        orig = [np.asarray(f) for f in imageio.mimread(run_dir / a["sample_id"] / f"generated_{cam}.mp4",
                                                       memtest=False)]
        alt = [np.asarray(f) for f in imageio.mimread(out / f"generated_{cam}.mp4", memtest=False)]
        i = min(args.frame, len(orig) - 1, len(alt) - 1)
        f_orig, f_alt = orig[i], alt[i]
        if f_orig.shape != f_alt.shape:   # belt and braces if a geometry flag is missed
            from PIL import Image
            f_alt = np.asarray(Image.fromarray(f_alt.astype(np.uint8))
                               .resize((f_orig.shape[1], f_orig.shape[0])))
        vids.append(float(np.abs(f_orig.astype(float) - f_alt.astype(float)).mean()))
        if action is not None:
            name = (a["cameras"].get(cam) or {}).get("action") or f"action_{cam}.npy"
            a0_path = run_dir / a["sample_id"] / name
            if a0_path.exists():
                acts.append(float(np.abs(np.load(a0_path) - action).mean()))

    index["seed_null"] = {
        "situation": a["situation_id"], "instruction": a["instruction"],
        "seed_a": index["seed"], "seed_b": alt_seed, "frame": args.frame,
        "video": float(np.mean(vids)), "action": float(np.mean(acts)) if acts else None,
        "per_camera_video": {c: round(v, 2) for c, v in zip(cams, vids)},
    }
    (run_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nseed null (same instruction, seed {index['seed']} vs {alt_seed}): "
          f"video {index['seed_null']['video']:.1f}  action {index['seed_null']['action']:.3f}")
    print(f"recorded in {run_dir}/index.json")


if __name__ == "__main__":
    main()
