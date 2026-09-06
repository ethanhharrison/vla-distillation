"""Measure the resample null for a finished run and record it in its index.json.

The control that makes every other number readable, and the direct analogue of
`explorations/cosmos3/seed_null.py`. There, the control re-runs condition A with
a different random seed. A hosted image API exposes no seed and resamples on
every call, so here the control is simply **the same request, asked twice**. Any
instruction effect smaller than this is indistinguishable from asking the same
question again.

Reuses the run's own settings out of index.json, so the null is measured under
exactly the configuration it will be compared against.

**This deliberately bypasses the edit cache.** An identical request is exactly
what the cache is built to serve for free, so a cached null would report a
difference of 0.0 and silently turn the control into a rubber stamp that every
instruction effect clears. Bypassing means the null is the one part of a rerun
that is never free — `--repeats` calls per camera, at the run's per-image price.

    ../../.venv/bin/python resample_null.py --run runs --ceiling 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from metrics import diff, load_rgb, load_rgb_bytes, mean  # noqa: E402

from pipeline.subgoal_image import prompts  # noqa: E402
from pipeline.subgoal_image.backends import build_image_backend, is_paid_backend  # noqa: E402
from pipeline.subgoal_image.cost import CostTracker  # noqa: E402
from pipeline.subgoal_image.edit import edit_camera  # noqa: E402

import canvas as canvas_mod  # noqa: E402


def rebuild_backend(index: dict, quality: str | None = None):
    """Rebuild the run's backend from its index.json.

    Anything that changes the produced image must be carried over or the re-run
    is not a null of the same configuration. The cosmos3 harness learned this
    the hard way: its `seed_null.py` silently mismatched geometry when a new
    flag was added and did not reach here.
    """
    kwargs: dict = {"model": index["model"]}
    if index["backend"] == "openai_image":
        oa = index.get("openai") or {}
        kwargs["quality"] = quality or oa.get("quality", "low")
        kwargs["size"] = oa.get("size", "auto")
    return build_image_backend(index["backend"], **kwargs)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="runs")
    p.add_argument("--repeats", type=int, default=1,
                   help="Extra samples per camera. >1 averages the null over more "
                        "draws, which matters when the effect sits near it.")
    p.add_argument("--ceiling", type=float, default=1.0, help="Hard $ ceiling for this measurement.")
    args = p.parse_args(argv)

    run_dir = HERE / "results" / args.run
    index = json.loads((run_dir / "index.json").read_text())
    sit_dir = Path(index["situations_dir"])
    cameras = index["cameras"]

    sample = next((s for s in index["samples"]
                   if s["condition"] == "A" and s["cameras"]), None)
    if sample is None:
        raise SystemExit("no condition-A sample with images in this run — nothing to null against.")
    sdir = sit_dir / sample["situation_id"]
    _, tpl_text = prompts.resolve_template(index["prompt_template"] or prompts.DEFAULT_TEMPLATE)
    prompt_text = prompts.build_prompt(tpl_text, sample["instruction"])

    # A canvas run's null must re-issue the CANVAS request, not three per-frame
    # edits: the quantity being controlled for is the variance of the call the run
    # actually made, and a single-frame null would be a different experiment's
    # control silently pasted into this one.
    is_canvas = index.get("strategy") == "canvas"
    layout = index.get("canvas_layout")
    if is_canvas:
        prompt_text = canvas_mod.canvas_prompt(prompt_text)

    backend = rebuild_backend(index)
    n_calls = args.repeats if is_canvas else len(cameras) * args.repeats
    if is_paid_backend(index["backend"]):
        projected = n_calls * backend.estimate_cost()
        print(f"resample null: {n_calls} uncached edits, projected <= ${projected:.2f}")
        if projected > args.ceiling:
            raise SystemExit(
                f"ABORT: projected ${projected:.2f} exceeds --ceiling ${args.ceiling:.2f}."
            )

    tracker = CostTracker(ceiling_usd=args.ceiling, costs_path=run_dir / "costs.jsonl")
    out = run_dir / "_resample_null"
    out.mkdir(parents=True, exist_ok=True)

    per_camera: dict[str, float] = {}

    if is_canvas:
        srcs = {c: (sdir / "history" / f"{c}_0.png").read_bytes() for c in cameras
                if (sdir / "history" / f"{c}_0.png").exists()}
        canvas_bytes, _, _ = canvas_mod.build(layout, srcs)
        originals = {
            c: load_rgb(run_dir / sample["sample_id"] / sample["cameras"][c]["image"])
            for c in cameras
            if (sample["cameras"].get(c) or {}).get("image")
        }
        deltas: dict[str, list[float]] = {}
        for r in range(args.repeats):
            res = canvas_mod.edit_canvas(
                backend_name=index["backend"], backend=backend, layout=layout,
                prompt=prompt_text, canvas_bytes=canvas_bytes,
                cache=None, tracker=tracker, use_cache=False,  # never cached — see docstring
                example_id=f"_resample_null/canvas:{layout}",
            )
            if res["error"]:
                print(f"  ! canvas repeat {r}: {res['error']}")
                continue
            (out / f"canvas_r{r}.png").write_bytes(res["image"])
            panels = canvas_mod.split(layout, load_rgb_bytes(res["image"]))
            for c, orig in originals.items():
                deltas.setdefault(c, []).append(diff(orig, panels[c]))
        for c, vals in deltas.items():
            per_camera[c] = mean(vals)
            print(f"  {c}: resample null {per_camera[c]:.1f} over {len(vals)} draw(s)")
        cameras = []   # skip the per-frame path below

    for cam in cameras:
        info = sample["cameras"].get(cam)
        src = sdir / "history" / f"{cam}_0.png"
        if not info or not info.get("image") or not src.exists():
            continue
        original = load_rgb(run_dir / sample["sample_id"] / info["image"])
        src_bytes = src.read_bytes()

        deltas = []
        for r in range(args.repeats):
            outcome = edit_camera(
                backend_name=index["backend"], backend=backend, camera=cam,
                source_bytes=src_bytes, instruction=sample["instruction"],
                prompt=prompt_text, k=index.get("k"),
                cache=None, tracker=tracker, use_cache=False,   # never cached — see module docstring
                example_id=f"_resample_null/{cam}",
            )
            if outcome.over_budget:
                raise SystemExit(f"BUDGET CEILING HIT: {outcome.error}")
            if not outcome.ok:
                print(f"  ! {cam} repeat {r}: {outcome.error}")
                continue
            (out / f"subgoal_{cam}_r{r}.{outcome.ext}").write_bytes(outcome.image_bytes)
            deltas.append(diff(original, load_rgb_bytes(outcome.image_bytes)))
        if deltas:
            per_camera[cam] = mean(deltas)
            print(f"  {cam}: resample null {per_camera[cam]:.1f} over {len(deltas)} draw(s)")

    if not per_camera:
        raise SystemExit("no resample succeeded; index.json left unchanged.")

    index["resample_null"] = {
        "strategy": index.get("strategy", "independent"),
        "canvas_layout": layout if is_canvas else None,
        "situation": sample["situation_id"],
        "instruction": sample["instruction"],
        "repeats": args.repeats,
        "image": mean(list(per_camera.values())),
        "per_camera": {c: round(v, 2) for c, v in per_camera.items()},
        "cost_usd": tracker.summary()["spent_usd"],
    }
    (run_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nresample null (same instruction, re-asked): "
          f"{index['resample_null']['image']:.1f}   "
          f"cost ${index['resample_null']['cost_usd']:.4f}")
    print(f"recorded in {run_dir}/index.json")


if __name__ == "__main__":
    main()
