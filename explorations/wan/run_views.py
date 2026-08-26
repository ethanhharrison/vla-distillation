"""Generate all three DROID views from Wan, two ways, so they can be compared.

The open question this exists to answer: Stage B needs three mutually consistent
camera views, and a single-image-conditioned video model gives no guarantee of
that. Two strategies, same prompt, same seed:

  independent  one generation per camera. Same seed means the same initial noise
               (identical latent shape), which is common-random-numbers variance
               reduction — NOT consistency. The three rollouts share no state, so
               nothing stops the marker moving in one view and not another.

  concat       one generation on the stitched three-view canvas, split back into
               per-camera clips. One image, one latent, one denoising trajectory,
               so the views *can* stay coherent. Also 3x cheaper: one generation
               instead of three.

The canvas geometry is **imported from `explorations/cosmos3/run_experiment.py`**
rather than copied, so it is byte-identical to the layout the Cosmos concat runs
used and the two families stay comparable. That layout (wrist 640x360 above two
320x180 exteriors, 640x540) is the one Cosmos3-Nano-Policy-DROID was post-trained
on — which is a reason to expect it to work there and *not* here: for a general
model like Wan a stitched canvas is off-distribution, and it may simply read as a
split-screen video wall. That is the thing to look at in the report.

Prior on the canvas from cosmos3: best lever on both 16B models (jitter 1.06
general / 0.89 specialist, drift roughly halved) but the WORST config on Super.
So it is not a safe bet, just the cheapest one worth trying.

    CUDA_VISIBLE_DEVICES=1 ../cosmos3/.venv/bin/python run_views.py \
        --prompt-template robot_prefix --view-mode concat --out results/v_prefix_concat
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import prompts
from smoke_i2v import (
    DROID_FPS,
    NEGATIVE_PROMPT,
    WAN_FPS,
    diff,
    fit_size,
    to_pil,
)

HERE = Path(__file__).resolve().parent


def _load_cosmos_canvas():
    """Borrow the canvas geometry from the Cosmos harness, by path.

    Same layout by import rather than by copy: two implementations of one
    geometry would drift and silently break the cross-family comparison. Loaded
    via importlib rather than a sys.path insert because `explorations/cosmos3`
    also contains a `smoke_i2v.py` and a `make_report.py`, so putting it on the
    path would shadow this directory's modules depending on import order.
    """
    import importlib.util

    path = HERE.parent / "cosmos3" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("cosmos3_run_experiment", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cosmos = _load_cosmos_canvas()
CANVAS_W, CANVAS_H = _cosmos.CANVAS_W, _cosmos.CANVAS_H
build_canvas, split_canvas = _cosmos.build_canvas, _cosmos.split_canvas

# --------------------------------------------------------------------------- #
# Canvas layouts
#
# The Cosmos layout (wrist 640x360 above two 320x180 exteriors) is the one
# Cosmos3-Nano-Policy-DROID was post-trained on, which is a reason to expect it
# to work *there* and not here. Measured on Wan it collapsed exterior motion to
# ~6.5 against real motion of 6.8 while the wrist barely moved against real
# motion of 61 — consistent either with "the canvas enforces coherence" or with
# "each view got a third of the pixels and the model gave up".
#
# `grid2x2` tests the second reading from a different angle: a plain 2x2 grid of
# native-resolution cells, which is also the shape DreamZero emits natively, so
# it is the likelier layout for a general model to have seen. The fourth cell is
# left black rather than duplicated — a duplicate would let the model satisfy the
# frame by copying, which is the failure we are trying to detect.
# --------------------------------------------------------------------------- #

GRID_W, GRID_H = 640, 360      # 2x2 of the native 320x180 DROID frames
GRID_CELLS = {"exterior_1": (0, 0), "exterior_2": (320, 0), "wrist": (0, 180)}


def build_grid2x2(frame_paths: dict[str, Path]) -> Image.Image:
    canvas = Image.new("RGB", (GRID_W, GRID_H))
    for cam, xy in GRID_CELLS.items():
        canvas.paste(Image.open(frame_paths[cam]).convert("RGB").resize((320, 180)), xy)
    return canvas


def split_grid2x2(frame: np.ndarray) -> dict[str, np.ndarray]:
    h, w = frame.shape[:2]
    hh, hw = h // 2, w // 2
    return {"exterior_1": frame[:hh, :hw], "exterior_2": frame[:hh, hw:],
            "wrist": frame[hh:, :hw]}


#: name -> (build, split). Selected with --canvas-layout.
LAYOUTS = {
    "cosmos": (build_canvas, split_canvas),
    "grid2x2": (build_grid2x2, split_grid2x2),
}

DEFAULT_MODEL = "/raid/users/tiger/vla-distillation/models/Wan2.2-I2V-A14B"
DEFAULT_SITUATIONS = HERE.parent / "cosmos3/results/situations_multitraj"
CAMERAS = ("exterior_1", "exterior_2", "wrist")


def generate(pipe, image, prompt_text, negative, args, height, width):
    t0 = time.time()
    frames = pipe(
        image=image, prompt=prompt_text, negative_prompt=negative,
        height=height, width=width, num_frames=args.num_frames,
        guidance_scale=args.guidance_scale, guidance_scale_2=args.guidance_scale_2,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).frames[0]
    return to_pil(frames), time.time() - t0


def even(im: Image.Image) -> Image.Image:
    """Crop to even width/height — libx264 + yuv420p rejects odd dimensions.

    The canvas thirds are routinely odd once the generated canvas is cut up; this
    is the failure that killed a whole Cosmos sweep group with a broken pipe.
    """
    w, h = im.size
    return im if not (w % 2 or h % 2) else im.crop((0, 0, w - w % 2, h - h % 2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--situations", default=str(DEFAULT_SITUATIONS))
    p.add_argument("--situation", default="ep000_t0000")
    p.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    p.add_argument("--view-mode", choices=["independent", "concat"], default="independent")
    p.add_argument("--prompt-template", default="robot_prefix")
    p.add_argument("--negative-extra", default="no_human_no_camera")
    p.add_argument("--instruction", default=None)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--max-area", type=int, default=480 * 832)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--guidance-scale-2", type=float, default=None)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image

    sdir = Path(args.situations) / args.situation
    sit = json.loads((sdir / "situation.json").read_text())
    instruction = args.instruction or sit["instruction"]
    tpl_name, tpl_text = prompts.resolve_template(args.prompt_template)
    prompt_text = prompts.build_prompt(tpl_text, instruction)
    neg_name, neg_text = prompts.resolve_negative(args.negative_extra)
    negative = NEGATIVE_PROMPT + ("，" + neg_text if neg_text else "")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.situation} · mode={args.view_mode} · template={tpl_name} · "
          f"cfg={args.guidance_scale}/{args.guidance_scale_2} · seed={args.seed}")
    print(f"prompt: {prompt_text!r}")

    t0 = time.time()
    pipe = WanImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    load_s = time.time() - t0
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    print(f"loaded in {load_s:.1f}s")

    per_cam: dict[str, list[Image.Image]] = {}
    timings: dict[str, float] = {}

    if args.view_mode == "concat":
        paths = {c: sdir / "history" / f"{c}_0.png" for c in CAMERAS}
        if not all(p.exists() for p in paths.values()):
            raise SystemExit("concat needs all three cameras present")
        canvas = build_canvas(paths)          # 640x540, the Cosmos layout
        h, w = fit_size(canvas, args.max_area, mod)
        canvas = canvas.resize((w, h))
        print(f"canvas {CANVAS_W}x{CANVAS_H} -> {w}x{h}, one generation for all views")
        frames, dt = generate(pipe, canvas, prompt_text, negative, args, h, w)
        timings["concat"] = dt
        for cam in args.cameras:
            per_cam[cam] = [Image.fromarray(split_canvas(np.asarray(f))[cam]) for f in frames]
        print(f"  generated in {dt:.1f}s -> per-view "
              f"{per_cam[args.cameras[0]][0].size}")
    else:
        for cam in args.cameras:
            src = sdir / "history" / f"{cam}_0.png"
            if not src.exists():
                continue
            image = load_image(str(src))
            h, w = fit_size(image, args.max_area, mod)
            frames, dt = generate(pipe, image.resize((w, h)), prompt_text, negative, args, h, w)
            per_cam[cam], timings[cam] = frames, dt
            print(f"  [{cam}] generated in {dt:.1f}s at {w}x{h}")

    # --- write clips + per-camera calibration -------------------------------
    gen_idx = None
    cams_rec = {}
    for cam, frames in per_cam.items():
        frames = [even(f) for f in frames]
        export_to_video(frames, str(out / f"generated_{cam}.mp4"), fps=int(WAN_FPS),
                        macro_block_size=1)
        src = np.asarray(load_image(str(sdir / "history" / f"{cam}_0.png")).convert("RGB"))
        gen_idx = min(round(args.k / DROID_FPS * WAN_FPS), len(frames) - 1)
        rec = {
            "video": f"generated_{cam}.mp4",
            "num_frames": len(frames),
            "frame_size": list(frames[0].size),
            "latency_s": round(timings.get(cam, timings.get("concat", 0.0)), 1),
            "recon_floor": round(diff(np.asarray(frames[0].convert("RGB")), src), 2),
            "gen_vs_src": round(diff(np.asarray(frames[gen_idx].convert("RGB")), src), 2),
        }
        real_p = sdir / "future" / f"{cam}_f{args.k:02d}.png"
        if real_p.exists():
            real = np.asarray(load_image(str(real_p)).convert("RGB"))
            rec["gen_vs_real"] = round(diff(np.asarray(frames[gen_idx].convert("RGB")), real), 2)
            rec["real_vs_src"] = round(diff(real, src), 2)
        cams_rec[cam] = rec
        print(f"  [{cam}] floor {rec['recon_floor']} · vs src {rec['gen_vs_src']} · "
              f"vs real {rec.get('gen_vs_real', 'n/a')} · real motion {rec.get('real_vs_src', 'n/a')}")

    index = {
        "model": Path(args.model_path).name,
        "view_mode": args.view_mode,
        "situations_dir": str(Path(args.situations).resolve()),
        "situation": args.situation,
        "cameras": list(per_cam),
        "fps": WAN_FPS, "k_droid": args.k, "matched_generated_frame": gen_idx,
        "instruction": instruction, "prompt": prompt_text,
        "prompt_template": tpl_name, "negative_preset": neg_name,
        "num_frames": args.num_frames, "steps": args.steps,
        "guidance_scale": args.guidance_scale, "guidance_scale_2": args.guidance_scale_2,
        "seed": args.seed, "load_s": round(load_s, 1),
        "total_generate_s": round(sum(timings.values()), 1),
        "camera_records": cams_rec,
    }
    (out / "run.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {out}  (total generate {index['total_generate_s']}s)")


if __name__ == "__main__":
    main()
