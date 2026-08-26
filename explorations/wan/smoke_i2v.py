"""One-call smoke test: Wan2.2-I2V-A14B as a Stage B subgoal generator.

The first *general* (not robot-post-trained) image-to-video model put on our DROID
frames. Cosmos 3 established the axis this is probing: post-training on DROID
bought physical faithfulness but cost language responsiveness, and the general
checkpoints followed the instruction more. Wan2.2 is the strongest open-weight
general I2V model available (Wan 2.5/2.6 exist but are API-only), so this asks
whether that trade continues to hold outside the Cosmos family.

This is a smoke, not an experiment: one situation, one camera, one seed. It
answers "does it load, fit and produce a sane clip, and at what cost" — nothing
about instruction sensitivity, which needs the A/B/C harness and a seed null.

Notes on the recipe, taken from the model card rather than from memory:

- **MoE, two transformers.** `transformer` (high-noise) + `transformer_2`
  (low-noise) with `boundary_ratio=0.9` deciding the handover. 27B total, 14B
  active, so compute is 14B-ish but the weights of both must be resident.
- **fps 16**, not the 24 of Wan2.2-TI2V-5B and not DROID's 15. 81 frames is
  ~5.06 s of model time; the frame matching a real DROID frame `t+k` is
  therefore `round(k / 15 * 16)`, which is what `--k` below converts.
- **Aspect-preserving sizing.** The card computes height/width from a target
  *area* and rounds to a multiple of the VAE/patch stride, rather than forcing
  832x480. Our DROID frames are exactly 16:9, so this keeps them undistorted —
  worth preserving, since every metric we run is a pixel difference.
- **The negative prompt ships in Chinese** and is used verbatim, as the card
  does. Dropping it is a deviation, not a simplification.

    CUDA_VISIBLE_DEVICES=7 ../cosmos3/.venv/bin/python smoke_i2v.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import prompts

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "/raid/users/tiger/vla-distillation/models/Wan2.2-I2V-A14B"
DEFAULT_SITUATIONS = HERE.parent / "cosmos3/results/situations_multitraj"
DEFAULT_SITUATION = "ep000_t0000"

#: The card's default negative prompt, verbatim. Steers away from oversaturation,
#: static frames, subtitles and malformed anatomy.
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

DROID_FPS = 15.0
WAN_FPS = 16.0


def fit_size(image, max_area: int, mod_value: int) -> tuple[int, int]:
    """Largest (height, width) under `max_area` that keeps aspect and hits the stride."""
    aspect = image.height / image.width
    height = round(np.sqrt(max_area * aspect)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect)) // mod_value * mod_value
    return height, width


def to_pil(frames) -> list:
    """Normalise whatever the pipeline returned into PIL images.

    The Wan pipelines default to `output_type="np"` and hand back float [0,1]
    arrays, not PIL. Keeping everything PIL from here on is the convention the
    cosmos3 harness arrived at the hard way: `export_to_video` branches on type
    and multiplies an ndarray by 255, so a uint8 array wraps modulo 256 into
    garbage that still encodes and still plays. Float arrays happen to survive
    that path, which is worse — the bug only shows up once someone changes the
    dtype. Convert once, explicitly.
    """
    from PIL import Image

    out = []
    for f in frames:
        if isinstance(f, Image.Image):
            out.append(f)
            continue
        arr = np.asarray(f)
        if arr.dtype != np.uint8:  # float [0,1] -> uint8, no modulo wrap
            arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
        out.append(Image.fromarray(arr))
    return out


def diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference on the smaller grid (see image_edit/metrics.py)."""
    from PIL import Image

    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    to = lambda x: (x if (x.shape[0], x.shape[1]) == (h, w) else np.asarray(  # noqa: E731
        Image.fromarray(x.astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS)))
    return float(np.abs(to(a).astype(float) - to(b).astype(float)).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--situations", default=str(DEFAULT_SITUATIONS))
    p.add_argument("--situation", default=DEFAULT_SITUATION)
    p.add_argument("--camera", default="exterior_1")
    p.add_argument("--instruction", default=None, help="Default: the situation's own.")
    p.add_argument("--prompt-template", default="bare",
                   help=f"Template name or literal containing {{instruction}}. "
                        f"Registered: {', '.join(prompts.TEMPLATES)}.")
    p.add_argument("--num-frames", type=int, default=81, help="VAE wants 4k+1.")
    p.add_argument("--max-area", type=int, default=480 * 832,
                   help="Target pixel area; aspect ratio is preserved.")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--k", type=int, default=32,
                   help="Real DROID future frame to compare against (15 Hz).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-negative", action="store_true",
                   help="Drop the card's negative prompt (a deviation — logged).")
    p.add_argument("--negative-extra", default="none",
                   help=f"Preset name or literal, appended to the card's negative prompt. "
                        f"Registered: {', '.join(prompts.NEGATIVES)}.")
    p.add_argument("--guidance-scale-2", type=float, default=None,
                   help="Guidance for the LOW-noise MoE expert (the one that resolves "
                        "detail). None = inherit --guidance-scale, which is what every run "
                        "so far did. Splitting the two is untested here.")
    p.add_argument("--out", default=str(HERE / "results/smoke_i2v"))
    args = p.parse_args()

    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image

    sdir = Path(args.situations) / args.situation
    sit = json.loads((sdir / "situation.json").read_text())
    instruction = args.instruction or sit["instruction"]
    tpl_name, tpl_text = prompts.resolve_template(args.prompt_template)
    prompt_text = prompts.build_prompt(tpl_text, instruction)
    neg_name, neg_text = prompts.resolve_negative(args.negative_extra)
    src_path = sdir / "history" / f"{args.camera}_0.png"
    image = load_image(str(src_path))
    print(f"situation {args.situation} [{args.camera}] source {image.size}")
    print(f"instruction: {instruction!r}")
    print(f"template {tpl_name!r} + negative {neg_name!r} -> {prompt_text!r}")

    t0 = time.time()
    pipe = WanImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s · {torch.cuda.memory_allocated()/1e9:.1f} GB resident")

    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height, width = fit_size(image, args.max_area, mod_value)
    image = image.resize((width, height))
    print(f"generating {args.num_frames}f at {width}x{height} "
          f"(stride {mod_value}, aspect {width/height:.3f} vs source 1.778)")

    negative = None if args.no_negative else (
        NEGATIVE_PROMPT + ("，" + neg_text if neg_text else "")
    )

    t0 = time.time()
    frames = pipe(
        image=image,
        prompt=prompt_text,
        negative_prompt=negative,
        height=height, width=width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).frames[0]
    gen_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"generated in {gen_s:.1f}s · peak {peak:.1f} GB")

    frames = to_pil(frames)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out / "generated.mp4"), fps=int(WAN_FPS))

    # Calibration, in the same units as every other harness here: how far the
    # clip moved from the source, and how close it landed to what really
    # happened at the matched timestamp.
    src = np.asarray(load_image(str(src_path)).convert("RGB"))
    gen_idx = min(round(args.k / DROID_FPS * WAN_FPS), len(frames) - 1)
    gen_at_k = np.asarray(frames[gen_idx].convert("RGB"))
    real_path = sdir / "future" / f"{args.camera}_f{args.k:02d}.png"
    stats = {
        "model": Path(args.model_path).name,
        # Provenance the report needs to find the real frames again. Without
        # these a run is unreadable once the CLI defaults change under it.
        "situations_dir": str(Path(args.situations).resolve()),
        "situation": args.situation,
        "camera": args.camera,
        "fps": WAN_FPS,
        "instruction": instruction,
        "prompt": prompt_text,
        "prompt_template": tpl_name,
        "negative_preset": neg_name,
        "frames": len(frames), "size": [width, height],
        "load_s": round(load_s, 1), "generate_s": round(gen_s, 1),
        "peak_vram_gb": round(peak, 1),
        "steps": args.steps, "guidance_scale": args.guidance_scale, "seed": args.seed,
        "negative_prompt": not args.no_negative,
        "guidance_scale_2": args.guidance_scale_2,
        "k_droid": args.k, "matched_generated_frame": gen_idx,
        "recon_floor": round(diff(np.asarray(frames[0].convert("RGB")), src), 2),
        "gen_vs_src": round(diff(gen_at_k, src), 2),
    }
    if real_path.exists():
        real = np.asarray(load_image(str(real_path)).convert("RGB"))
        stats["gen_vs_real"] = round(diff(gen_at_k, real), 2)
        stats["real_vs_src"] = round(diff(real, src), 2)
    (out / "smoke.json").write_text(json.dumps(stats, indent=2))
    frames[0].save(out / "frame_000.png")
    frames[gen_idx].save(out / f"frame_{gen_idx:03d}.png")

    print(f"\nwrote {out}")
    print(f"  recon floor {stats['recon_floor']} (frame 0 vs real source — the VAE tax)")
    print(f"  gen vs src  {stats['gen_vs_src']} at generated frame {gen_idx} "
          f"(= real DROID frame {args.k})")
    if "gen_vs_real" in stats:
        print(f"  gen vs real {stats['gen_vs_real']}   "
              f"real motion {stats['real_vs_src']}  <- does generation beat copying the input?")


if __name__ == "__main__":
    main()
