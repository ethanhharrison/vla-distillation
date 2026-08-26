"""A/B/C evaluation for Wan — the grid the hosted image editors got, on video.

Same situations, conditions, cameras and units as `explorations/image_edit`,
`explorations/cosmos3` and `explorations/dreamzero`, so every number lands in
tables that can be read against each other.

  A  the episode's original instruction (sanity)
  B  plausible counterfactuals proposed by Gemini
  C  authored impossible instructions ("fold the laundry")

Built to run unattended for hours, so the design is defensive:

- **One job = one (config, situation).** Jobs write only into their own sample
  directories and never touch a shared index, so many can run concurrently
  across GPUs with no locking and no race. The report scans the directory
  afterwards, the way `cosmos3/rebuild_index.py` recovers a crashed run.
- **Resumable.** A sample whose clips already exist is skipped, so a killed or
  re-launched sweep costs only the work it had not finished.
- **A failed generation never kills the job.** It is recorded in the sample and
  the loop continues; a whole situation dying silently is the failure mode that
  wastes a night.
- **No seed null, no interpretation.** `--seed-null` re-runs condition A of the
  first situation at seed+1; without it a sensitivity number cannot be told
  apart from re-rolling the dice.

    CUDA_VISIBLE_DEVICES=1 ../cosmos3/.venv/bin/python run_eval.py \
        --config-name e_prefix --prompt-template robot_prefix --situation ep000_t0000
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import prompts
from run_views import CAMERAS, DEFAULT_MODEL, DEFAULT_SITUATIONS, LAYOUTS, even
from smoke_i2v import DROID_FPS, NEGATIVE_PROMPT, WAN_FPS, diff, fit_size, to_pil

HERE = Path(__file__).resolve().parent


def load_conditions(sdir: Path, keep: list[str] | None) -> dict[str, list[str]]:
    sit = json.loads((sdir / "situation.json").read_text())
    path = sdir / "conditions.json"
    conds = json.loads(path.read_text()) if path.exists() else {"A": [sit["instruction"]]}
    conds = {k: v for k, v in conds.items() if v}
    return {k: v for k, v in conds.items() if k in keep} if keep else conds


def generate_views(pipe, sdir, prompt_text, negative, args, mod):
    """Return {camera: [PIL frames]} plus the wall time, for either view mode."""
    from diffusers.utils import load_image

    def _call(image, h, w):
        t0 = time.time()
        out = pipe(
            image=image, prompt=prompt_text, negative_prompt=negative,
            height=h, width=w, num_frames=args.num_frames,
            guidance_scale=args.guidance_scale, guidance_scale_2=args.guidance_scale_2,
            num_inference_steps=args.steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).frames[0]
        return to_pil(out), time.time() - t0

    if args.view_mode == "concat":
        paths = {c: sdir / "history" / f"{c}_0.png" for c in CAMERAS}
        if not all(p.exists() for p in paths.values()):
            raise FileNotFoundError("concat needs all three cameras")
        build, split = LAYOUTS[args.canvas_layout]
        canvas = build(paths)
        h, w = fit_size(canvas, args.max_area, mod)
        frames, dt = _call(canvas.resize((w, h)), h, w)
        return {c: [Image.fromarray(split(np.asarray(f))[c]) for f in frames]
                for c in args.cameras}, dt

    per_cam, total = {}, 0.0
    for cam in args.cameras:
        src = sdir / "history" / f"{cam}_0.png"
        if not src.exists():
            continue
        image = load_image(str(src))
        h, w = fit_size(image, args.max_area, mod)
        frames, dt = _call(image.resize((w, h)), h, w)
        per_cam[cam], total = frames, total + dt
    return per_cam, total


def score(frames, sdir, cam, k) -> dict:
    from diffusers.utils import load_image

    src = np.asarray(load_image(str(sdir / "history" / f"{cam}_0.png")).convert("RGB"))
    j = min(round(k / DROID_FPS * WAN_FPS), len(frames) - 1)
    rec = {"num_frames": len(frames), "matched_frame": j,
           "recon_floor": round(diff(np.asarray(frames[0].convert("RGB")), src), 2),
           "gen_vs_src": round(diff(np.asarray(frames[j].convert("RGB")), src), 2)}
    real_p = sdir / "future" / f"{cam}_f{k:02d}.png"
    if real_p.exists():
        real = np.asarray(load_image(str(real_p)).convert("RGB"))
        rec["gen_vs_real"] = round(diff(np.asarray(frames[j].convert("RGB")), real), 2)
        rec["real_vs_src"] = round(diff(real, src), 2)
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", required=True, help="Output dir under results/.")
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--situations", default=str(DEFAULT_SITUATIONS))
    p.add_argument("--situation", default=None,
                   help="One situation id (a job). Default: every situation in the set.")
    p.add_argument("--conditions", nargs="+", default=None)
    p.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    p.add_argument("--view-mode", choices=["independent", "concat"], default="independent")
    p.add_argument("--canvas-layout", choices=list(LAYOUTS), default="cosmos",
                   help="Canvas arrangement for --view-mode concat.")
    p.add_argument("--prompt-template", default="robot_prefix")
    p.add_argument("--negative-extra", default="no_human_no_camera")
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--max-area", type=int, default=480 * 832)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--guidance-scale-2", type=float, default=None)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-null", action="store_true",
                   help="Instead of the grid: re-run condition A of the FIRST situation at "
                        "seed+1, into _seed_null/. The control every sensitivity number "
                        "is read against.")
    args = p.parse_args()

    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video

    sit_dir = Path(args.situations).resolve()
    meta = json.loads((sit_dir / "meta.json").read_text())
    situations = [args.situation] if args.situation else meta["situations"]
    tpl_name, tpl_text = prompts.resolve_template(args.prompt_template)
    neg_name, neg_text = prompts.resolve_negative(args.negative_extra)
    negative = NEGATIVE_PROMPT + ("，" + neg_text if neg_text else "")

    run_dir = HERE / "results" / args.config_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Written by every job with identical content, so concurrent jobs cannot
    # disagree and there is nothing to lock.
    (run_dir / "config.json").write_text(json.dumps({
        "model": Path(args.model_path).name, "situations_dir": str(sit_dir),
        "view_mode": args.view_mode, "canvas_layout": args.canvas_layout,
        "prompt_template": tpl_name,
        "negative_preset": neg_name, "cameras": args.cameras,
        "num_frames": args.num_frames, "steps": args.steps,
        "guidance_scale": args.guidance_scale, "guidance_scale_2": args.guidance_scale_2,
        "seed": args.seed, "k_droid": args.k, "fps": WAN_FPS,
    }, indent=2))

    if args.seed_null:
        situations = situations[:1]
        args.conditions = ["A"]
        args.seed += 1

    # Plan first, so a resumed job can say what it is skipping before loading 69 GB.
    plan = []
    for sid in situations:
        sdir = sit_dir / sid
        if not (sdir / "situation.json").exists():
            print(f"  ! {sid}: no situation.json, skipping")
            continue
        for cond, instrs in load_conditions(sdir, args.conditions).items():
            for j, instruction in enumerate(instrs):
                name = "_seed_null" if args.seed_null else f"{sid}__{cond}{j}"
                out = run_dir / name
                want = [c for c in args.cameras
                        if (sdir / "history" / f"{c}_0.png").exists()]
                have = all((out / f"generated_{c}.mp4").exists() for c in want) and want
                plan.append({"sid": sid, "sdir": sdir, "cond": cond, "j": j,
                             "instruction": instruction, "out": out, "done": bool(have)})

    todo = [x for x in plan if not x["done"]]
    print(f"{args.config_name} · {len(plan)} samples, {len(plan) - len(todo)} already done, "
          f"{len(todo)} to run · mode={args.view_mode} · template={tpl_name} · "
          f"cfg={args.guidance_scale}/{args.guidance_scale_2} · seed={args.seed}")
    if not todo:
        print("nothing to do.")
        return

    t0 = time.time()
    pipe = WanImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    print(f"loaded in {time.time() - t0:.1f}s")

    for n, item in enumerate(todo, 1):
        name = item["out"].name
        prompt_text = prompts.build_prompt(tpl_text, item["instruction"])
        try:
            per_cam, dt = generate_views(pipe, item["sdir"], prompt_text, negative, args, mod)
        except Exception:
            # One bad sample must not end the night. Record and move on.
            item["out"].mkdir(parents=True, exist_ok=True)
            (item["out"] / "error.txt").write_text(traceback.format_exc())
            print(f"  ! [{n}/{len(todo)}] {name}: FAILED (see error.txt)")
            continue

        item["out"].mkdir(parents=True, exist_ok=True)
        cams_rec = {}
        for cam, frames in per_cam.items():
            frames = [even(f) for f in frames]
            export_to_video(frames, str(item["out"] / f"generated_{cam}.mp4"),
                            fps=int(WAN_FPS), macro_block_size=1)
            rec = score(frames, item["sdir"], cam, args.k)
            rec["video"] = f"generated_{cam}.mp4"
            cams_rec[cam] = rec
        (item["out"] / "sample.json").write_text(json.dumps({
            "sample_id": name, "situation_id": item["sid"], "condition": item["cond"],
            "instruction": item["instruction"], "prompt": prompt_text,
            "prompt_template": tpl_name, "view_mode": args.view_mode,
            "seed": args.seed, "guidance_scale": args.guidance_scale,
            "guidance_scale_2": args.guidance_scale_2, "steps": args.steps,
            "latency_s": round(dt, 1), "camera_records": cams_rec,
        }, indent=2))
        vs = " ".join(f"{c}:{r['gen_vs_src']}" for c, r in cams_rec.items())
        print(f"  [{n}/{len(todo)}] {name} '{item['instruction'][:32]}' {dt:.0f}s  {vs}",
              flush=True)

    print(f"{args.config_name}/{args.situation or 'all'} finished in "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
