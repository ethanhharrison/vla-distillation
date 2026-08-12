"""Run the Cosmos 3 A/B/C experiment over the DreamZero situation set.

Same protocol as `explorations/dreamzero/run_experiment.py`, same situations, so
the two models are comparable on the channel that matters here (generated video
under counterfactual instructions). Differences forced by the model:

- Cosmos conditions on ONE frame, no proprioception and no camera grid, so each
  camera is a separate call rather than one 2x2-grid call.
- The action chunk is `(chunk_size, 10)` in the generic `droid_lerobot` space
  (3D translation + 6D rotation end-effector pose DELTAS, plus absolute gripper),
  model-normalized. That is a different space from DreamZero's absolute joint
  positions, and no denormalization stats ship with the checkpoint — so actions
  are only compared BETWEEN conditions here, never to logged DROID actions.

Conditions come from each situation's conditions.json (written by DreamZero's
prepare_instructions.py): A = original, B = counterfactual, C = null/stress,
MP = the targeted "move the pot" probe.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python run_experiment.py            # default run
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python run_experiment.py --fps-sweep  # fps calibration only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "/home/tiger/proj/staging/vla/models/Cosmos3-Nano-Policy-DROID"
DEFAULT_SITUATIONS = HERE.parent / "dreamzero/results/situations/Mon_Apr_17_14:48:05_2023"

# DROID exteriors watch the robot from the front; the wrist camera is hand-mounted.
VIEWPOINT = {"exterior_1": "third_person_view", "exterior_2": "third_person_view",
             "wrist": "wrist_view", "concat": "concat_view"}
CANVAS_CAMERAS = ("exterior_1", "exterior_2", "wrist")


def build_pipe(model_path: str, flow_shift: float | None = None):
    from diffusers import Cosmos3OmniPipeline

    pipe = Cosmos3OmniPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        # Off deliberately: the video guardrail pixelates faces, which would
        # silently corrupt generated frames relative to the real ones we compare
        # against (DROID scenes have people at the edges). Logged in index.json.
        enable_safety_checker=False,
    )
    # The checkpoint ships flow_shift=1.0 + karras sigmas, but every documented
    # recipe overrides it: NVIDIA's diffusers action example uses 10.0 (karras
    # off), vLLM-Omni's action examples use 5.0, and the tech report describes
    # "a shifted noise schedule of 5" for the deployed DROID policy. Flow shift
    # sets how much sampling time goes to high noise (global structure) versus
    # low noise (detail), so it is a prime suspect for structural artifacts.
    # None = leave the checkpoint default untouched.
    if flow_shift is not None:
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=flow_shift, use_karras_sigmas=False
        )
    return pipe


def policy_call(pipe, image, prompt: str, cam: str, args) -> tuple[list, np.ndarray | None, float]:
    from diffusers import CosmosActionCondition

    # Output resolution tracks the INPUT, not resolution_tier — the tier only
    # sizes the conditioning canvas and never upscales (confirmed: tier 256 and
    # 480 both return 320x176 from our 320x180 DROID frames). So the only way to
    # get a larger generation is to hand the model a larger image. Cosmos was
    # trained at up to 720p, so a 2-3x upscale is closer to its training
    # distribution than our native 180x320 frames are.
    upscale = getattr(args, "upscale", 1) or 1
    if upscale != 1:
        image = image.resize((image.width * upscale, image.height * upscale), Image.LANCZOS)

    t0 = time.time()
    result = pipe(
        prompt=prompt,
        action=CosmosActionCondition(
            mode="policy",
            chunk_size=args.chunk_size,
            domain_name="droid_lerobot",
            resolution_tier=args.resolution_tier,
            image=image,
            view_point=VIEWPOINT.get(cam, "third_person_view"),
        ),
        fps=args.fps,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        use_system_prompt=False,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )
    dt = time.time() - t0
    action = None
    if result.action is not None:
        action = np.asarray(result.action[0].float().cpu())
    return result.video, action, dt


# The layout Cosmos3-Nano-Policy-DROID was actually post-trained on (tech report
# 4.2.5): the wrist view at 360x640 on top, the two exteriors at 180x320 side by
# side beneath it, giving a 540x640 canvas. Feeding that as ONE image with
# view_point="concat_view" is the closest this diffusers path can get to the
# checkpoint's training distribution — every other run here feeds a single
# camera, which is off-distribution by construction.
CANVAS_W, CANVAS_H, WRIST_H = 640, 540, 360


def build_canvas(frame_paths: dict[str, Path]):
    from PIL import Image

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H))
    canvas.paste(Image.open(frame_paths["wrist"]).convert("RGB").resize((CANVAS_W, WRIST_H)), (0, 0))
    canvas.paste(Image.open(frame_paths["exterior_1"]).convert("RGB").resize((CANVAS_W // 2, CANVAS_H - WRIST_H)),
                 (0, WRIST_H))
    canvas.paste(Image.open(frame_paths["exterior_2"]).convert("RGB").resize((CANVAS_W // 2, CANVAS_H - WRIST_H)),
                 (CANVAS_W // 2, WRIST_H))
    return canvas


def split_canvas(frame: np.ndarray) -> dict[str, np.ndarray]:
    """Cut a generated canvas frame back into the three camera views."""
    h, w = frame.shape[:2]
    top = round(h * WRIST_H / CANVAS_H)
    return {"wrist": frame[:top], "exterior_1": frame[top:, : w // 2], "exterior_2": frame[top:, w // 2:]}


def save_sample(out: Path, cam: str, frames, action, fps: float) -> None:
    from diffusers.utils import export_to_video

    out.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out / f"generated_{cam}.mp4"), fps=int(fps), macro_block_size=1)
    if action is not None:
        np.save(out / f"action_{cam}.npy", action)


def fps_sweep(pipe, args) -> None:
    """Which conditioning fps matches DROID's real motion? Cheap 2-call check.

    NVIDIA's canonical policy request uses fps=5, but this checkpoint was
    post-trained on DROID at 15 Hz. The two imply very different real-time spans
    for the same 16-step chunk (3.2 s vs ~1.1 s), which changes which real future
    frame each generated frame should be compared against.
    """
    from diffusers.utils import load_image

    sit_dir = Path(args.situations)
    sid = json.loads((sit_dir / "meta.json").read_text())["situations"][0]
    sdir = sit_dir / sid
    instruction = json.loads((sdir / "situation.json").read_text())["instruction"]
    image = load_image(str(sdir / "history" / "exterior_1_0.png"))
    real = [np.asarray(load_image(str(p)).convert("RGB")) for p in sorted((sdir / "future").glob("exterior_1_f*.png"))]

    out = HERE / "results" / "fps_sweep"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fps in (5.0, 15.0):
        args.fps = fps
        frames, action, dt = policy_call(pipe, image, instruction, "exterior_1", args)
        save_sample(out / f"fps{int(fps)}", "exterior_1", frames, action, fps)
        # Align the LAST generated frame to the real frame it should depict.
        span_s = args.chunk_size / fps
        real_idx = min(int(round(span_s * 15)) - 1, len(real) - 1)
        gen_last = np.asarray(frames[-1].convert("RGB").resize((real[0].shape[1], real[0].shape[0])), dtype=float)
        diff_real = float(np.abs(gen_last - real[real_idx].astype(float)).mean())
        src = np.asarray(image.convert("RGB"), dtype=float)
        rows.append({
            "fps": fps, "span_s": round(span_s, 2), "real_frame_index": real_idx + 1,
            "latency_s": round(dt, 2),
            "gen_last_vs_real": round(diff_real, 2),
            "gen_last_vs_source": round(float(np.abs(gen_last - src).mean()), 2),
            "real_at_idx_vs_source": round(float(np.abs(real[real_idx].astype(float) - src).mean()), 2),
        })
        print(f"  fps={fps}: span {span_s:.2f}s -> real frame {real_idx + 1}/{len(real)}, "
              f"|gen-real|={rows[-1]['gen_last_vs_real']}, |gen-src|={rows[-1]['gen_last_vs_source']}, "
              f"|real-src|={rows[-1]['real_at_idx_vs_source']}  ({dt:.1f}s)")
    (out / "fps_sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}/fps_sweep.json")


def run(args) -> None:
    from diffusers.utils import load_image

    sit_dir = Path(args.situations)
    meta = json.loads((sit_dir / "meta.json").read_text())
    pipe = build_pipe(args.model_path, flow_shift=args.flow_shift)

    if args.fps_sweep:
        fps_sweep(pipe, args)
        return

    runs_dir = HERE / "results" / args.run_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    situations = meta["situations"][: args.limit_situations] if args.limit_situations else meta["situations"]
    samples = []
    for sid in situations:
        sdir = sit_dir / sid
        sit = json.loads((sdir / "situation.json").read_text())
        cond_path = sdir / "conditions.json"
        conditions = json.loads(cond_path.read_text()) if cond_path.exists() else {"A": [sit["instruction"]]}
        if args.conditions:
            conditions = {k: v for k, v in conditions.items() if k in args.conditions}

        for cond, instrs in conditions.items():
            for j, instruction in enumerate(instrs):
                sample_id = f"{sid}__{cond}{j}"
                out = runs_dir / sample_id
                rec = {
                    "sample_id": sample_id, "situation_id": sid, "anchor": sit["anchor"],
                    "condition": cond, "instruction": instruction, "cameras": {},
                }
                if args.concat_view:
                    # One call on the three-view canvas, then cut the generated
                    # canvas back into per-camera videos so everything
                    # downstream (report, metrics) is unchanged.
                    paths = {c: sdir / "history" / f"{c}_0.png" for c in CANVAS_CAMERAS}
                    if not all(p.exists() for p in paths.values()):
                        continue
                    canvas = build_canvas(paths)
                    frames, action, dt = policy_call(pipe, canvas, instruction, "concat", args)
                    out.mkdir(parents=True, exist_ok=True)
                    if action is not None:
                        np.save(out / "action_concat.npy", action)
                    for cam in CANVAS_CAMERAS:
                        cam_frames = [Image.fromarray(split_canvas(np.asarray(f))[cam]) for f in frames]
                        save_sample(out, cam, cam_frames, None, args.fps)
                        rec["cameras"][cam] = {
                            "video": f"generated_{cam}.mp4",
                            "num_frames": len(cam_frames),
                            "frame_size": list(cam_frames[0].size),
                            "action": "action_concat.npy" if action is not None else None,
                            "action_shape": list(action.shape) if action is not None else None,
                            "action_mean_abs": (round(float(np.abs(action).mean()), 4)
                                                if action is not None else None),
                            "latency_s": round(dt, 2),
                        }
                    print(f"  {sample_id} [concat] '{instruction[:38]}' -> {len(frames)}f "
                          f"{frames[0].size} action{action.shape if action is not None else None} {dt:.1f}s")
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "sample.json").write_text(json.dumps(rec, indent=2))
                    samples.append(rec)
                    continue

                for cam in args.cameras:
                    src = sdir / "history" / f"{cam}_0.png"
                    if not src.exists():
                        continue
                    image = load_image(str(src))
                    frames, action, dt = policy_call(pipe, image, instruction, cam, args)
                    save_sample(out, cam, frames, action, args.fps)
                    rec["cameras"][cam] = {
                        "video": f"generated_{cam}.mp4",
                        "num_frames": len(frames),
                        "frame_size": list(frames[0].size),
                        "action": f"action_{cam}.npy" if action is not None else None,
                        "action_shape": list(action.shape) if action is not None else None,
                        "action_mean_abs": round(float(np.abs(action).mean()), 4) if action is not None else None,
                        "latency_s": round(dt, 2),
                    }
                    print(f"  {sample_id} [{cam}] '{instruction[:38]}' -> "
                          f"{len(frames)}f {frames[0].size} action{action.shape if action is not None else None} {dt:.1f}s")
                out.mkdir(parents=True, exist_ok=True)
                (out / "sample.json").write_text(json.dumps(rec, indent=2))
                samples.append(rec)

    index = {
        # Multi-trajectory situation sets have one episode per situation, so
        # there is no single episode_id — fall back to the set's layout string.
        "situations_dir": str(sit_dir),
        "episode_id": meta.get("episode_id") or meta.get("layout", "multiple episodes"),
        "model_path": args.model_path, "model": "Cosmos3-Nano-Policy-DROID",
        "action_mode": "policy", "domain_name": "droid_lerobot",
        "action_space": "10D EEF pose delta (3D translation + 6D rotation) + gripper, model-normalized",
        "cameras": args.cameras, "fps": args.fps, "chunk_size": args.chunk_size,
        "resolution_tier": args.resolution_tier, "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale, "seed": args.seed,
        "flow_shift": args.flow_shift, "concat_view": args.concat_view,
        "upscale": args.upscale,
        "guardrails": False, "prompt_upsampling": False,
        "samples": samples,
    }
    (runs_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nWrote {len(samples)} samples to {runs_dir}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--situations", default=str(DEFAULT_SITUATIONS))
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--run-name", default="runs")
    p.add_argument("--cameras", nargs="+", default=["exterior_1"])
    p.add_argument("--conditions", nargs="+", default=None, help="Subset of A/B/C/MP (default: all).")
    p.add_argument("--limit-situations", type=int, default=None)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--resolution-tier", type=int, default=256)
    p.add_argument("--steps", type=int, default=30)
    # 3.0 is what the tech report uses for the DROID policy. NVIDIA's diffusers
    # example passes 1.0 (= no classifier-free guidance), which measurably
    # collapses instruction sensitivity — see the CFG sweep in README.md.
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flow-shift", type=float, default=None,
                   help="Override the scheduler flow shift (checkpoint default is 1.0; "
                        "NVIDIA recipes use 5-10). None = leave untouched.")
    p.add_argument("--upscale", type=int, default=1,
                   help="Resize the conditioning frame by this integer factor before generating. "
                        "Output resolution follows the input, so this is the only real quality knob.")
    p.add_argument("--concat-view", action="store_true",
                   help="Feed the 540x640 three-view canvas as one image with view_point=concat_view "
                        "(the layout the checkpoint was post-trained on), then split the output back "
                        "into per-camera videos. Pair with --resolution-tier 480.")
    p.add_argument("--fps-sweep", action="store_true", help="Only run the fps calibration check.")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
