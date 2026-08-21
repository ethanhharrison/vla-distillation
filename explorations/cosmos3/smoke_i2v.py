"""One-call smoke test: Cosmos3-Super (64B) in IMAGE-TO-VIDEO mode.

The Stage-B analogue that has never been run here. Everything so far has been
`mode="policy"` on the 16B Nano checkpoints; this is the general image-to-video
path on the Super tier, which is the largest checkpoint NVIDIA released. Note
there is NO Cosmos3-Super-Policy-DROID — Policy-DROID exists only at Nano and
Edge — so image-to-video is the only subgoal route at this tier.

Two things this answers, in order:
  1. Does ~131 GB of BF16 weights actually fit and run on ONE H200 (143.7 GB)?
     The diffusers docs say Super "does not fit on one 96 GB GPU, so it needs
     TP" — an H200 is half again bigger, so this is genuinely untested.
  2. Does an in-distribution JSON prompt steer the rollout?

On prompt format: Cosmos 3 was trained on long structured JSON captions, not
short imperatives. Our earlier probes deliberately fed raw instructions and
noted that as a caveat. Here we build the checkpoint's own caption schema
(see `assets/example_i2v_prompt.json`) around the instruction, and reuse the
shipped `assets/negative_prompt.json` verbatim.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python smoke_i2v.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "/raid/users/tiger/vla-distillation/models/Cosmos3-Super"
DEFAULT_FRAME = (
    HERE.parent / "dreamzero/results/situations/Mon_Apr_17_14:48:05_2023/t0024/history/exterior_1_0.png"
)


def build_json_prompt(instruction: str, seconds: float, fps: float, w: int, h: int) -> dict:
    """The checkpoint's caption schema, filled for a DROID manipulation scene.

    Only the fields the reference asset actually populates are set; the rest are
    left empty exactly as the reference does. The instruction drives `action`,
    `state_changes`, the single segment and the temporal caption, so that
    swapping the instruction changes the *task* text and nothing else — which is
    what makes an instruction-sensitivity comparison clean.
    """
    dur = f"{seconds:.1f}s"
    return {
        "subjects": [
            {
                "description": (
                    "A Franka Panda robot arm mounted at a tabletop workspace, viewed from a "
                    "fixed third-person camera. The arm and its two-finger parallel gripper are "
                    "positioned over the table surface among several small household objects."
                ),
                "appearance_details": (
                    "Matte black and white robot arm segments with visible joints, a two-finger "
                    "parallel-jaw gripper, cables running along the links"
                ),
                "relationship": "Primary actor manipulating objects on the table",
                "location": "Center of frame, extending over the tabletop",
                "relative_size": "Large within frame",
                "orientation": "Reaching forward and downward toward the table surface",
                "pose": "Articulated arm partially extended, gripper oriented toward the work area",
                "action": instruction,
                "state_changes": (
                    f"Begins from the pose shown in the first frame and makes the initial "
                    f"{dur} of visible progress toward: {instruction}"
                ),
                "clothing": "",
                "expression": "",
                "gender": "",
                "age": "",
                "skin_tone_and_texture": "",
                "facial_features": "",
                "number_of_subjects": 1,
                "number_of_arms": 1,
                "number_of_legs": 0,
            }
        ],
        "background_setting": (
            "An indoor tabletop robot workspace. A flat table surface holds a small number of "
            "everyday objects within the arm's reach. The background is an ordinary room "
            "interior, slightly out of focus, static throughout."
        ),
        "lighting": {
            "conditions": "Even indoor artificial lighting, consistent throughout",
            "direction": "Top-lit from overhead room lights",
            "shadows": "Soft contact shadows beneath the objects and the gripper",
            "illumination_effect": "Flat, clear visibility of the workspace with no strong highlights",
        },
        "aesthetics": {
            "composition": "Fixed third-person view of the workspace, arm centered, table filling the lower frame",
            "color_scheme": "Neutral table surface, black and white robot arm, coloured household objects as accents",
            "mood_atmosphere": "Neutral, procedural, documentary",
            "patterns": "Flat table surface, regular arm link geometry",
        },
        "cinematography": {
            "camera_motion": "Completely static camera, locked off on a tripod, no pan/tilt/zoom",
            "framing": "Medium wide shot of the tabletop workspace",
            "camera_angle": "Slightly above table height, looking down at the work area",
            "depth_of_field": "Deep",
            "focus": "Robot gripper and manipulated objects in sharp focus",
            "lens_focal_length": "Wide-angle",
        },
        "style_medium": "Live-action video",
        "artistic_style": "Realistic robot teleoperation recording, fixed surveillance-style camera",
        "context": (
            "Recorded footage from a robot manipulation dataset. A single continuous take from a "
            "stationary camera as the robot begins executing one instruction."
        ),
        "actions": [
            {"time": f"0:00-0:{seconds:04.1f}".replace(".", ""), "description": instruction}
        ],
        "text_and_signage_elements": [],
        "segments": [
            {
                "segment_index": 0,
                "time_range": f"0:00-{dur}",
                "description": (
                    f"The robot arm begins to execute the instruction: {instruction}. The motion "
                    f"is smooth and continuous. Only the arm and the objects it touches move; the "
                    f"camera, the table and the rest of the scene stay completely still."
                ),
                "key_changes": (
                    f"The gripper and the referenced object move toward the goal of: {instruction}. "
                    f"The task is started but NOT completed within this clip."
                ),
                "camera": "Static locked-off camera, no movement",
            }
        ],
        "transitions": [],
        "temporal_caption": (
            f"The video opens on a static third-person view of a robot arm at a tabletop "
            f"workspace, exactly as shown in the first frame. Over the following {dur} the robot "
            f"begins to carry out the instruction: {instruction}. The arm moves smoothly and the "
            f"referenced object begins to shift toward the goal. The camera never moves and the "
            f"background is unchanged. The clip ends while the task is still in progress, not "
            f"finished."
        ),
        "audio_description": "",
        "resolution": {"W": w, "H": h},
        "aspect_ratio": f"{w},{h}",
        "duration": dur,
        "fps": int(fps),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--frame", default=str(DEFAULT_FRAME))
    p.add_argument("--instruction", default="Put the marker in the pot")
    p.add_argument("--num-frames", type=int, default=17, help="VAE wants 4k+1.")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--fps", type=float, default=15.0, help="DROID is 15 Hz.")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=6.0, help="I2V reference config uses 6.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw-prompt", action="store_true",
                   help="Send the bare instruction instead of the JSON caption (off-distribution control).")
    p.add_argument("--out", default=str(HERE / "results/smoke_super_i2v"))
    args = p.parse_args()

    from diffusers import Cosmos3OmniPipeline
    from diffusers.utils import export_to_video, load_image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    image = load_image(args.frame)
    print(f"conditioning frame: {args.frame}  size={image.size}")

    seconds = args.num_frames / args.fps
    if args.raw_prompt:
        prompt = args.instruction
    else:
        prompt = json.dumps(build_json_prompt(
            args.instruction, seconds, args.fps, args.width, args.height))
    neg_path = Path(args.model_path) / "assets/negative_prompt.json"
    negative_prompt = json.dumps(json.load(open(neg_path))) if neg_path.exists() else None
    print(f"prompt: {'RAW imperative' if args.raw_prompt else 'JSON caption'} "
          f"({len(prompt)} chars); negative={'yes' if negative_prompt else 'no'}")

    t0 = time.time()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        # Same reasoning as the Nano probes: the video guardrail pixelates faces,
        # which would silently corrupt generated frames relative to the real
        # DROID frames we diff against. Off, and recorded in provenance.
        enable_safety_checker=False,
    )
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s   "
          f"VRAM after load {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    t1 = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    )
    dt = time.time() - t1

    frames = result.video[0] if isinstance(result.video[0], list) else result.video
    print(f"\ninference {dt:.1f}s   frames={len(frames)}  frame0 size={frames[0].size}")

    export_to_video(frames, str(out / "smoke_i2v.mp4"), fps=args.fps, macro_block_size=1)
    frames[0].save(out / "frame_first.png")
    frames[-1].save(out / "frame_last.png")

    rec = {
        "model_path": args.model_path,
        "mode": "image2video",
        "prompt_style": "raw" if args.raw_prompt else "json_caption",
        "instruction": args.instruction,
        "frame": args.frame,
        "source_size": list(image.size),
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "load_s": round(load_s, 1),
        "latency_s": round(dt, 2),
        "out_frames": len(frames),
        "out_frame_size": list(frames[0].size),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
    }
    (out / "smoke_i2v.json").write_text(json.dumps(rec, indent=2))
    print(f"peak VRAM {rec['peak_vram_gb']} GB")
    print(f"wrote {out}/smoke_i2v.json")


if __name__ == "__main__":
    main()
