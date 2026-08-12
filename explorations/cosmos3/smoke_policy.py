"""One-call smoke test: does Cosmos3-Nano-Policy-DROID work through diffusers?

The open question from the research pass. The checkpoint was post-trained on a
540x640 three-view canvas + proprioception, predicting 32 absolute joint
positions at 15 Hz (tech report 4.2.5) — but the diffusers action path offers
only a single conditioning frame, no proprio, and the generic `droid_lerobot`
domain (10D end-effector pose deltas + gripper, normalized). This runs exactly
one policy call on a real DROID frame and reports what actually comes back, so
we know whether the experiment runs on this checkpoint or the general Nano.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python smoke_policy.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "/home/tiger/proj/staging/vla/models/Cosmos3-Nano-Policy-DROID"
DEFAULT_FRAME = (
    HERE.parent / "dreamzero/results/situations/Mon_Apr_17_14:48:05_2023/t0024/history/exterior_1_0.png"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--frame", default=str(DEFAULT_FRAME))
    p.add_argument("--prompt", default="Put the marker in the pot")
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--resolution-tier", type=int, default=256)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--out", default=str(HERE / "results/smoke"))
    args = p.parse_args()

    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
    from diffusers.utils import export_to_video, load_image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    image = load_image(args.frame)
    print(f"conditioning frame: {args.frame}  size={image.size}")

    t0 = time.time()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        # Guardrails would pull the gated nvidia/Cosmos-1.0-Guardrail repo and
        # pixelate faces in generated frames — a silent corruption relative to
        # the real frames we compare against. Off, and recorded in provenance.
        enable_safety_checker=False,
    )
    print(f"loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    result = pipe(
        prompt=args.prompt,
        action=CosmosActionCondition(
            mode="policy",
            chunk_size=args.chunk_size,
            domain_name="droid_lerobot",
            resolution_tier=args.resolution_tier,
            image=image,
            # DROID exteriors look at the robot from the front, not ego/wrist.
            view_point="third_person_view",
        ),
        fps=5,
        num_inference_steps=args.steps,
        guidance_scale=1.0,
        use_system_prompt=False,
    )
    dt = time.time() - t1

    frames = result.video
    print(f"\ninference {dt:.1f}s   frames={len(frames)}  frame0 size={frames[0].size}")
    export_to_video(frames, str(out / "smoke.mp4"), fps=5, macro_block_size=1)

    rec = {
        "model_path": args.model_path,
        "prompt": args.prompt,
        "frame": args.frame,
        "chunk_size": args.chunk_size,
        "resolution_tier": args.resolution_tier,
        "steps": args.steps,
        "latency_s": round(dt, 2),
        "num_frames": len(frames),
        "frame_size": list(frames[0].size),
        "action": None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
    }
    if result.action is not None:
        a = np.asarray(result.action[0].float().cpu())
        np.save(out / "smoke_action.npy", a)
        rec["action"] = {
            "shape": list(a.shape),
            "min": float(a.min()),
            "max": float(a.max()),
            "mean_abs": float(np.abs(a).mean()),
            "per_dim_mean_abs": [round(float(x), 4) for x in np.abs(a).mean(axis=0)],
        }
        print(f"action shape={a.shape} range=[{a.min():.3f},{a.max():.3f}] mean|a|={np.abs(a).mean():.3f}")
    else:
        print("action: None  <-- the checkpoint did not emit an action chunk")

    print(f"peak VRAM {rec['peak_vram_gb']} GB")
    (out / "smoke.json").write_text(json.dumps(rec, indent=2))
    print(f"wrote {out}/smoke.json")


if __name__ == "__main__":
    main()
