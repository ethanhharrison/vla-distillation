"""Run the DreamZero A/B/C experiment against a running inference server.

Runs in the DreamZero venv (reuses the repo's WebsocketClientPolicy). Reads the
situation set produced by prepare_situations.py and, per situation, drives the AR
server once per (condition, instruction): sends the initial frame + the 4-frame
history window (with the correct server camera-key remap + real proprio), gets a
(24, 8) joint-position+gripper action chunk, and grabs the generated video the
server saves on reset. Everything is written under results/runs/<sample_id>/.

Conditions (instructions come from conditions.json per situation, written by
prepare_instructions.py; falls back to condition A only):
  A = original instruction ; B = counterfactual (Stage A) ; C = null/stress

Usage (server must be running — see serve.sh):
    python explorations/dreamzero/run_experiment.py \
        --situations results/situations/<episode> --port 8901 \
        --model-path /home/tiger/proj/staging/vla/models/DreamZero-DROID
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "repo"))
import imageio.v2 as imageio  # noqa: E402
from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402

OFFSETS = [-23, -16, -8, 0]
SERVER_CAMERA_MAP = {
    "observation/exterior_image_0_left": "exterior_1",
    "observation/exterior_image_1_left": "exterior_2",
    "observation/wrist_image_left": "wrist",
}


def _off_tag(off: int) -> str:
    return f"m{abs(off)}" if off < 0 else f"{off}"


def load_history(sdir: Path) -> dict[str, np.ndarray]:
    """Return {server_key: (4,H,W,3) uint8} from a situation's history PNGs."""
    out = {}
    for skey, cam in SERVER_CAMERA_MAP.items():
        frames = [imageio.imread(sdir / "history" / f"{cam}_{_off_tag(o)}.png") for o in OFFSETS]
        out[skey] = np.stack(frames, axis=0).astype(np.uint8)
    return out


def make_obs(hist, proprio, frame_indices, prompt, session_id):
    obs = {}
    for skey, allframes in hist.items():
        sel = allframes[frame_indices]
        obs[skey] = sel[0] if len(frame_indices) == 1 else sel
    obs["observation/joint_position"] = proprio["joint"].astype(np.float32)
    obs["observation/cartesian_position"] = proprio["cart"].astype(np.float32)
    obs["observation/gripper_position"] = proprio["grip"].astype(np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def find_output_dir(model_path: str) -> Path | None:
    # The server saves videos to  <parent-of-model_path>/real_world_eval_gen_*/<ckpt_name>/
    mp = os.path.normpath(model_path)
    parent, name = os.path.dirname(mp), os.path.basename(mp)
    cands = sorted(glob.glob(os.path.join(parent, "real_world_eval_gen_*", name)))
    cands = [c for c in cands if os.path.isdir(c)]
    return Path(cands[-1]) if cands else None


def run(args):
    sit_dir = Path(args.situations)
    meta = json.loads((sit_dir / "meta.json").read_text())
    runs_dir = HERE / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for sid in meta["situations"]:
        sdir = sit_dir / sid
        sit = json.loads((sdir / "situation.json").read_text())
        gt = np.load(sdir / "gt.npz")
        proprio = {"joint": gt["proprio_joint"], "grip": gt["proprio_gripper"], "cart": gt["proprio_cartesian"]}
        hist = load_history(sdir)

        cond_path = sdir / "conditions.json"
        if cond_path.exists():
            conditions = json.loads(cond_path.read_text())
        else:
            conditions = {"A": [sit["instruction"]]}

        for cond, instrs in conditions.items():
            for j, instruction in enumerate(instrs):
                sample_id = f"{sid}__{cond}{j}"
                out = runs_dir / sample_id
                out.mkdir(parents=True, exist_ok=True)
                session_id = str(uuid.uuid4())
                client = WebsocketClientPolicy(host=args.host, port=args.port)

                outdir = find_output_dir(args.model_path)
                before = set(glob.glob(str(outdir / "*.mp4"))) if outdir else set()

                t0 = time.time()
                client.infer(make_obs(hist, proprio, [0], instruction, session_id))  # init
                actions = client.infer(make_obs(hist, proprio, [0, 1, 2, 3], instruction, session_id))
                dt = time.time() - t0
                client.reset({})

                actions = np.asarray(actions)
                np.save(out / "action_chunk.npy", actions)

                # grab the newly-saved generated video
                video_rel = None
                outdir = outdir or find_output_dir(args.model_path)
                if outdir:
                    after = set(glob.glob(str(outdir / "*.mp4")))
                    new = sorted(after - before)
                    if new:
                        shutil.copy(new[-1], out / "generated.mp4")
                        video_rel = "generated.mp4"

                rec = {
                    "sample_id": sample_id, "situation_id": sid, "anchor": sit["anchor"],
                    "condition": cond, "instruction": instruction,
                    "action_shape": list(actions.shape),
                    "action_min": float(actions.min()), "action_max": float(actions.max()),
                    "latency_s": round(dt, 2), "video": video_rel,
                }
                (out / "sample.json").write_text(json.dumps(rec, indent=2))
                samples.append(rec)
                print(f"  {sample_id}: {cond} '{instruction[:40]}' -> action{actions.shape} "
                      f"[{actions.min():.3f},{actions.max():.3f}] {dt:.1f}s video={video_rel}")

    index = {"situations_dir": str(sit_dir), "episode_id": meta["episode_id"],
             "model_path": args.model_path, "samples": samples}
    (runs_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nWrote {len(samples)} samples to {runs_dir}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--situations", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8901)
    p.add_argument("--model-path", default="/home/tiger/proj/staging/vla/models/DreamZero-DROID")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
