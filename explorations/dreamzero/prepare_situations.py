"""Extract a few DROID "situations" for the DreamZero experiment.

A *situation* = one timestep (anchor `t`) of one droid_100 episode, loaded via
our existing RLDS adapter (`future/datasets`). For each situation we dump, to
disk, everything the experiment runner needs — so the runner (which lives in the
DreamZero venv, no tfds) never has to touch tensorflow:

  - history frames per camera at offsets [-23,-16,-8,0] (the window the AR
    server consumes; see repo test_client_AR.py RELATIVE_OFFSETS)
  - the anchor's proprioceptive state (joint_position[7], gripper[1], cart[6])
  - ground-truth FUTURE for calibration (condition A): real frames t+1..t+H and
    the logged joint_position / gripper / action over the same horizon
  - the episode's original language instruction(s)

Run in the MAIN vla-distillation venv (has tensorflow-datasets):

    python explorations/dreamzero/prepare_situations.py \
        --dataset-dir datasets/droid/droid_100/1.0.0 --episode-index 0 \
        --num-situations 5

Output: explorations/dreamzero/results/situations/<episode_id>/
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))  # to import the parked RLDS adapter
from future.datasets import build_adapter  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "situations"

# History window the AR server expects (repo test_client_AR.py) and horizon.
OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24
CAMERAS = ("exterior_1", "exterior_2", "wrist")

# How the DreamZero server's roboarena obs keys map onto our canonical cameras.
# NOTE the 0- vs 1-indexing trap: server exterior_image_0 = DROID exterior_1.
SERVER_CAMERA_MAP = {
    "observation/exterior_image_0_left": "exterior_1",  # DROID exterior_image_1_left
    "observation/exterior_image_1_left": "exterior_2",  # DROID exterior_image_2_left
    "observation/wrist_image_left": "wrist",            # DROID wrist_image_left
}


def _off_tag(off: int) -> str:
    return f"m{abs(off)}" if off < 0 else f"{off}"


def _jpeg_to_png(jpeg: bytes, path: Path) -> None:
    from PIL import Image

    Image.open(io.BytesIO(jpeg)).convert("RGB").save(path, format="PNG")


def choose_anchors(length: int, n: int) -> list[int]:
    """Evenly spaced anchors with room for history (>=23) and future (>=H)."""
    lo = -min(OFFSETS) + 1               # need >= 23 frames of history
    hi = length - ACTION_HORIZON - 1     # need a full future horizon
    if hi <= lo:
        raise SystemExit(f"episode too short ({length}) for history+horizon")
    if n == 1:
        return [(lo + hi) // 2]
    step = (hi - lo) / (n - 1)
    return [int(round(lo + i * step)) for i in range(n)]


def build(args: argparse.Namespace) -> Path:
    adapter = build_adapter("droid_rlds")
    ep = adapter.load_episode(args.dataset_dir, index=args.episode_index)
    anchors = (args.anchors if args.anchors else
               choose_anchors(ep.length, args.num_situations))

    out = Path(args.out) if args.out else (DEFAULT_OUT / ep.episode_id)
    out.mkdir(parents=True, exist_ok=True)

    situations = []
    for t in anchors:
        sid = f"t{t:04d}"
        sdir = out / sid
        (sdir / "history").mkdir(parents=True, exist_ok=True)
        (sdir / "future").mkdir(parents=True, exist_ok=True)

        # history frames per camera
        for cam in CAMERAS:
            for off in OFFSETS:
                frame = ep.step(t + off).images[cam]
                _jpeg_to_png(frame, sdir / "history" / f"{cam}_{_off_tag(off)}.png")

        # future frames (for video comparison in condition A)
        horizon = min(ACTION_HORIZON, ep.length - 1 - t)
        for cam in CAMERAS:
            for f in range(1, horizon + 1):
                _jpeg_to_png(ep.step(t + f).images[cam],
                             sdir / "future" / f"{cam}_f{f:02d}.png")

        # ground-truth proprio + future trajectory + logged actions
        anchor = ep.step(t)
        fut_joint = np.array([ep.step(t + f).state["joint_position"] for f in range(1, horizon + 1)], dtype=np.float32)
        fut_grip = np.array([ep.step(t + f).state["gripper_position"] for f in range(1, horizon + 1)], dtype=np.float32)
        logged_action = np.array([ep.step(t + f).action for f in range(horizon)], dtype=np.float32)
        np.savez(
            sdir / "gt.npz",
            proprio_joint=np.array(anchor.state["joint_position"], dtype=np.float32),
            proprio_gripper=np.array(anchor.state["gripper_position"], dtype=np.float32),
            proprio_cartesian=np.array(anchor.state["cartesian_position"], dtype=np.float32),
            future_joint=fut_joint,
            future_gripper=fut_grip,
            logged_action=logged_action,
        )

        (sdir / "situation.json").write_text(json.dumps({
            "situation_id": sid,
            "anchor": t,
            "horizon": horizon,
            "instruction": anchor.instructions[0] if anchor.instructions else "",
            "all_instructions": anchor.instructions,
        }, indent=2))
        situations.append(sid)
        print(f"  {sid}: anchor={t} horizon={horizon} instr={anchor.instructions[:1]}")

    meta = {
        "dataset": ep.dataset,
        "episode_id": ep.episode_id,
        "record_uri": ep.record_uri,
        "episode_index": args.episode_index,
        "length": ep.length,
        "fps": ep.metadata.get("fps"),
        "offsets": OFFSETS,
        "action_horizon": ACTION_HORIZON,
        "cameras": list(CAMERAS),
        "server_camera_map": SERVER_CAMERA_MAP,
        "situations": situations,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(situations)} situations to {out}")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default=str(PROJECT_ROOT / "datasets" / "droid" / "droid_100" / "1.0.0"))
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--num-situations", type=int, default=5)
    p.add_argument("--anchors", type=int, nargs="+", default=None,
                   help="Explicit anchor timesteps (overrides --num-situations).")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
