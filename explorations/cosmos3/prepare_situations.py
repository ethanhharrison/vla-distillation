"""Extract one situation per DROID episode, anchored at the episode's FIRST frame.

Differs from the DreamZero situation set on purpose. That one was five anchors
along a *single* trajectory, so every counterfactual instruction had to be about
the same pot and marker — scene diversity was zero and the instruction-
sensitivity measurement was confounded with it. This walks N *different*
episodes and takes each one's opening frame, so each situation is a distinct
scene with its own real task.

Only 46 of the 100 `droid_100` episodes carry a language instruction; the rest
have empty instruction fields and cannot serve as condition A, so they are
skipped.

Writes the same on-disk layout the runner and report already consume:

  <out>/<situation_id>/history/<cam>_0.png     the anchor (conditioning) frame
  <out>/<situation_id>/future/<cam>_fNN.png    real frames t+1..t+H (the yardstick)
  <out>/<situation_id>/situation.json
  <out>/meta.json

No proprioception or history window is written: Cosmos conditions on a single
frame and this path takes no state (unlike DreamZero, which needed both).

`real_motion` is recorded per situation — mean absolute pixel difference between
the anchor and t+H on exterior_1. DROID episodes often idle at the start, and a
situation whose real_motion sits near the model's ~6-7 VAE reconstruction floor
cannot show a meaningful subgoal no matter what the model does.

Run in the MAIN venv (has tensorflow-datasets):

    python explorations/cosmos3/prepare_situations.py --episode-indexes 0 1 2 3 10 13 20 21
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "situations_multitraj"
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "droid" / "droid_100" / "1.0.0"

# DROID observation feature -> canonical camera role.
CAMERA_MAP = {
    "exterior_1": "exterior_image_1_left",
    "exterior_2": "exterior_image_2_left",
    "wrist": "wrist_image_left",
}
INSTRUCTION_KEYS = ("language_instruction", "language_instruction_2", "language_instruction_3")
FPS = 15


def _save_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(arr.astype(np.uint8)).convert("RGB").save(path, format="PNG")


def _episode_id(file_path: str, index: int) -> str:
    parts = [p for p in file_path.split("/") if p]
    return parts[-2] if len(parts) >= 2 else f"episode-{index}"


def build(args: argparse.Namespace) -> Path:
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(args.dataset_dir))
    ds = builder.as_dataset(split="train")

    wanted = set(args.episode_indexes) if args.episode_indexes else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    situations, records = [], []
    for i, ep in enumerate(ds):
        if wanted is not None and i not in wanted:
            continue
        if wanted is None and len(situations) >= args.num_episodes:
            break

        md = {k: (v.numpy().decode("utf-8", "replace") if v.dtype.name == "string" else v.numpy().tolist())
              for k, v in ep["episode_metadata"].items()}
        steps = list(ep["steps"])
        s0 = steps[args.anchor] if args.anchor < len(steps) else None
        if s0 is None:
            print(f"  skip {i}: episode shorter than anchor {args.anchor}")
            continue

        instrs: list[str] = []
        for key in INSTRUCTION_KEYS:
            if key in s0:
                t = s0[key].numpy().decode("utf-8", "replace").strip()
                if t and t not in instrs:
                    instrs.append(t)
        if not instrs:
            if wanted is not None:
                print(f"  skip {i}: no language instruction (cannot serve as condition A)")
            continue

        horizon = min(args.horizon, len(steps) - 1 - args.anchor)
        if horizon < args.min_horizon:
            print(f"  skip {i}: only {horizon} future frames after anchor {args.anchor}")
            continue

        eid = _episode_id(md.get("file_path", ""), i)
        sid = f"ep{i:03d}_t{args.anchor:04d}"
        sdir = out / sid
        (sdir / "history").mkdir(parents=True, exist_ok=True)
        (sdir / "future").mkdir(parents=True, exist_ok=True)

        frames = {}
        for role, feat in CAMERA_MAP.items():
            frames[role] = steps[args.anchor]["observation"][feat].numpy()
            _save_png(frames[role], sdir / "history" / f"{role}_0.png")
            for f in range(1, horizon + 1):
                _save_png(steps[args.anchor + f]["observation"][feat].numpy(),
                          sdir / "future" / f"{role}_f{f:02d}.png")

        last = steps[args.anchor + horizon]["observation"][CAMERA_MAP["exterior_1"]].numpy()
        real_motion = float(np.abs(last.astype(float) - frames["exterior_1"].astype(float)).mean())

        (sdir / "situation.json").write_text(json.dumps({
            "situation_id": sid,
            "episode_index": i,
            "episode_id": eid,
            "episode_length": len(steps),
            "anchor": args.anchor,
            "horizon": horizon,
            "instruction": instrs[0],
            "all_instructions": instrs,
            "real_motion": round(real_motion, 2),
        }, indent=2))
        situations.append(sid)
        records.append({"situation_id": sid, "episode_index": i, "episode_id": eid,
                        "instruction": instrs[0], "real_motion": round(real_motion, 2),
                        "horizon": horizon, "length": len(steps)})
        print(f"  {sid}  {eid:26} len={len(steps):4} H={horizon:3} "
              f"motion={real_motion:5.1f}  “{instrs[0][:46]}”")

    meta = {
        "dataset": "droid_100",
        "record_uri": str(args.dataset_dir),
        "layout": "one situation per episode, anchored at the episode's first frame",
        "anchor": args.anchor,
        "horizon": args.horizon,
        "fps": FPS,
        "cameras": list(CAMERA_MAP),
        "episodes": records,
        "situations": situations,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    weak = [r["situation_id"] for r in records if r["real_motion"] < 7.0]
    print(f"\nWrote {len(situations)} situations (one per episode) to {out}")
    if weak:
        print(f"NOTE: {len(weak)} situation(s) have real_motion below the ~7 VAE floor "
              f"and cannot show a meaningful subgoal: {', '.join(weak)}")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    p.add_argument("--episode-indexes", type=int, nargs="+", default=None,
                   help="Explicit episode indexes (default: first --num-episodes with an instruction).")
    p.add_argument("--num-episodes", type=int, default=8)
    p.add_argument("--anchor", type=int, default=0, help="Timestep within each episode (0 = first frame).")
    p.add_argument("--horizon", type=int, default=32, help="Real future frames to dump for comparison.")
    p.add_argument("--min-horizon", type=int, default=16)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
