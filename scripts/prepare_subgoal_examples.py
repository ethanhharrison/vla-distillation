"""Prepare a Stage B example set from a DROID RLDS episode.

Extracts a handful of trajectory steps from one episode of the public DROID
`droid_100` build and writes them to disk in the simple, dataset-neutral layout
that `pipeline.subgoal_image.generate` consumes. Each example carries the three
camera frames, the real t+k *future* frames (for the `real_future` subgoal
source), the language instruction(s), and the proprioceptive state — so Stage B
can run standalone against real data.

    outputs/subgoal_examples/<episode_id>/
        meta.json
        images/step0020_exterior_1.jpg ...

Usage:
    python scripts/prepare_subgoal_examples.py \
        --dataset-dir datasets/droid/droid_100/1.0.0 \
        --episode-index 0 --num-examples 10 --interval 12 --start 20 --k 30
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "subgoal_examples"

# DROID observation feature -> canonical camera role (quarantined here).
CAMERA_MAP = {
    "exterior_1": "exterior_image_1_left",
    "exterior_2": "exterior_image_2_left",
    "wrist": "wrist_image_left",
}
INSTRUCTION_KEYS = ("language_instruction", "language_instruction_2", "language_instruction_3")
DROID_FPS = 15


def _encode_jpeg(arr) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _episode_id(file_path: str, index: int) -> str:
    parts = [p for p in file_path.split("/") if p]
    return parts[-2] if len(parts) >= 2 else f"episode-{index}"


def load_episode_steps(dataset_dir: str, episode_index: int):
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(dataset_dir)
    ds = builder.as_dataset(split="train")
    for i, ep in enumerate(ds):
        if i == episode_index:
            md = {
                k: (v.numpy().decode("utf-8", "replace") if v.dtype.name == "string"
                    else v.numpy().tolist())
                for k, v in ep["episode_metadata"].items()
            }
            steps = list(ep["steps"])
            return md, steps
    raise IndexError(f"{dataset_dir} has no episode at index {episode_index}")


def build_examples(args: argparse.Namespace) -> Path:
    md, steps = load_episode_steps(args.dataset_dir, args.episode_index)
    length = len(steps)
    episode_id = _episode_id(md.get("file_path", ""), args.episode_index)
    cameras = args.cameras or list(CAMERA_MAP.keys())

    out = Path(args.out) if args.out else (DEFAULT_OUT / episode_id)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # choose step indices
    idxs = list(range(args.start, length, args.interval))[: args.num_examples]
    if not idxs:
        raise SystemExit(f"no steps selected (episode length={length}, start={args.start})")

    def frame_bytes(step, role):
        return _encode_jpeg(step["observation"][CAMERA_MAP[role]].numpy())

    def instructions(step):
        out_i = []
        for key in INSTRUCTION_KEYS:
            if key in step:
                t = step[key].numpy().decode("utf-8", "replace").strip()
                if t and t not in out_i:
                    out_i.append(t)
        return out_i

    examples = []
    for si in idxs:
        step = steps[si]
        fi = min(si + args.k, length - 1)
        clamped = (si + args.k) > (length - 1)
        instrs = instructions(step)

        cam_files, fut_files = {}, {}
        for role in cameras:
            p = img_dir / f"step{si:04d}_{role}.jpg"
            p.write_bytes(frame_bytes(step, role))
            cam_files[role] = f"images/{p.name}"
            fp = img_dir / f"step{fi:04d}_{role}.jpg"
            if not fp.exists():
                fp.write_bytes(frame_bytes(steps[fi], role))
            fut_files[role] = f"images/{fp.name}"

        obs = step["observation"]
        examples.append({
            "id": f"step{si:04d}",
            "step_index": si,
            "instruction": instrs[0] if instrs else "",
            "all_instructions": instrs,
            "cameras": cam_files,
            "future": {
                "k": args.k, "step_index": fi, "clamped": clamped, "cameras": fut_files,
            },
            "state": {
                "joint_position": obs["joint_position"].numpy().tolist(),
                "gripper_position": obs["gripper_position"].numpy().tolist(),
                "cartesian_position": obs["cartesian_position"].numpy().tolist(),
                "source": "real@step",
            },
            "action": step["action"].numpy().tolist(),
        })

    meta = {
        "dataset": "droid",
        "episode_id": episode_id,
        "record_uri": args.dataset_dir,
        "episode_index": args.episode_index,
        "trajectory_length": length,
        "fps": DROID_FPS,
        "k_default": args.k,
        "cameras": cameras,
        "file_path": md.get("file_path"),
        "examples": examples,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(examples)} examples (episode {episode_id!r}, length {length}) to {out}")
    print(f"  step indices: {idxs}")
    print(f"  instruction of first example: {examples[0]['instruction']!r}")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default=str(PROJECT_ROOT / "datasets" / "droid" / "droid_100" / "1.0.0"),
                   help="Path to the DROID RLDS version dir (contains dataset_info.json).")
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--num-examples", type=int, default=10)
    p.add_argument("--interval", type=int, default=12, help="Steps between examples.")
    p.add_argument("--start", type=int, default=20, help="First step index to sample.")
    p.add_argument("--k", type=int, default=30, help="Future offset for real_future (steps; ~2s @15fps).")
    p.add_argument("--cameras", nargs="+", default=None, help="Canonical camera roles (default: all).")
    p.add_argument("--out", default=None, help="Output example-set dir.")
    return p.parse_args(argv)


if __name__ == "__main__":
    build_examples(parse_args())
