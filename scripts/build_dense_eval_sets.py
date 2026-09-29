"""Build scene-disjoint train/test clip sets for iterating on the dense-
description prompt (see pipeline/dense_description/eval_sets/README.md).

Two stages:

  select   Scan the source shards, pick one informative 4s clip per chosen
           episode (a grasp, release, carry, reach or fine-manipulation
           window, found from the robot's gripper/cartesian state), and write
           the manifest YAML. Scenes come from eval_sets/scene_groups.json;
           each scene belongs wholly to one split, so no room appears in both.
  extract  Copy each manifest clip (plus 1s of context either side) into a
           small partner-format .tfrecord per split, so load_trajectory and
           every existing generator/judge work on it unchanged and fast.

Usage:
    python scripts/build_dense_eval_sets.py select
    python scripts/build_dense_eval_sets.py extract
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # to import the pipeline package

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.core.framework import tensor_pb2

EVAL_DIR = PROJECT_ROOT / "pipeline" / "dense_description" / "eval_sets"
SCENE_GROUPS = EVAL_DIR / "scene_groups.json"
MANIFEST = EVAL_DIR / "dense_eval_v1.yaml"
EXAMPLES_YAML = PROJECT_ROOT / "pipeline" / "dense_description" / "fewshot_examples.yaml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets" / "dense_eval"

FPS = 15
CLIP_FRAMES = 60  # 4s @ 15fps - the "current ideal" clip length
PAD_FRAMES = 15  # 1s of context kept either side of the clip when extracting
WINDOW_STRIDE = 5

# Scene -> split. Test scenes are never seen in training; CLVR (one office)
# is held out entirely so the test set also covers an unseen lab. Every
# few-shot example lives in TRI-stove-kitchen, which is therefore train-only.
SCENE_SPLIT = {
    "IPRL-bedroom": "train",
    "TRI-stove-kitchen": "train",
    "TRI-window-kitchen": "train",
    "IRIS-wood-table-apartment": "train",
    "IRIS-granite-kitchen": "train",
    "CLVR-office": "test",
    "IPRL-kitchen": "test",
    "TRI-sink-kitchen": "test",
    "IRIS-white-cabinets": "test",
    "IRIS-round-table": "test",
}
# Clips drawn per scene (one clip per episode): 40 train / 30 test.
SCENE_QUOTA = {
    "IPRL-bedroom": 9, "TRI-stove-kitchen": 9, "TRI-window-kitchen": 8,
    "IRIS-wood-table-apartment": 9, "IRIS-granite-kitchen": 5,
    "CLVR-office": 10, "IPRL-kitchen": 6, "TRI-sink-kitchen": 6,
    "IRIS-white-cabinets": 5, "IRIS-round-table": 3,
}
# Target mix of event types within each scene, in priority order.
EVENT_TYPES = ("grasp", "release", "carry", "reach", "fine")

GRIP_OPEN = 0.1  # gripper_position below this = open
GRIP_CLOSED = 0.3  # above this = closed (on an object, closure is ~0.4-1.0)
BIG_MOVE_M = 0.12  # end-effector displacement over the clip for carry/reach
SMALL_MOVE_M = 0.03  # minimum displacement for a "fine" (holding, small motion) clip


def _array(raw: bytes) -> np.ndarray:
    proto = tensor_pb2.TensorProto()
    proto.ParseFromString(raw)
    return tf.make_ndarray(proto)


def _norm_instruction(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def classify_window(gripper: np.ndarray, xyz: np.ndarray, start: int) -> tuple[str, float] | None:
    """Event type and a quality score for the clip starting at `start`, or
    None if nothing clear happens (idle, or a transient open/close that
    returns to where it started)."""
    g = gripper[start : start + CLIP_FRAMES]
    p = xyz[start : start + CLIP_FRAMES]
    g0, g1 = g[:3].mean(), g[-3:].mean()
    moved = float(np.linalg.norm(p[-1] - p[0]))
    if g.max() - g.min() > GRIP_CLOSED and abs(g1 - g0) < GRIP_CLOSED / 2:
        return None  # grasp-and-release (or vice versa) inside one clip
    # For state changes, prefer the change happening mid-clip.
    crossing = int(np.argmax(np.abs(g - g0) > (GRIP_CLOSED - GRIP_OPEN) / 2))
    centered = 1.0 - abs(crossing / CLIP_FRAMES - 0.5) * 2
    if g0 < GRIP_OPEN and g1 > GRIP_CLOSED:
        return "grasp", centered + moved
    if g0 > GRIP_CLOSED and g1 < GRIP_OPEN:
        return "release", centered + moved
    if g0 > GRIP_CLOSED and g1 > GRIP_CLOSED:
        if moved > BIG_MOVE_M:
            return "carry", moved
        if moved > SMALL_MOVE_M:
            return "fine", moved
    if g0 < GRIP_OPEN and g1 < GRIP_OPEN and moved > BIG_MOVE_M:
        return "reach", moved
    return None


def excluded_episodes() -> set[tuple[str, int]]:
    """Few-shot examples (labeled or not) and the earlier ablation query
    clips - kept out of both splits."""
    out = {(ex["record"], ex["example_index"]) for ex in yaml.safe_load(EXAMPLES_YAML.read_text())}
    out |= {("datasets/droid/success/success-00285.tfrecord", i) for i in (5, 12, 13, 14)}
    return out


def scan_candidates(scene_groups: dict) -> list[dict]:
    """One entry per usable episode: metadata plus its best window per event type."""
    by_record = defaultdict(dict)
    for episode_id, info in scene_groups.items():
        by_record[info["record"]][info["example_index"]] = {"episode_id": episode_id, **info}
    skip = excluded_episodes()
    candidates = []
    for record, episodes in sorted(by_record.items()):
        print(f"scanning {record}")
        for index, raw in enumerate(tf.data.TFRecordDataset([str(PROJECT_ROOT / record)])):
            info = episodes.get(index)
            if info is None or info["scene"] not in SCENE_SPLIT or (record, index) in skip:
                continue
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            f = ex.features.feature
            instruction = f["language_instruction1"].bytes_list.value[0].decode("utf-8").strip()
            if not _norm_instruction(instruction) or _norm_instruction(instruction) == "no action":
                continue
            gripper = _array(f["observation/robot_state/gripper_position"].bytes_list.value[0]).ravel()
            xyz = _array(f["observation/robot_state/cartesian_position"].bytes_list.value[0])[:, :3]
            windows: dict[str, tuple[float, int]] = {}
            for start in range(PAD_FRAMES, len(gripper) - CLIP_FRAMES - PAD_FRAMES + 1, WINDOW_STRIDE):
                hit = classify_window(gripper, xyz, start)
                if hit and hit[1] > windows.get(hit[0], (-1.0, 0))[0]:
                    windows[hit[0]] = (hit[1], start)
            if windows:
                candidates.append({**info, "instruction": instruction, "length": len(gripper), "windows": windows})
    return candidates


def select(seed: int) -> None:
    scene_groups = json.loads(SCENE_GROUPS.read_text())
    candidates = scan_candidates(scene_groups)
    rng = random.Random(seed)
    by_scene = defaultdict(list)
    for c in candidates:
        by_scene[c["scene"]].append(c)

    clips, used_instructions = [], set()
    for scene in SCENE_SPLIT:  # train scenes first, so test instructions are checked against them
        pool = by_scene[scene]
        rng.shuffle(pool)
        chosen, counts = [], Counter()
        while len(chosen) < SCENE_QUOTA[scene]:
            # Take the least-represented event type that some remaining episode can supply.
            pick = None
            for event in sorted(EVENT_TYPES, key=lambda e: (counts[e], EVENT_TYPES.index(e))):
                pick = next((c for c in pool if event in c["windows"]
                             and _norm_instruction(c["instruction"]) not in used_instructions), None)
                if pick:
                    break
            if pick is None:
                break
            pool.remove(pick)
            used_instructions.add(_norm_instruction(pick["instruction"]))
            counts[event] += 1
            start = pick["windows"][event][1]
            chosen.append({
                "id": f"{SCENE_SPLIT[scene]}-{len([c for c in clips if c['split'] == SCENE_SPLIT[scene]]) + len(chosen):03d}",
                "split": SCENE_SPLIT[scene],
                "scene": scene,
                "org": pick["org"],
                "event": event,
                "instruction": pick["instruction"],
                "source": {"record": pick["record"], "example_index": pick["example_index"],
                           "episode_id": pick["episode_id"], "start_step": start,
                           "end_step": start + CLIP_FRAMES - 1, "trajectory_length": pick["length"]},
            })
        if len(chosen) < SCENE_QUOTA[scene]:
            print(f"warning: {scene} supplied only {len(chosen)}/{SCENE_QUOTA[scene]} clips")
        clips += chosen

    # Position of each clip inside its split's extracted .tfrecord.
    for split in ("train", "test"):
        for i, clip in enumerate(c for c in clips if c["split"] == split):
            clip["example_index"] = i
            clip["start_step"] = PAD_FRAMES
            clip["end_step"] = PAD_FRAMES + CLIP_FRAMES - 1
            clip["clip_seconds"] = CLIP_FRAMES / FPS

    header = (
        "# Dense-description prompt-tuning sets (v1). Generated by scripts/build_dense_eval_sets.py select;\n"
        "# see eval_sets/README.md. example_index/start_step/end_step index into\n"
        "# datasets/dense_eval/<split>.tfrecord (built by `extract`); `source` is the original clip.\n"
    )
    MANIFEST.write_text(header + yaml.safe_dump({"seed": seed, "clips": clips}, sort_keys=False, width=110))
    for split in ("train", "test"):
        sel = [c for c in clips if c["split"] == split]
        print(f"{split}: {len(sel)} clips | scenes {dict(Counter(c['scene'] for c in sel))} | "
              f"events {dict(Counter(c['event'] for c in sel))}")
    print(f"Wrote {MANIFEST}")


def extract(out_dir: Path) -> None:
    clips = yaml.safe_load(MANIFEST.read_text())["clips"]
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = defaultdict(list)  # (record, index) -> clips
    for clip in clips:
        wanted[(clip["source"]["record"], clip["source"]["example_index"])].append(clip)
    trimmed: dict[str, tf.train.Example] = {}
    for record in sorted({r for r, _ in wanted}):
        print(f"reading {record}")
        for index, raw in enumerate(tf.data.TFRecordDataset([str(PROJECT_ROOT / record)])):
            for clip in wanted.get((record, index), []):
                ex = tf.train.Example()
                ex.ParseFromString(raw.numpy())
                f = ex.features.feature
                lo = clip["source"]["start_step"] - PAD_FRAMES
                hi = clip["source"]["end_step"] + PAD_FRAMES + 1
                out = tf.train.Example()
                for key, feat in f.items():
                    if key in ("shoulder_image_1", "shoulder_image_2", "wrist_image"):
                        out.features.feature[key].bytes_list.value.extend(feat.bytes_list.value[lo:hi])
                    elif key.startswith(("observation/", "action/")):
                        continue  # robot state isn't needed downstream; keeps files small
                    else:
                        out.features.feature[key].CopyFrom(feat)
                out.features.feature["traj_len"].int64_list.value[:] = [hi - lo]
                out.features.feature["clip_id"].bytes_list.value.append(clip["id"].encode())
                trimmed[clip["id"]] = out
    for split in ("train", "test"):
        path = out_dir / f"{split}.tfrecord"
        with tf.io.TFRecordWriter(str(path)) as writer:
            for clip in sorted((c for c in clips if c["split"] == split), key=lambda c: c["example_index"]):
                writer.write(trimmed[clip["id"]].SerializeToString())
        print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["select", "extract"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)
    if args.stage == "select":
        select(args.seed)
    else:
        extract(Path(args.out_dir))


if __name__ == "__main__":
    main()
