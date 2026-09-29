"""Dense-description prompt-tuning sets: split hygiene and clip selection."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_dense_eval_sets import CLIP_FRAMES, EXAMPLES_YAML, MANIFEST, SCENE_SPLIT, classify_window

CLIPS = yaml.safe_load(MANIFEST.read_text())["clips"]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def test_no_scene_appears_in_both_splits():
    train = {c["scene"] for c in CLIPS if c["split"] == "train"}
    test = {c["scene"] for c in CLIPS if c["split"] == "test"}
    assert train and test and not train & test
    assert all(SCENE_SPLIT[c["scene"]] == c["split"] for c in CLIPS)


def test_instructions_and_episodes_are_unique_across_both_splits():
    instructions = [_norm(c["instruction"]) for c in CLIPS]
    episodes = [c["source"]["episode_id"] for c in CLIPS]
    assert len(set(instructions)) == len(instructions)
    assert len(set(episodes)) == len(episodes)


def test_fewshot_example_episodes_are_excluded():
    fewshot = {(ex["record"], ex["example_index"]) for ex in yaml.safe_load(EXAMPLES_YAML.read_text())}
    used = {(c["source"]["record"], c["source"]["example_index"]) for c in CLIPS}
    assert not fewshot & used


def test_clip_indices_are_contiguous_per_split():
    for split in ("train", "test"):
        idx = sorted(c["example_index"] for c in CLIPS if c["split"] == split)
        assert idx == list(range(len(idx)))
        ids = [c["id"] for c in CLIPS if c["split"] == split]
        assert len(set(ids)) == len(ids)


def _window(g_start, g_end, move_m):
    g = np.linspace(g_start, g_end, CLIP_FRAMES)
    xyz = np.zeros((CLIP_FRAMES, 3))
    xyz[:, 0] = np.linspace(0, move_m, CLIP_FRAMES)
    return g, xyz


def test_classify_window_event_types():
    assert classify_window(*_window(0.0, 0.9, 0.05), 0)[0] == "grasp"
    assert classify_window(*_window(0.9, 0.0, 0.05), 0)[0] == "release"
    assert classify_window(*_window(0.8, 0.8, 0.3), 0)[0] == "carry"
    assert classify_window(*_window(0.0, 0.0, 0.3), 0)[0] == "reach"
    assert classify_window(*_window(0.8, 0.8, 0.06), 0)[0] == "fine"
    assert classify_window(*_window(0.0, 0.0, 0.01), 0) is None  # idle


def test_classify_window_skips_transient_grasp_and_release():
    g = np.concatenate([np.zeros(20), np.full(20, 0.9), np.zeros(20)])
    assert classify_window(g, np.zeros((CLIP_FRAMES, 3)), 0) is None
