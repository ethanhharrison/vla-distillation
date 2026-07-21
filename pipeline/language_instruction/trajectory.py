"""Load DROID tfrecords and expose per-step camera frames.

A DROID tfrecord stores each episode as a single `tf.train.Example`. The image
features (`shoulder_image_1`, `shoulder_image_2`, `wrist_image`) are stored as a
`bytes_list` with one JPEG per timestep, so `images[camera][step]` is the raw
JPEG bytes for that camera at that step.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import tensorflow as tf

DEFAULT_CAMERAS = ("shoulder_image_1", "shoulder_image_2", "wrist_image")

@dataclass
class Trajectory:
    """A single episode's image streams plus lightweight metadata."""

    record_path: str
    length: int
    images: dict[str, list[bytes]]
    metadata: dict = field(default_factory=dict)

    @property
    def cameras(self) -> list[str]:
        return list(self.images.keys())

    def frame(self, step: int, cameras: tuple[str, ...] | None = None) -> dict[str, bytes]:
        """Return the JPEG bytes for each requested camera at a given step."""
        selected = cameras if cameras is not None else self.cameras
        return {cam: self.images[cam][step] for cam in selected if cam in self.images}

    def steps(self, interval: int, max_steps: int | None = None) -> list[int]:
        """Step indices sampled every `interval` frames, optionally capped."""
        if interval < 1:
            raise ValueError("interval must be >= 1")
        stop = self.length if max_steps is None else min(self.length, max_steps)
        return list(range(0, stop, interval))

def decode_metadata(features) -> dict:
    """Pull a few human-readable scalar fields out of the example, if present."""
    metadata: dict = {}
    text_keys = (
        "episode_id",
        "language_instruction1",
        "language_instruction2",
        "language_instruction3",
        "org",
        "rel_path",
        "split",
    )
    for key in text_keys:
        if key in features and features[key].bytes_list.value:
            try:
                metadata[key] = features[key].bytes_list.value[0].decode("utf-8")
            except UnicodeDecodeError:
                continue
    for key in ("traj_len", "image_height", "image_width"):
        if key in features and features[key].int64_list.value:
            metadata[key] = features[key].int64_list.value[0]
    return metadata

def load_trajectories(
    record_path: str | Path,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
) -> Iterator[Trajectory]:
    """Yield one `Trajectory` per example stored in the tfrecord."""
    raw_dataset = tf.data.TFRecordDataset([str(record_path)])
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        images: dict[str, list[bytes]] = {}
        for cam in cameras:
            if cam in features:
                images[cam] = list(features[cam].bytes_list.value)

        length = min((len(v) for v in images.values()), default=0)
        yield Trajectory(
            record_path=str(record_path),
            length=length,
            images=images,
            metadata=decode_metadata(features),
        )

def load_trajectory(
    record_path: str | Path,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    index: int = 0,
) -> Trajectory:
    """Load a single episode (the `index`-th example) from the tfrecord."""
    for i, trajectory in enumerate(load_trajectories(record_path, cameras)):
        if i == index:
            return trajectory
    raise IndexError(f"tfrecord {record_path} has no example at index {index}")
