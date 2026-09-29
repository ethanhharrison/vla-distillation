"""Load DROID tfrecords and expose per-step camera frames."""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import tensorflow as tf

DEFAULT_CAMERAS = ("shoulder_image_1", "shoulder_image_2", "wrist_image")

# Canonical camera name -> the public droid_100 RLDS build's observation key,
# so an RLDS episode decodes into the exact same Trajectory shape (and the
# same camera names) as the partner tfrecord format - every other module
# (rotation defaults, prompt building, video stacking) needs no changes to
# work with either source.
RLDS_CAMERA_MAP = {
    "shoulder_image_1": "exterior_image_1_left",
    "shoulder_image_2": "exterior_image_2_left",
    "wrist_image": "wrist_image_left",
}
RLDS_INSTRUCTION_KEYS = ("language_instruction", "language_instruction_2", "language_instruction_3")

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

def _load_tfrecord_trajectories(
    record_path: str | Path,
    cameras: tuple[str, ...],
) -> Iterator[Trajectory]:
    """Yield one `Trajectory` per example in a partner-format .tfrecord."""
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


def _encode_jpeg(arr, quality: int = 95) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _load_rlds_trajectories(
    dataset_dir: str | Path,
    cameras: tuple[str, ...],
) -> Iterator[Trajectory]:
    """Yield one `Trajectory` per episode of a TFDS-built RLDS dataset (e.g.
    the public droid_100 build) - same shape as the tfrecord loader above, so
    downstream code doesn't care which source a clip came from."""
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(dataset_dir))
    ds = builder.as_dataset(split="train")
    for episode in ds:
        steps = list(episode["steps"])
        images: dict[str, list[bytes]] = {cam: [] for cam in cameras}
        for step in steps:
            obs = step["observation"]
            for cam in cameras:
                rlds_key = RLDS_CAMERA_MAP.get(cam, cam)
                images[cam].append(_encode_jpeg(obs[rlds_key].numpy()))

        instructions: list[str] = []
        for key in RLDS_INSTRUCTION_KEYS:
            if steps and key in steps[0]:
                text = steps[0][key].numpy().decode("utf-8", "replace").strip()
                if text:
                    instructions.append(text)

        md = episode.get("episode_metadata", {})
        file_path = md["file_path"].numpy().decode("utf-8", "replace") if "file_path" in md else ""
        metadata = {"rel_path": file_path}
        metadata["episode_id"] = Path(file_path).parent.name if file_path else str(dataset_dir)
        for i, text in enumerate(instructions[:3], start=1):
            metadata[f"language_instruction{i}"] = text

        length = min((len(v) for v in images.values()), default=0)
        yield Trajectory(
            record_path=str(dataset_dir),
            length=length,
            images=images,
            metadata=metadata,
        )


def load_trajectories(
    record_path: str | Path,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
) -> Iterator[Trajectory]:
    """Yield one `Trajectory` per episode.

    Dispatches on whether `record_path` is a file (a partner-format
    .tfrecord) or a directory (a TFDS-built RLDS dataset, e.g. droid_100) -
    every caller can pass either interchangeably.
    """
    if Path(record_path).is_dir():
        yield from _load_rlds_trajectories(record_path, cameras)
    else:
        yield from _load_tfrecord_trajectories(record_path, cameras)

def load_trajectory(
    record_path: str | Path,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    index: int = 0,
) -> Trajectory:
    """Load a single episode (the `index`-th example) from `record_path`."""
    for i, trajectory in enumerate(load_trajectories(record_path, cameras)):
        if i == index:
            return trajectory
    raise IndexError(f"{record_path} has no example at index {index}")
