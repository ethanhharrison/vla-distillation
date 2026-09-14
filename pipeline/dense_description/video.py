"""Render a trajectory clip as a single stacked-view video.

Vertically stacks any number of cameras (e.g. both shoulder/exterior views and
the wrist view) frame-by-frame across a clip and encodes the result as an
mp4, so a video-capable VLM sees every intermediate frame instead of just the
clip's start and end stills.
"""

from __future__ import annotations

import io
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

from pipeline.language_instruction.trajectory import Trajectory

from .imaging import DEFAULT_ROTATE_180_CAMERAS, maybe_rotate


def _decode(jpeg: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg)) as im:
        return np.array(im.convert("RGB"))


def _stack_frame(frames: list[np.ndarray]) -> np.ndarray:
    """Vertically stack frames top-to-bottom, resizing to the first one's width."""
    width = frames[0].shape[1]
    resized = [frames[0]]
    for frame in frames[1:]:
        if frame.shape[1] != width:
            with Image.fromarray(frame) as im:
                new_height = round(im.height * width / im.width)
                frame = np.array(im.resize((width, new_height)))
        resized.append(frame)
    return np.vstack(resized)


def render_clip_video(
    trajectory: Trajectory,
    start_step: int,
    end_step: int,
    cameras: tuple[str, ...],
    fps: float,
    out_path: Path,
    rotate_180: frozenset[str] = DEFAULT_ROTATE_180_CAMERAS,
) -> bytes:
    """Render `trajectory[start_step:end_step]` on `cameras` as a stacked mp4.

    `cameras` are stacked top-to-bottom in the given order. Cameras named in
    `rotate_180` are rotated 180 degrees before stacking (see `imaging.py` -
    the wrist view is rotated by default so its framing agrees with the
    others). Writes the mp4 to `out_path` and returns its bytes - callers
    decide whether to keep or delete the file afterward.
    """
    per_camera_frames = [
        [
            _decode(maybe_rotate(jpeg, cam, rotate_180))
            for jpeg in trajectory.images[cam][start_step : end_step + 1]
        ]
        for cam in cameras
    ]
    frames = [_stack_frame(list(step_frames)) for step_frames in zip(*per_camera_frames)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(frames), fps=fps, codec="libx264")
    return out_path.read_bytes()
