"""Render a trajectory clip as a single stacked-view video.

Vertically stacks two cameras (e.g. an exterior/shoulder view and the wrist
view) frame-by-frame across a clip and encodes the result as an mp4, so a
video-capable VLM sees every intermediate frame instead of just the clip's
start and end stills.
"""

from __future__ import annotations

import io
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

from pipeline.language_instruction.trajectory import Trajectory


def _decode(jpeg: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg)) as im:
        return np.array(im.convert("RGB"))


def _stack_frame(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Vertically stack two frames, resizing `bottom` to `top`'s width if needed."""
    if bottom.shape[1] != top.shape[1]:
        with Image.fromarray(bottom) as im:
            new_height = round(im.height * top.shape[1] / im.width)
            bottom = np.array(im.resize((top.shape[1], new_height)))
    return np.vstack([top, bottom])


def render_clip_video(
    trajectory: Trajectory,
    start_step: int,
    end_step: int,
    cameras: tuple[str, str],
    fps: float,
    out_path: Path,
) -> bytes:
    """Render `trajectory[start_step:end_step]` on `cameras` as a stacked mp4.

    `cameras` is (top, bottom). Writes the mp4 to `out_path` and returns its
    bytes — callers decide whether to keep or delete the file afterward.
    """
    top_cam, bottom_cam = cameras
    top_frames = trajectory.images[top_cam][start_step : end_step + 1]
    bottom_frames = trajectory.images[bottom_cam][start_step : end_step + 1]
    frames = [
        _stack_frame(_decode(top), _decode(bottom))
        for top, bottom in zip(top_frames, bottom_frames)
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(frames), fps=fps, codec="libx264")
    return out_path.read_bytes()
