"""Small image utilities shared by the dense_description generators."""

from __future__ import annotations

import io

from PIL import Image, ImageEnhance

# The wrist camera is mounted such that its raw frames read upside down and
# mirrored relative to the shoulder/exterior views - rotating it 180 degrees
# (top<->bottom, left<->right) makes its framing agree with the other
# cameras, which should make cross-view motion easier for a VLM to track.
DEFAULT_ROTATE_180_CAMERAS: frozenset[str] = frozenset({"wrist_image"})


def rotate_180(jpeg: bytes, quality: int = 95) -> bytes:
    """Rotate a JPEG 180 degrees (top<->bottom, left<->right)."""
    with Image.open(io.BytesIO(jpeg)) as im:
        rotated = im.transpose(Image.ROTATE_180)
        buf = io.BytesIO()
        rotated.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def maybe_rotate(
    jpeg: bytes,
    camera: str,
    rotate_cameras: frozenset[str] = DEFAULT_ROTATE_180_CAMERAS,
) -> bytes:
    """Rotate `jpeg` 180 degrees iff `camera` is in `rotate_cameras`."""
    return rotate_180(jpeg) if camera in rotate_cameras else jpeg


def increase_contrast(jpeg: bytes, factor: float, quality: int = 95) -> bytes:
    """Scale contrast by `factor` (1.0 = unchanged, >1.0 = more contrast)."""
    with Image.open(io.BytesIO(jpeg)) as im:
        enhanced = ImageEnhance.Contrast(im.convert("RGB")).enhance(factor)
        buf = io.BytesIO()
        enhanced.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def process_image(
    jpeg: bytes,
    camera: str,
    rotate_cameras: frozenset[str] = DEFAULT_ROTATE_180_CAMERAS,
    contrast_factor: float | None = None,
) -> bytes:
    """Rotate (if applicable) then optionally boost contrast - the one place
    that applies every per-image transform a generator might use, so the
    generator and its visualizer always agree on what the model actually saw."""
    out = maybe_rotate(jpeg, camera, rotate_cameras)
    if contrast_factor is not None and contrast_factor != 1.0:
        out = increase_contrast(out, contrast_factor)
    return out
