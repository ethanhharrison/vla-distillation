"""Small image utilities shared by the dense_description generators."""

from __future__ import annotations

import io

from PIL import Image

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
