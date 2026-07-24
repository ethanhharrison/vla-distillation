"""Image + hashing helpers for the subgoal-image stage.

Dependency-light (Pillow only), so the stage runs by itself:

- content hashing : `sha256_hex`, `request_key` (the money-safety cache key)
- perceptual hash : `dhash` / `hamming` / `phash_delta` — the cheap verify-B
                    signal for "how much did the subgoal actually change vs source"
- pixel helpers   : `image_size`, `downscale` (to the 224 policy resolution)
"""

from __future__ import annotations

import hashlib
import io

POLICY_RES = 224  # student policy input resolution


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def request_key(*parts: object) -> str:
    """Deterministic key over the inputs of a (potentially paid) edit call.

    Identical inputs -> identical key -> cache hit -> the paid call is skipped,
    so a rerun of the same config costs $0.
    """
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, bytes):
            h.update(b"\x00b:")
            h.update(hashlib.sha256(p).digest())
        else:
            h.update(b"\x00s:")
            h.update(str(p).encode("utf-8"))
    return h.hexdigest()


# --- perceptual hash (difference hash), dependency-free --------------------- #

def dhash(image_bytes: bytes, hash_size: int = 8) -> int:
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    px = list(img.getdata())
    w = hash_size + 1
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            bits = (bits << 1) | (1 if px[row * w + col] > px[row * w + col + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def phash_delta(src_bytes: bytes, dst_bytes: bytes, hash_size: int = 8) -> dict:
    """dHash delta: {"bits", "norm" in [0,1], "hash_size"}.

    norm ~0 = "no visible change" (edit too weak); norm large = "different
    scene" (edit too strong). Verify-B wants a sane mid-band.
    """
    total = hash_size * hash_size
    bits = hamming(dhash(src_bytes, hash_size), dhash(dst_bytes, hash_size))
    return {"bits": bits, "norm": round(bits / total, 4), "hash_size": hash_size}


# --- pixel helpers ---------------------------------------------------------- #

def image_size(image_bytes: bytes) -> list[int]:
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as im:
        return [im.width, im.height]


def downscale(image_bytes: bytes, size: int = POLICY_RES, fmt: str = "JPEG") -> bytes:
    """Resize to a square `size`x`size` (policy resolution) for the contact sheet."""
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as im:
        im = im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format=fmt, quality=90)
        return buf.getvalue()


def is_png(image_bytes: bytes) -> bool:
    return image_bytes[:8].startswith(b"\x89PNG")
