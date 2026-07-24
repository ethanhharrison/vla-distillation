"""Content-addressed edit cache — the money-safety layer for Stage B.

Before any paid edit, the runner computes a `request_key` (imaging.request_key)
over the inputs that determine the output bytes (source frame + prompt + backend
+ model + params) and looks it up here. A hit returns the previously produced
image and its recorded cost, so **a rerun of the same config costs $0**. Because
the index is keyed by inputs, the cache works across runs and across output dirs.

    <root>/blobs/<sha[:2]>/<sha>.<ext>     # the produced images
    <root>/index/<key[:2]>/<key>.json      # request_key -> {blob, kind, cost, ...}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .imaging import sha256_hex


@dataclass
class BlobCache:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.blobs = self.root / "blobs"
        self.index = self.root / "index"

    def put_blob(self, data: bytes, ext: str) -> str:
        sha = sha256_hex(data)
        rel = f"{sha[:2]}/{sha}.{ext.lstrip('.')}"
        path = self.blobs / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)  # atomic; safe against a killed rerun
        return rel

    def get_blob(self, rel: str) -> bytes:
        return (self.blobs / rel).read_bytes()

    def lookup(self, key: str) -> dict | None:
        p = self.index / key[:2] / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def store(self, key: str, payload: dict) -> None:
        p = self.index / key[:2] / f"{key}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(p)
