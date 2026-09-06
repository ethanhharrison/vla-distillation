"""Multi-view canvas: stitch three camera views into one image, edit once, split back.

One implementation, three consumers — `multiview_probe.py` (the go/no-go probe),
`run_experiment.py` (the A/B/C protocol) and `resample_null.py` (the control).
Three copies of this would be three chances for the geometry, the prompt suffix
or the cache key to drift apart, and any of those silently breaks comparability
between a run and its own null.

The `cosmos` geometry is imported from `explorations/cosmos3/run_experiment.py`
rather than copied, the same way `explorations/wan/run_views.py` does it, so every
harness in the repo stitches views identically. `grid2x2` is defined here because
it lives in `wan/run_views.py`, which pulls in diffusers via `smoke_i2v` and is
therefore not importable in the main venv; the numbers match wan's exactly.

**Cache keys here are load-bearing and must not be "improved".** The A and N
canvas edits for all 8 situations were already paid for by `multiview_probe.py`;
they only stay free if the key is byte-identical, which means the camera label
stays `canvas:<layout>` and the parameter order stays as it is.
"""

from __future__ import annotations

import importlib.util
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image

from metrics import diff

from pipeline.subgoal_image.backends import ImageEditBackend
from pipeline.subgoal_image.cache import BlobCache
from pipeline.subgoal_image.cost import BudgetExceeded, CostTracker
from pipeline.subgoal_image.imaging import request_key, sha256_hex

HERE = Path(__file__).resolve().parent
CAMERAS = ("exterior_1", "exterior_2", "wrist")


class Aborted(RuntimeError):
    """The ceiling refused a call; the caller must stop."""


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def _load_cosmos_canvas():
    path = HERE.parent / "cosmos3" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("cosmos3_run_experiment", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cosmos = _load_cosmos_canvas()

GRID_W, GRID_H = 640, 360
GRID_CELLS = {"exterior_1": (0, 0), "exterior_2": (320, 0), "wrist": (0, 180)}


def build_grid2x2(frames: dict[str, bytes]) -> Image.Image:
    """2x2 of native 320x180 cells; the fourth cell is left BLACK on purpose —
    a duplicated view would let the model satisfy the frame by copying, which is
    the failure this is built to detect."""
    canvas = Image.new("RGB", (GRID_W, GRID_H))
    for cam, xy in GRID_CELLS.items():
        canvas.paste(Image.open(io.BytesIO(frames[cam])).convert("RGB").resize((320, 180)), xy)
    return canvas


def split_grid2x2(frame: np.ndarray) -> dict[str, np.ndarray]:
    h, w = frame.shape[:2]
    hh, hw = h // 2, w // 2
    return {"exterior_1": frame[:hh, :hw], "exterior_2": frame[:hh, hw:],
            "wrist": frame[hh:, :hw]}


def build_cosmos(frames: dict[str, bytes]) -> Image.Image:
    """The layout Cosmos3-Nano-Policy-DROID was post-trained on: wrist 640x360
    above two 320x180 exteriors. Adapts the imported path-based builder to bytes."""
    canvas = Image.new("RGB", (_cosmos.CANVAS_W, _cosmos.CANVAS_H))
    half_h = _cosmos.CANVAS_H - _cosmos.WRIST_H

    def _im(cam):
        return Image.open(io.BytesIO(frames[cam])).convert("RGB")

    canvas.paste(_im("wrist").resize((_cosmos.CANVAS_W, _cosmos.WRIST_H)), (0, 0))
    canvas.paste(_im("exterior_1").resize((_cosmos.CANVAS_W // 2, half_h)), (0, _cosmos.WRIST_H))
    canvas.paste(_im("exterior_2").resize((_cosmos.CANVAS_W // 2, half_h)),
                 (_cosmos.CANVAS_W // 2, _cosmos.WRIST_H))
    return canvas


#: name -> (build, split, "WxH").
#:
#: The aspect was expected to matter, since `gpt-image-2` documents only
#: 1024x1024 / 1536x1024 / 1024x1536 as output sizes. Measured 2026-08-26 it does
#: not: with `size="auto"` both providers returned both canvases at the input
#: aspect to within 0.003. Kept as two layouts because panel count and cell area
#: still differ.
LAYOUTS = {
    "grid2x2": (build_grid2x2, split_grid2x2, f"{GRID_W}x{GRID_H}"),
    "cosmos": (build_cosmos, _cosmos.split_canvas, f"{_cosmos.CANVAS_W}x{_cosmos.CANVAS_H}"),
}

CANVAS_PROMPT_SUFFIX = (
    " IMPORTANT: this image is a composite of {n} separate camera views of the "
    "SAME scene at the SAME moment, arranged in a fixed grid. Keep the grid "
    "layout exactly as it is — same number of panels, same positions, same "
    "boundaries. Edit every panel so that all of them show the SAME single "
    "physical change, each from its own viewpoint. Do not merge, reorder, crop "
    "or re-frame the panels, and do not turn this into one single image."
)


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def build(layout: str, frames: dict[str, bytes]) -> tuple[bytes, Image.Image, dict]:
    """(canvas png bytes, canvas image, source panels) for one layout."""
    builder, splitter, _ = LAYOUTS[layout]
    im = builder(frames)
    return png_bytes(im), im, splitter(np.asarray(im))


def split(layout: str, arr: np.ndarray) -> dict[str, np.ndarray]:
    return LAYOUTS[layout][1](arr)


def canvas_prompt(base_prompt: str, *, noop_prompt: str | None = None,
                  n: int = len(CAMERAS)) -> str:
    """The prompt actually sent. A no-op prompt is passed through VERBATIM: adding
    "keep the grid" language would help the model hold the layout and so
    understate the very re-render tax the no-op condition exists to measure."""
    return noop_prompt if noop_prompt is not None else base_prompt + CANVAS_PROMPT_SUFFIX.format(n=n)


def layout_check(out_panels: dict[str, np.ndarray],
                 src_panels: dict[str, np.ndarray]) -> dict:
    """Did the panels come back in their original positions?

    For each output panel, find which SOURCE panel it most resembles. If the grid
    survived, every output panel matches its own source (the diagonal). Off the
    diagonal means the model reflowed the composite — which a single vs-source
    number cannot distinguish from a large edit.
    """
    cams = list(src_panels)
    matched, detail = 0, {}
    for cam in cams:
        d = {c: round(diff(out_panels[cam], src_panels[c]), 2) for c in cams}
        best = min(d, key=d.get)
        matched += best == cam
        detail[cam] = {"best_match": best, "diffs": d}
    return {"panels_matched": matched, "n_panels": len(cams),
            "preserved": matched == len(cams), "detail": detail}


# --------------------------------------------------------------------------- #
# the one paid call
#
# This does NOT go through `ImageEditBackend.edit()`, and the reason is not
# stylistic: `GeminiImageBackend.edit` hardcodes `mime_type="image/jpeg"` because
# the pipeline feeds it JPEG frames, while a composite is sent as PNG (lossless,
# so no JPEG ringing across the panel boundaries the split relies on). Declaring
# jpeg for png bytes would also make condition A — already cached, sent as png —
# inconsistent with a freshly-called B or C, which is exactly the |B - A|
# difference the sensitivity number is built on.
# --------------------------------------------------------------------------- #


def single_edit(backend_name: str, be, prompt: str, images: list[bytes]):
    """One stateless edit call with one or more PNG input images.

    Returns (image_bytes, ext, actual_cost_usd, meta). The multi-image form is
    what makes cross-view reference conditioning possible on OpenAI without a
    conversation: `images.edit`'s `image` parameter accepts a sequence.
    """
    client = be._get_client()
    if backend_name == "gemini_image":
        from google.genai import types

        resp = client.models.generate_content(
            model=be.model,
            contents=[prompt, *[types.Part.from_bytes(data=d, mime_type="image/png")
                                for d in images]],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        cands = getattr(resp, "candidates", None) or []
        if not cands:
            raise RuntimeError("no candidates (refusal?)")
        parts = getattr(cands[0].content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None and inline.data:
                usage, cost = be._usage(resp)
                return inline.data, "png", cost, {"usage": usage,
                                                  "n_input_images": len(images)}
        finish = getattr(cands[0], "finish_reason", None)
        txt = " ".join(getattr(pt, "text", "") or "" for pt in parts).strip()
        raise RuntimeError(f"no image part (finish={finish}); text={txt[:160]!r}")

    import base64

    kwargs = {
        "model": be.model, "prompt": prompt, "size": be.size, "quality": be.quality,
        "output_format": "png", "n": 1,
        "image": [(f"in{i}.png", d, "image/png") for i, d in enumerate(images)],
    }
    if be.sends_input_fidelity:
        kwargs["input_fidelity"] = be.input_fidelity
    resp = client.images.edit(**kwargs)
    data = getattr(resp, "data", None) or []
    if not data or not getattr(data[0], "b64_json", None):
        raise RuntimeError("no image data returned")
    usage, cost = be._usage(resp)
    return (base64.b64decode(data[0].b64_json), "png", cost,
            {"usage": usage, "n_input_images": len(images)})

def canvas_cache_key(backend_name: str, backend: ImageEditBackend, *, layout: str,
                     prompt: str, canvas_bytes: bytes) -> str:
    """Key for one canvas edit. See the module docstring: do not reorder."""
    return request_key(backend_name, backend.model, getattr(backend, "quality", ""),
                       getattr(backend, "size", ""), getattr(backend, "input_fidelity", ""),
                       f"canvas:{layout}", prompt, sha256_hex(canvas_bytes))


def edit_canvas(
    *, backend_name: str, backend: ImageEditBackend, layout: str, prompt: str,
    canvas_bytes: bytes, cache: BlobCache | None = None,
    tracker: CostTracker | None = None, use_cache: bool = True,
    example_id: str | None = None,
) -> dict:
    """One cached, budget-checked canvas edit.

    Mirrors `pipeline.subgoal_image.edit.edit_camera`'s ordering exactly —
    key -> lookup -> precheck -> call -> record -> store — which is the whole
    money-safety guarantee. `edit_camera` itself cannot be reused because its
    backend call is one frame in, one frame out.

    Returns {image, ext, cached, cost_usd, meta, error}.
    """
    key = canvas_cache_key(backend_name, backend, layout=layout, prompt=prompt,
                           canvas_bytes=canvas_bytes)
    label = f"canvas:{layout}"

    if cache is not None and use_cache:
        hit = cache.lookup(key)
        if hit is not None:
            if tracker is not None:
                tracker.record(backend=backend_name, model=backend.model, cost_usd=0.0,
                               cached=True, example_id=example_id, camera=label,
                               note="cache hit")
            return {"image": cache.get_blob(hit["blob"]), "ext": "png", "cached": True,
                    "cost_usd": 0.0, "meta": hit.get("meta", {}), "error": None}

    if backend.is_paid and tracker is not None:
        try:
            tracker.precheck(backend.estimate_cost(), what=label)
        except BudgetExceeded as exc:
            raise Aborted(str(exc)) from exc

    t0 = time.time()
    try:
        img, ext, cost, meta = single_edit(backend_name, backend, prompt, [canvas_bytes])
    except Exception as exc:  # noqa: BLE001 — one bad sample, the run goes on
        if tracker is not None:
            tracker.record(backend=backend_name, model=backend.model, cost_usd=0.0,
                           cached=False, example_id=example_id, camera=label,
                           note=f"ERROR: {type(exc).__name__}: {exc}")
        return {"image": None, "ext": "png", "cached": False, "cost_usd": 0.0,
                "meta": {}, "error": f"{type(exc).__name__}: {exc}"}

    meta = dict(meta or {})
    meta["latency_s"] = round(time.time() - t0, 1)
    if tracker is not None:
        tracker.record(backend=backend_name, model=backend.model, cost_usd=cost,
                       cached=False, example_id=example_id, camera=label)
    if cache is not None and use_cache:
        cache.store(key, {"blob": cache.put_blob(img, ext), "kind": "edited",
                          "meta": meta, "cost_usd_est": cost})
    return {"image": img, "ext": ext, "cached": False, "cost_usd": cost,
            "meta": meta, "error": None}
