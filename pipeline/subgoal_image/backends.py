"""Stage B image-edit backends behind a registry (mirrors `register_vlm`).

Every backend turns one source camera frame into a **subgoal** frame. Two
"free" backends make the layer fully runnable offline, two paid backends do real
instruction-conditioned editing:

- `dummy_image`  : offline placeholder edit (visible tint + label). $0.
- `real_future`  : no API — the subgoal is the REAL frame at t+k from the same
                   trajectory (provided by the runner). $0. Tag=real_future.
- `gemini_image` : Google Gemini image edit (generate_content, IMAGE modality).
- `openai_image` : OpenAI images.edit (gpt-image-1.5 / mini).

All backends return identically-shaped `SubgoalResult`s so `edited` and
`real_future` samples are interchangeable downstream. Model names / call shapes
verified July 2026 against installed SDKs (openai 2.47.0, google-genai 2.14.0).

Caching and the $ ceiling are enforced by the runner (run.py) around `edit()`;
backends themselves are stateless and make exactly one API call per `edit()`.
"""

from __future__ import annotations

import base64
import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Type


# --------------------------------------------------------------------------- #
# price tables (per-image USD estimates; hosted image models bill by token so
# these are conservative estimates used for the ceiling and per-sample cost).
# --------------------------------------------------------------------------- #

GEMINI_PRICES = {
    "gemini-2.5-flash-image": 0.04,
    "gemini-3.1-flash-image": 0.06,
    "gemini-3.1-flash-lite-image": 0.03,
    "gemini-3-pro-image": 0.14,
}
# openai: per (model, quality) at ~1024px, padded for edit input-image tokens.
OPENAI_PRICES = {
    "gpt-image-1.5": {"low": 0.04, "medium": 0.06, "high": 0.15, "auto": 0.06},
    "gpt-image-1": {"low": 0.04, "medium": 0.06, "high": 0.18, "auto": 0.06},
    "gpt-image-1-mini": {"low": 0.02, "medium": 0.03, "high": 0.05, "auto": 0.03},
}
_FALLBACK_PRICE = 0.20  # conservative if a model is unknown


@dataclass
class SubgoalResult:
    image_bytes: bytes | None
    ext: str                      # "png" | "jpg"
    model: str
    kind: str                     # "edited" | "real_future" | "dummy"
    cost_usd_est: float
    error: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class SubgoalRequest:
    source_bytes: bytes
    camera: str
    instruction: str
    prompt: str                   # rendered edit prompt (paid backends use this)
    future_bytes: bytes | None = None   # provided by runner for real_future
    k: int | None = None


class ImageEditBackend(ABC):
    name: str = "base"
    is_paid: bool = False

    def __init__(self, model: str | None = None, **kwargs):
        self.model = model or self.default_model
        self.opts = kwargs

    default_model: str = ""

    def estimate_cost(self) -> float:
        return 0.0

    @abstractmethod
    def edit(self, req: SubgoalRequest) -> SubgoalResult:
        raise NotImplementedError


_REGISTRY: dict[str, Type[ImageEditBackend]] = {}


def register_image_backend(name: str) -> Callable[[Type[ImageEditBackend]], Type[ImageEditBackend]]:
    def deco(cls: Type[ImageEditBackend]) -> Type[ImageEditBackend]:
        cls.name = name
        _REGISTRY[name.lower()] = cls
        return cls

    return deco


def available_image_backends() -> list[str]:
    return sorted(_REGISTRY)


def build_image_backend(name: str, model: str | None = None, **kwargs) -> ImageEditBackend:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown image backend {name!r}. Available: {', '.join(available_image_backends())}"
        )
    return _REGISTRY[key](model=model, **kwargs)


def is_paid_backend(name: str) -> bool:
    key = name.lower()
    return key in _REGISTRY and _REGISTRY[key].is_paid


# --------------------------------------------------------------------------- #
# free backends
# --------------------------------------------------------------------------- #

@register_image_backend("real_future")
class RealFutureBackend(ImageEditBackend):
    """Subgoal = the real frame at t+k (supplied by the runner). No API call."""

    is_paid = False
    default_model = "real_future"

    def edit(self, req: SubgoalRequest) -> SubgoalResult:
        if req.future_bytes is None:
            return SubgoalResult(
                image_bytes=None, ext="jpg", model=self.model, kind="real_future",
                cost_usd_est=0.0, error="no future frame available (t+k out of range?)",
            )
        return SubgoalResult(
            image_bytes=req.future_bytes, ext="jpg", model=self.model,
            kind="real_future", cost_usd_est=0.0, meta={"k": req.k},
        )


@register_image_backend("dummy_image")
class DummyImageBackend(ImageEditBackend):
    """Offline placeholder 'edit': tint the source + stamp a label. No API call."""

    is_paid = False
    default_model = "dummy"

    def edit(self, req: SubgoalRequest) -> SubgoalResult:
        from PIL import Image, ImageDraw

        im = Image.open(io.BytesIO(req.source_bytes)).convert("RGB")
        # deterministic, visible change so phash delta is non-zero and the
        # contact sheet obviously shows a "subgoal": green wash + corner label.
        overlay = Image.new("RGB", im.size, (0, 90, 0))
        im = Image.blend(im, overlay, 0.18)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
        d.text((2, 3), f"DUMMY {req.camera}", fill=(0, 255, 0))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return SubgoalResult(
            image_bytes=buf.getvalue(), ext="jpg", model=self.model,
            kind="dummy", cost_usd_est=0.0,
        )


# --------------------------------------------------------------------------- #
# paid backends
# --------------------------------------------------------------------------- #

@register_image_backend("gemini_image")
class GeminiImageBackend(ImageEditBackend):
    """Gemini image edit via google-genai `generate_content` (IMAGE modality)."""

    is_paid = True
    default_model = "gemini-2.5-flash-image"

    def __init__(self, model: str | None = None, api_key: str | None = None, **kwargs):
        super().__init__(model=model, **kwargs)
        self._api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def estimate_cost(self) -> float:
        return GEMINI_PRICES.get(self.model, _FALLBACK_PRICE)

    def edit(self, req: SubgoalRequest) -> SubgoalResult:
        from google.genai import types

        client = self._get_client()
        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=[
                    req.prompt,
                    types.Part.from_bytes(data=req.source_bytes, mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )
        except Exception as e:  # network / API / quota errors surfaced honestly
            return SubgoalResult(None, "png", self.model, "edited",
                                 self.estimate_cost(), error=f"{type(e).__name__}: {e}")

        candidates = getattr(resp, "candidates", None) or []
        if not candidates:
            return SubgoalResult(None, "png", self.model, "edited",
                                 self.estimate_cost(), error="no candidates (refusal?)")
        finish = getattr(candidates[0], "finish_reason", None)
        parts = getattr(candidates[0].content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None and inline.data:
                ext = "png"
                if inline.mime_type and "/" in inline.mime_type:
                    ext = inline.mime_type.split("/", 1)[1].replace("jpeg", "jpg")
                return SubgoalResult(
                    image_bytes=inline.data, ext=ext, model=self.model, kind="edited",
                    cost_usd_est=self.estimate_cost(),
                    meta={"finish_reason": str(finish)},
                )
        # text-only response (often a refusal explanation)
        txt = " ".join(getattr(p, "text", "") or "" for p in parts).strip()
        return SubgoalResult(None, "png", self.model, "edited", self.estimate_cost(),
                             error=f"no image part (finish={finish}); text={txt[:200]!r}")


@register_image_backend("openai_image")
class OpenAIImageBackend(ImageEditBackend):
    """OpenAI image edit via `client.images.edit` (gpt-image-1.5 / mini)."""

    is_paid = True
    default_model = "gpt-image-1.5"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        quality: str = "low",
        size: str = "auto",
        input_fidelity: str = "high",
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.quality = quality
        self.size = size
        self.input_fidelity = input_fidelity
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def estimate_cost(self) -> float:
        table = OPENAI_PRICES.get(self.model)
        if table is None:
            return _FALLBACK_PRICE
        return table.get(self.quality, table.get("auto", _FALLBACK_PRICE))

    def edit(self, req: SubgoalRequest) -> SubgoalResult:
        client = self._get_client()
        kwargs = dict(
            model=self.model,
            image=("source.jpg", req.source_bytes, "image/jpeg"),
            prompt=req.prompt,
            size=self.size,
            quality=self.quality,
            output_format="png",
            n=1,
        )
        # input_fidelity is unsupported on the -mini model.
        if "mini" not in self.model:
            kwargs["input_fidelity"] = self.input_fidelity
        try:
            resp = client.images.edit(**kwargs)
        except Exception as e:
            return SubgoalResult(None, "png", self.model, "edited",
                                 self.estimate_cost(), error=f"{type(e).__name__}: {e}")

        data = getattr(resp, "data", None) or []
        if not data or not getattr(data[0], "b64_json", None):
            return SubgoalResult(None, "png", self.model, "edited",
                                 self.estimate_cost(), error="no image data returned")
        img = base64.b64decode(data[0].b64_json)
        return SubgoalResult(
            image_bytes=img, ext="png", model=self.model, kind="edited",
            cost_usd_est=self.estimate_cost(),
            meta={"quality": self.quality, "size": self.size,
                  "input_fidelity": self.input_fidelity if "mini" not in self.model else None},
        )
