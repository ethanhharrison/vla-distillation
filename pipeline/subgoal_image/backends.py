"""Stage B image-edit backends behind a registry (mirrors `register_vlm`).

Every backend turns one source camera frame into a **subgoal** frame. Two
"free" backends make the layer fully runnable offline, two paid backends do real
instruction-conditioned editing:

- `dummy_image`  : offline placeholder edit (visible tint + label). $0.
- `real_future`  : no API — the subgoal is the REAL frame at t+k from the same
                   trajectory (provided by the runner). $0. Tag=real_future.
- `gemini_image` : Google Gemini image edit (generate_content, IMAGE modality).
- `openai_image` : OpenAI images.edit (gpt-image-2 / 1.5 / 1 / mini).

All backends return identically-shaped `SubgoalResult`s so `edited` and
`real_future` samples are interchangeable downstream. Model names / call shapes
verified against the installed SDKs (openai 2.47.0, google-genai 2.14.0); the
model lists and prices below were re-verified 2026-08-20 against the live
`models.list()` of both providers and one real call per configuration.

Two different cost numbers, deliberately:

- `estimate_cost()` is a **static per-image upper bound**, known before the call,
  and is what the ceiling checks. It must never under-estimate.
- `SubgoalResult.cost_usd_est` is the **actual** cost derived from the token
  usage the API reports, when it reports any, and falls back to the static
  estimate otherwise. This is what lands in the ledger.

Caching and the $ ceiling are enforced by `edit.edit_camera` around `edit()`;
backends themselves are stateless and make exactly one API call per `edit()`.
"""

from __future__ import annotations

import base64
import io
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------- #
# price tables
#
# Hosted image models bill by token, not by image, so a per-image price depends
# on the resolution the model chooses to return. Everything marked *measured*
# below was billed on a real call from a 320x180 DROID frame — the size this
# pipeline actually sends — and rounded UP: these numbers gate the ceiling, so
# an under-estimate silently removes the protection.
#
# `auto` quality is priced as `high`, because that is what it may pick.
# --------------------------------------------------------------------------- #

GEMINI_PRICES = {
    "gemini-3.1-flash-image": 0.09,       # measured $0.0833 (1389 out tok @ $60/Mtok)
    "gemini-3.1-flash-lite-image": 0.05,  # list $0.0336 per 1K image @ $30/Mtok
    "gemini-3-pro-image": 0.18,           # list $0.134 per 1K-2K image @ $120/Mtok
    "gemini-2.5-flash-image": 0.04,       # list $0.039 per image
}
# openai: per (model, quality). gpt-image-2 measured; the rest are list prices.
OPENAI_PRICES = {
    "gpt-image-2": {"low": 0.01, "medium": 0.05, "high": 0.16, "auto": 0.16},
    "gpt-image-1.5": {"low": 0.04, "medium": 0.06, "high": 0.15, "auto": 0.15},
    "gpt-image-1": {"low": 0.04, "medium": 0.06, "high": 0.18, "auto": 0.18},
    "gpt-image-1-mini": {"low": 0.02, "medium": 0.03, "high": 0.05, "auto": 0.05},
}
_FALLBACK_PRICE = 0.20  # conservative if a model is unknown

# Token rates (USD per 1M tokens), used to turn reported usage into the ACTUAL
# cost of a call. Source: each provider's published pricing page, 2026-08-20.
#: Gemini bills generated images as output tokens. Prompt tokens are billed at
#: the model's text rate and come to <1% of a call at our frame sizes (298 vs
#: 1389 tokens on the reference call), so only the image output is counted here
#: — an under-count of well under a cent, and never the ceiling's basis.
GEMINI_IMAGE_OUTPUT_USD_PER_MTOK = {
    "gemini-3.1-flash-image": 60.0,
    "gemini-3.1-flash-lite-image": 30.0,
    "gemini-3-pro-image": 120.0,
    "gemini-2.5-flash-image": 30.0,
}
#: OpenAI reports input text/image and output image tokens separately, so the
#: actual cost of an edit is exact rather than estimated.
OPENAI_USD_PER_MTOK = {
    "gpt-image-2": {"text_in": 5.0, "image_in": 8.0, "image_out": 30.0},
    "gpt-image-1.5": {"text_in": 5.0, "image_in": 8.0, "image_out": 32.0},
    "gpt-image-1": {"text_in": 5.0, "image_in": 10.0, "image_out": 40.0},
    "gpt-image-1-mini": {"text_in": 2.0, "image_in": 2.5, "image_out": 8.0},
}

#: `input_fidelity` is not universal. gpt-image-2 rejects it with a 400
#: ("the model does not support the 'input_fidelity' parameter", verified
#: 2026-08-20) because it always processes inputs at high fidelity, and the mini
#: model never supported it. Send it only where it exists — a substring check on
#: the model name is not enough.
OPENAI_INPUT_FIDELITY_MODELS = frozenset({"gpt-image-1", "gpt-image-1.5"})


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


_REGISTRY: dict[str, type[ImageEditBackend]] = {}


def register_image_backend(name: str) -> Callable[[type[ImageEditBackend]], type[ImageEditBackend]]:
    def deco(cls: type[ImageEditBackend]) -> type[ImageEditBackend]:
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
    #: Newest image-edit model on the API (`models.list()`, 2026-08-20). The
    #: flash tier is the default rather than `gemini-3-pro-image` because Stage B
    #: pays per image at dataset scale; pass --gemini-model to go up a tier.
    default_model = "gemini-3.1-flash-image"

    def __init__(self, model: str | None = None, api_key: str | None = None, **kwargs):
        super().__init__(model=model, **kwargs)
        self._api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def estimate_cost(self) -> float:
        return GEMINI_PRICES.get(self.model, _FALLBACK_PRICE)

    def _usage(self, resp) -> tuple[dict, float]:
        """(usage dict, actual USD) from the response, falling back to the estimate."""
        um = getattr(resp, "usage_metadata", None)
        out_tokens = getattr(um, "candidates_token_count", None) if um else None
        usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", None) if um else None,
            "output_tokens": out_tokens,
            "total_tokens": getattr(um, "total_token_count", None) if um else None,
        }
        rate = GEMINI_IMAGE_OUTPUT_USD_PER_MTOK.get(self.model)
        if out_tokens is None or rate is None:
            return usage, self.estimate_cost()
        return usage, round(out_tokens * rate / 1e6, 6)

    def edit(self, req: SubgoalRequest) -> SubgoalResult:
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
        except Exception as e:  # network / API / quota errors surfaced honestly  # noqa: BLE001
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
                usage, cost = self._usage(resp)
                return SubgoalResult(
                    image_bytes=inline.data, ext=ext, model=self.model, kind="edited",
                    cost_usd_est=cost,
                    meta={"finish_reason": str(finish), "usage": usage},
                )
        # text-only response (often a refusal explanation)
        txt = " ".join(getattr(p, "text", "") or "" for p in parts).strip()
        return SubgoalResult(None, "png", self.model, "edited", self.estimate_cost(),
                             error=f"no image part (finish={finish}); text={txt[:200]!r}")


@register_image_backend("openai_image")
class OpenAIImageBackend(ImageEditBackend):
    """OpenAI image edit via `client.images.edit` (gpt-image-2 / 1.5 / 1 / mini)."""

    is_paid = True
    #: Newest image model on the API (`models.list()`, 2026-08-20). Measured on a
    #: 320x180 DROID frame it is both cheaper and better-priced per quality step
    #: than gpt-image-1.5: $0.006 low / $0.037 medium / $0.141 high, at 19 / 38 /
    #: 100 s per image — quality is the dominant cost AND latency knob here, so a
    #: whole-grid run at `high` is hours, not minutes.
    default_model = "gpt-image-2"

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
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def estimate_cost(self) -> float:
        table = OPENAI_PRICES.get(self.model)
        if table is None:
            return _FALLBACK_PRICE
        return table.get(self.quality, table.get("auto", _FALLBACK_PRICE))

    @property
    def sends_input_fidelity(self) -> bool:
        return self.model in OPENAI_INPUT_FIDELITY_MODELS

    def _usage(self, resp) -> tuple[dict, float]:
        """(usage dict, actual USD) from the response, falling back to the estimate."""
        u = getattr(resp, "usage", None)
        if u is None:
            return {}, self.estimate_cost()
        details = getattr(u, "input_tokens_details", None)
        text_in = getattr(details, "text_tokens", 0) or 0
        image_in = getattr(details, "image_tokens", 0) or 0
        image_out = getattr(u, "output_tokens", 0) or 0
        usage = {"text_input_tokens": text_in, "image_input_tokens": image_in,
                 "output_tokens": image_out,
                 "total_tokens": getattr(u, "total_tokens", None)}
        rates = OPENAI_USD_PER_MTOK.get(self.model)
        if rates is None:
            return usage, self.estimate_cost()
        cost = (text_in * rates["text_in"]
                + image_in * rates["image_in"]
                + image_out * rates["image_out"]) / 1e6
        return usage, round(cost, 6)

    def edit(self, req: SubgoalRequest) -> SubgoalResult:
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "image": ("source.jpg", req.source_bytes, "image/jpeg"),
            "prompt": req.prompt,
            "size": self.size,
            "quality": self.quality,
            "output_format": "png",
            "n": 1,
        }
        # Only where the model accepts it — gpt-image-2 hard-fails the request.
        if self.sends_input_fidelity:
            kwargs["input_fidelity"] = self.input_fidelity
        try:
            resp = client.images.edit(**kwargs)
        except Exception as e:  # noqa: BLE001
            return SubgoalResult(None, "png", self.model, "edited",
                                 self.estimate_cost(), error=f"{type(e).__name__}: {e}")

        data = getattr(resp, "data", None) or []
        if not data or not getattr(data[0], "b64_json", None):
            return SubgoalResult(None, "png", self.model, "edited",
                                 self.estimate_cost(), error="no image data returned")
        img = base64.b64decode(data[0].b64_json)
        usage, cost = self._usage(resp)
        return SubgoalResult(
            image_bytes=img, ext="png", model=self.model, kind="edited",
            cost_usd_est=cost,
            meta={"quality": self.quality, "size": self.size,
                  "input_fidelity": self.input_fidelity if self.sends_input_fidelity else None,
                  "usage": usage},
        )
