"""Swappable Vision-Language-Model backends.

Every backend implements the same `VLM.generate(prompt, images)` interface, so
the pipeline is agnostic to which provider is used. Add a new provider by
subclassing `VLM` and decorating it with `@register_vlm("name")`; it then
becomes selectable via `build_vlm("name", ...)`.
"""

from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from typing import Callable, Type

from openai import OpenAI
from google import genai
from google.genai import types

# Default model per provider; override with `build_vlm(..., model=...)`.
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "gemini": "gemini-3.6-flash",
    "dummy": "dummy",
}

MODEL_REGISTRY: dict[str, Type[VLM]] = {}

class VLM(ABC):
    """Common interface for a vision-language model backend."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, images: list[bytes]) -> str:
        raise NotImplementedError("Base VLM Class does not have built-in generate function")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"


def register_vlm(name: str) -> Callable[[Type[VLM]], Type[VLM]]:
    """Class decorator that registers a VLM backend under `name`."""
    def decorator(cls: Type[VLM]) -> Type[VLM]:
        MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def available_providers() -> list[str]:
    return sorted(MODEL_REGISTRY)

def build_vlm(provider: str, model: str | None = None, **kwargs) -> VLM:
    """Instantiate a registered VLM backend by provider name."""
    key = provider.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown VLM provider {provider!r}. Available: {', '.join(available_providers())}")
    resolved_model = model or DEFAULT_MODELS.get(key)
    return MODEL_REGISTRY[key](model=resolved_model, **kwargs)

@register_vlm("openai")
class OpenAIVLM(VLM):
    """OpenAI chat-completions backend (GPT-4o and friends)."""

    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        super().__init__(model)
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._extra = kwargs

    def generate(self, prompt: str, images: list[bytes]) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images:
            b64 = base64.b64encode(image).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            **self._extra,
        )
        return response.choices[0].message.content or ""

@register_vlm("gemini")
class GeminiVLM(VLM):
    """Google Gemini backend via the `google-genai` SDK."""

    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        super().__init__(model)
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=resolved_key)
        self._extra = kwargs

    def generate(self, prompt: str, images: list[bytes]) -> str:
        parts: list = [types.Part.from_text(text=prompt)]
        for image in images:
            parts.append(types.Part.from_bytes(data=image, mime_type="image/jpeg"))
        response = self.client.models.generate_content(
            model=self.model,
            contents=parts,
            **self._extra,
        )
        return response.text or ""

@register_vlm("dummy")
class DummyVLM(VLM):
    """Offline backend for testing the pipeline without any API calls."""

    def generate(self, prompt: str, images: list[bytes]) -> str:
        return "\n".join(
            [
                "Pick up the object on the table",
                "Move the arm toward the target",
                "Place the item at the goal location",
            ]
        )
