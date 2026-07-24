"""Swappable Vision-Language-Model backends."""

from __future__ import annotations

import base64
import io
import os
from abc import ABC, abstractmethod
from collections.abc import Callable

# Local HuggingFace VLM deps (run "uv pip install 'transformers>=4.57' torch torchvision accelerate pillow" to use, or comment out if not using)
import torch
from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .pricing import Usage

# Default model per provider; override with `build_vlm(..., model=...)`.
DEFAULT_MODELS = {
    "openai": "gpt-5.6-sol",
    "gemini": "gemini-3.6-flash",
    "hf": "Qwen/Qwen3-VL-4B-Instruct",
    "dummy": "dummy",
}

MODEL_REGISTRY: dict[str, type[VLM]] = {}

class VLM(ABC):
    """Common interface for a vision-language model backend."""

    def __init__(self, model: str):
        self.model = model
        self.usage = Usage()

    @abstractmethod
    def generate(self, prompt: str, images: list[bytes]) -> str:
        raise NotImplementedError("Base VLM Class does not have built-in generate function")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"


def register_vlm(name: str) -> Callable[[type[VLM]], type[VLM]]:
    """Class decorator that registers a VLM backend under `name`."""
    def decorator(cls: type[VLM]) -> type[VLM]:
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
    """OpenAI chat-completions backend"""

    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        super().__init__(model)
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.extra = kwargs

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
            **self.extra,
        )
        usage = response.usage
        self.usage.add(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
        return response.choices[0].message.content or ""

@register_vlm("gemini")
class GeminiVLM(VLM):
    """Google Gemini backend via the `google-genai` SDK."""

    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        super().__init__(model)
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=resolved_key)
        self.extra = kwargs

    def generate(self, prompt: str, images: list[bytes]) -> str:
        parts: list = [types.Part.from_text(text=prompt)]
        for image in images:
            parts.append(types.Part.from_bytes(data=image, mime_type="image/jpeg"))
        response = self.client.models.generate_content(
            model=self.model,
            contents=parts,
            **self.extra,
        )
        meta = response.usage_metadata
        self.usage.add(
            input_tokens=meta.prompt_token_count if meta else 0,
            output_tokens=meta.candidates_token_count if meta else 0,
        )
        return response.text or ""

@register_vlm("hf")
class HuggingFaceVLM(VLM):
    """Local image-text-to-text backend running downloaded weights via `transformers`."""

    def __init__(
        self,
        model: str,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 512,
        **kwargs,
    ):
        super().__init__(model)

        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model)
        self.hf_model = AutoModelForImageTextToText.from_pretrained(model, dtype=dtype, device_map=device_map)
        self.hf_model.eval()
        self.extra = kwargs

    def generate(self, prompt: str, images: list[bytes]) -> str:
        pil_images = [Image.open(io.BytesIO(img)).convert("RGB") for img in images]
        content: list[dict] = [{"type": "image"} for _ in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=pil_images or None,
            padding=True,
            return_tensors="pt",
        ).to(self.hf_model.device)

        with torch.no_grad():
            generated = self.hf_model.generate(**inputs, max_new_tokens=self.max_new_tokens, **self.extra)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        self.usage.add(
            input_tokens=int(inputs.input_ids.shape[-1]),
            output_tokens=int(trimmed[0].shape[-1]),
        )
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

@register_vlm("dummy")
class DummyVLM(VLM):
    """Offline backend for testing the pipeline without any API calls."""

    def generate(self, prompt: str, images: list[bytes]) -> str:
        self.usage.add()
        return "Pick up the object on the table\nMove the arm toward the target\nPlace the item at the goal location"
