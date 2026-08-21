"""Money-safety of the shared edit path, and the model-specific call shapes.

The invariants here are the ones whose failure costs real money or silently
invalidates a measurement: pay at most once per request, never exceed the
ceiling, never serve a cached image to the resample control, and send each model
only the parameters it accepts.
"""

from __future__ import annotations

import pytest

from pipeline.subgoal_image.backends import (
    OPENAI_INPUT_FIDELITY_MODELS,
    ImageEditBackend,
    SubgoalRequest,
    SubgoalResult,
    build_image_backend,
    register_image_backend,
)
from pipeline.subgoal_image.cache import BlobCache
from pipeline.subgoal_image.cost import CostTracker
from pipeline.subgoal_image.edit import edit_cache_key, edit_camera

SRC = b"\xff\xd8\xff\xe0source-frame-bytes"
FUTURE = b"\xff\xd8\xff\xe0real-future-bytes"


@register_image_backend("counting_test_image")
class CountingBackend(ImageEditBackend):
    """A paid backend that counts calls and returns a distinct image each time."""

    is_paid = True
    default_model = "counting-v1"

    def __init__(self, model=None, price=0.10, fail=False, **kwargs):
        super().__init__(model=model, **kwargs)
        self.calls = 0
        self.price = price
        self.fail = fail

    def estimate_cost(self) -> float:
        return self.price

    def edit(self, req: SubgoalRequest) -> SubgoalResult:
        self.calls += 1
        if self.fail:
            return SubgoalResult(None, "png", self.model, "edited", self.price,
                                 error="synthetic failure")
        return SubgoalResult(
            image_bytes=f"edited-{self.calls}".encode(), ext="png", model=self.model,
            kind="edited", cost_usd_est=self.price, meta={"call": self.calls},
        )


def _edit(backend, cache, tracker, *, use_cache=True, prompt="do the thing", camera="exterior_1"):
    return edit_camera(
        backend_name="counting_test_image", backend=backend, camera=camera,
        source_bytes=SRC, instruction="do the thing", prompt=prompt,
        cache=cache, tracker=tracker, use_cache=use_cache,
    )


@pytest.fixture
def cache(tmp_path):
    return BlobCache(tmp_path / "cache")


# --------------------------------------------------------------------------- #
# the cache is the money-safety layer
# --------------------------------------------------------------------------- #

def test_identical_request_is_paid_for_once(cache):
    backend = CountingBackend()
    tracker = CostTracker(ceiling_usd=10.0)

    first = _edit(backend, cache, tracker)
    second = _edit(backend, cache, tracker)

    assert backend.calls == 1, "the second identical request re-hit the API"
    assert first.cost_usd == pytest.approx(0.10)
    assert second.cached and second.cost_usd == 0.0
    assert second.image_bytes == first.image_bytes
    assert tracker.spent_usd == pytest.approx(0.10)


def test_use_cache_false_always_re_calls(cache):
    """What the resample null depends on: a served-from-cache null would be 0.0."""
    backend = CountingBackend()
    tracker = CostTracker(ceiling_usd=10.0)

    _edit(backend, cache, tracker)
    again = _edit(backend, cache, tracker, use_cache=False)

    assert backend.calls == 2
    assert not again.cached
    assert again.image_bytes != b"edited-1", "an uncached repeat returned the first image"


def test_a_different_prompt_is_a_different_request(cache):
    backend = CountingBackend()
    tracker = CostTracker(ceiling_usd=10.0)

    _edit(backend, cache, tracker, prompt="open the drawer")
    _edit(backend, cache, tracker, prompt="close the drawer")

    assert backend.calls == 2


# --------------------------------------------------------------------------- #
# the ceiling
# --------------------------------------------------------------------------- #

def test_ceiling_refuses_before_spending(cache):
    backend = CountingBackend(price=0.10)
    tracker = CostTracker(ceiling_usd=0.15)

    first = _edit(backend, cache, tracker, camera="exterior_1")
    second = _edit(backend, cache, tracker, camera="exterior_2")

    assert first.ok
    assert second.over_budget and not second.ok
    assert backend.calls == 1, "the refused call still reached the backend"
    assert tracker.spent_usd == pytest.approx(0.10)


def test_backend_error_is_not_a_budget_abort(cache):
    """One bad sample must not look like an exhausted budget: the run goes on."""
    backend = CountingBackend(fail=True)
    tracker = CostTracker(ceiling_usd=10.0)

    outcome = _edit(backend, cache, tracker)

    assert not outcome.ok
    assert not outcome.over_budget
    assert outcome.error == "synthetic failure"
    assert tracker.spent_usd == 0.0, "a failed edit was billed"


# --------------------------------------------------------------------------- #
# cache keys
# --------------------------------------------------------------------------- #

def test_key_separates_models_and_quality():
    a = build_image_backend("openai_image", model="gpt-image-2", quality="low")
    b = build_image_backend("openai_image", model="gpt-image-2", quality="high")
    c = build_image_backend("openai_image", model="gpt-image-1.5", quality="low")

    keys = {
        edit_cache_key("openai_image", be, camera="wrist", prompt="p", source_bytes=SRC)
        for be in (a, b, c)
    }
    assert len(keys) == 3, "two configurations that bill differently share a cache entry"


def test_real_future_is_keyed_on_the_frame_not_the_prompt():
    backend = build_image_backend("real_future")
    common = dict(camera="wrist", source_bytes=SRC, future_bytes=FUTURE, k=30)

    same = edit_cache_key("real_future", backend, prompt="one", **common)
    other = edit_cache_key("real_future", backend, prompt="two", **common)
    moved = edit_cache_key("real_future", backend, prompt="one",
                           **{**common, "future_bytes": b"different"})

    assert same == other, "no model runs, so the prompt cannot change the frame returned"
    assert same != moved


# --------------------------------------------------------------------------- #
# provider call shapes (verified against the live APIs 2026-08-20)
# --------------------------------------------------------------------------- #

def test_input_fidelity_is_sent_only_where_supported():
    """gpt-image-2 rejects the parameter with a 400; mini never supported it."""
    assert not build_image_backend("openai_image", model="gpt-image-2").sends_input_fidelity
    assert not build_image_backend("openai_image", model="gpt-image-1-mini").sends_input_fidelity
    assert build_image_backend("openai_image", model="gpt-image-1.5").sends_input_fidelity
    assert OPENAI_INPUT_FIDELITY_MODELS == {"gpt-image-1", "gpt-image-1.5"}


class _Details:
    text_tokens = 44
    image_tokens = 240


class _OpenAIUsage:
    input_tokens = 284
    input_tokens_details = _Details()
    output_tokens = 129
    total_tokens = 413


class _OpenAIResponse:
    usage = _OpenAIUsage()


class _GeminiUsage:
    prompt_token_count = 298
    candidates_token_count = 1389
    total_token_count = 1687


class _GeminiResponse:
    usage_metadata = _GeminiUsage()


def test_openai_actual_cost_comes_from_reported_usage():
    """The reference call: 240 image-in + 44 text-in + 129 image-out on gpt-image-2."""
    backend = build_image_backend("openai_image", model="gpt-image-2", quality="low")
    usage, cost = backend._usage(_OpenAIResponse())

    assert usage["image_input_tokens"] == 240
    # 240*$8 + 44*$5 + 129*$30 per 1M tokens
    assert cost == pytest.approx(0.006010, abs=1e-6)
    assert cost < backend.estimate_cost(), "the ceiling estimate must not under-state the bill"


def test_gemini_actual_cost_comes_from_reported_usage():
    backend = build_image_backend("gemini_image", model="gemini-3.1-flash-image")
    usage, cost = backend._usage(_GeminiResponse())

    assert usage["output_tokens"] == 1389
    assert cost == pytest.approx(1389 * 60.0 / 1e6, abs=1e-6)
    assert cost < backend.estimate_cost()


def test_unknown_model_falls_back_to_the_static_estimate():
    backend = build_image_backend("gemini_image", model="gemini-99-imaginary-image")
    _, cost = backend._usage(_GeminiResponse())

    assert cost == backend.estimate_cost() == 0.20
