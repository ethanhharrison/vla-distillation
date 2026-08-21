"""One cached, budget-checked image edit — the operation Stage B and its eval share.

Two callers need exactly this and nothing more: `generate.py`, which walks an
example set, and `explorations/image_edit`, which walks a situation set to
measure instruction sensitivity. Both must key the request, look it up, precheck
the ceiling, call the backend, record the spend and store the blob — in that
order, every time. That sequence *is* the money-safety guarantee, so it lives in
one place instead of being reimplemented per caller, where the two copies would
quietly drift apart on the detail that matters (which inputs enter the cache key).

Callers keep their own on-disk record formats; this returns the outcome only.

The cache key is deliberately over-specified: every parameter that can change the
returned bytes goes in, so two runs that differ in any of them are separate
entries rather than one wrong cache hit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backends import ImageEditBackend, SubgoalRequest
from .cache import BlobCache
from .cost import BudgetExceeded, CostTracker
from .imaging import is_png, request_key, sha256_hex


@dataclass
class EditOutcome:
    """What one edit produced — served from cache, freshly paid for, or refused."""

    image_bytes: bytes | None
    kind: str | None = None          # "edited" | "real_future" | "dummy"
    cached: bool = False
    cost_usd: float = 0.0
    error: str | None = None
    meta: dict = field(default_factory=dict)
    #: The ceiling refused the call *before* it was made, so nothing was spent.
    #: Distinct from `error`: a backend error is one bad sample and the run goes
    #: on, whereas this means the budget is gone and the caller must stop.
    over_budget: bool = False

    @property
    def ok(self) -> bool:
        return self.image_bytes is not None

    @property
    def ext(self) -> str:
        return "png" if self.image_bytes and is_png(self.image_bytes) else "jpg"


def edit_cache_key(
    backend_name: str,
    backend: ImageEditBackend,
    *,
    camera: str,
    prompt: str,
    source_bytes: bytes,
    future_bytes: bytes | None = None,
    k: int | None = None,
) -> str:
    """The content-addressed key for one edit request.

    `real_future` is keyed on the future frame it returns rather than on a
    prompt, because no model runs and the frame is the whole answer. Kept
    byte-for-byte compatible with the key `generate.py` used before this module
    existed, so entries already in the cache still hit.
    """
    if backend_name == "real_future":
        return request_key(
            "real_future", camera,
            sha256_hex(future_bytes) if future_bytes else "none", k,
        )
    return request_key(
        backend_name, backend.model,
        getattr(backend, "quality", ""), getattr(backend, "size", ""),
        getattr(backend, "input_fidelity", ""),
        camera, prompt, sha256_hex(source_bytes),
    )


def edit_camera(
    *,
    backend_name: str,
    backend: ImageEditBackend,
    camera: str,
    source_bytes: bytes,
    instruction: str,
    prompt: str,
    future_bytes: bytes | None = None,
    k: int | None = None,
    cache: BlobCache | None = None,
    tracker: CostTracker | None = None,
    use_cache: bool = True,
    example_id: str | None = None,
) -> EditOutcome:
    """Produce one subgoal image, paying at most once for any given request.

    `use_cache=False` forces a fresh call and skips the write, which is what a
    resample measurement needs: an identical request served from cache would
    report a difference of exactly zero and silently turn the null control into
    a rubber stamp.
    """
    # dummy_image is excluded from the cache on purpose: it is free, and caching
    # a placeholder would mask a backend that stopped producing real edits.
    cacheable = cache is not None and use_cache and backend_name != "dummy_image"
    key = edit_cache_key(
        backend_name, backend, camera=camera, prompt=prompt,
        source_bytes=source_bytes, future_bytes=future_bytes, k=k,
    )

    if cacheable:
        hit = cache.lookup(key)
        if hit is not None:
            if tracker is not None:
                tracker.record(
                    backend=backend_name, model=backend.model, cost_usd=0.0,
                    cached=True, example_id=example_id, camera=camera,
                    note="cache hit",
                )
            return EditOutcome(
                image_bytes=cache.get_blob(hit["blob"]), kind=hit.get("kind"),
                cached=True, cost_usd=0.0, meta=hit.get("meta", {}),
            )

    # The ceiling is checked before the call, never after, so a runaway config
    # cannot spend past it even once.
    if backend.is_paid and tracker is not None:
        try:
            tracker.precheck(backend.estimate_cost(), what=f"{backend_name}/{camera}")
        except BudgetExceeded as exc:
            return EditOutcome(None, error=str(exc), over_budget=True)

    result = backend.edit(SubgoalRequest(
        source_bytes=source_bytes, camera=camera, instruction=instruction,
        prompt=prompt, future_bytes=future_bytes, k=k,
    ))

    if result.error or result.image_bytes is None:
        if tracker is not None:
            tracker.record(
                backend=backend_name, model=backend.model, cost_usd=0.0,
                cached=False, example_id=example_id, camera=camera,
                note=f"ERROR: {result.error}",
            )
        return EditOutcome(
            None, kind=result.kind, error=result.error or "no image",
            meta=result.meta,
        )

    paid = backend.is_paid
    if tracker is not None:
        tracker.record(
            backend=backend_name, model=backend.model, cost_usd=result.cost_usd_est,
            cached=not paid, example_id=example_id, camera=camera,
        )
    if cacheable:
        blob = cache.put_blob(result.image_bytes, result.ext)
        cache.store(key, {
            "blob": blob, "kind": result.kind, "meta": result.meta,
            "cost_usd_est": result.cost_usd_est,
        })

    return EditOutcome(
        image_bytes=result.image_bytes, kind=result.kind, cached=False,
        cost_usd=result.cost_usd_est if paid else 0.0, meta=result.meta,
    )
