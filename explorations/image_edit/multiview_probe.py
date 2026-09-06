"""Probe: can the hosted editors produce all three DROID views coherently?

Subgoal generation needs three mutually consistent camera views. Today
`pipeline/subgoal_image` edits each camera with its own independent API call,
which gives no coherence guarantee at all: nothing stops the marker moving in
`exterior_1` and staying put in `exterior_2`. This script measures two ways to
ask for the views *together*, against that independent baseline.

    independent  one call per camera — what ships today. The control.
    canvas       stitch the three views into ONE image, edit it once, split the
                 result back into three. One request, so the views *can* be
                 mutually consistent; also 3x cheaper per situation.
    multiturn    edit view 1, then ask for view 2 and 3 in the SAME conversation,
                 with the previous edits in context. Sequential, so it costs the
                 same as `independent`, and the coherence has to come from the
                 model attending to its own earlier images.

The two providers reach `multiturn` differently, and finding out whether it is
even possible is half the point of this probe:

- **Gemini** has real multi-turn image editing: `client.chats.create(...)` keeps
  the returned images in history, so turn 2 can refer to turn 1's edit.
- **OpenAI** `images.edit` is stateless. Two routes exist and both are tried, in
  order, with whichever succeeded recorded per sample: (1) the Responses API with
  the `image_generation` tool threaded by `previous_response_id` — a genuine
  conversation; (2) `images.edit` with a LIST of input images (the SDK's
  `image` param accepts a sequence), passing the already-edited view alongside
  the next source frame as a reference. (2) is not a conversation but it does
  condition view N on view N-1, which is the property we actually want.

Deliberately NOT a new backend in `pipeline/subgoal_image/backends.py`. The
`ImageEditBackend.edit()` contract is one frame in, one frame out, and both
strategies here break it (canvas needs a composite source and a split; multiturn
needs conversation state). Committing an interface before knowing whether either
works would be backwards — promote whichever wins afterwards.

What IS reused, so this measures the real thing and cannot overspend:
`BlobCache` + `CostTracker` (lookup -> precheck -> call -> record -> store, the
same order `edit.edit_camera` enforces), `pipeline.subgoal_image.prompts`, this
directory's `metrics.diff`, and the canvas geometry imported from
`explorations/cosmos3` so every harness in the repo stitches views identically.

Conditions: **A only**, plus (with `--noop`) each strategy's own "change nothing"
floor. The question is view coherence, not instruction sensitivity, so paying for
B/C here would buy nothing — but the floor is not optional if the edit magnitudes
are going to be compared across strategies, because a stitched composite and a
single frame do not pay the same re-render tax.

    ../../.venv/bin/python multiview_probe.py --dry-run
    ../../.venv/bin/python multiview_probe.py --backend gemini_image --ceiling 2
    ../../.venv/bin/python make_multiview_report.py --run multiview_gemini_image --open
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from pipeline.subgoal_image import prompts  # noqa: E402
from pipeline.subgoal_image.backends import build_image_backend  # noqa: E402
from pipeline.subgoal_image.cache import BlobCache  # noqa: E402
from pipeline.subgoal_image.cost import BudgetExceeded, CostTracker  # noqa: E402
from pipeline.subgoal_image.imaging import request_key, sha256_hex  # noqa: E402

from metrics import diff, load_rgb_bytes  # noqa: E402

#: The "change nothing" prompt, imported from the A/B/C harness rather than
#: restated, so a floor measured here is the same quantity as the 3.7 / 5.6
#: floors in the full-run table. Wording is used verbatim for canvases too —
#: adding "keep the grid" language would help the model hold the layout and so
#: understate the very tax this condition exists to measure.
from run_experiment import NOOP_PROMPT  # noqa: E402

DEFAULT_SITUATIONS = HERE.parent / "cosmos3/results/situations_multitraj"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs" / "subgoal_images" / "cache"
CAMERAS = ("exterior_1", "exterior_2", "wrist")


#: Canvas geometry, prompt suffix, cache keys and the single paid canvas call all
#: live in `canvas.py`, shared with `run_experiment.py` and `resample_null.py`.
#: They are NOT redefined here: this script's condition-A numbers are the A row of
#: the A/B/C run, so both must come from one implementation or the comparison is
#: between two different experiments.
from canvas import (  # noqa: E402
    LAYOUTS,
    Aborted,
    build as build_canvas,
    canvas_cache_key,
    canvas_prompt,
    edit_canvas,
    layout_check,
    single_edit as _canvas_single_edit,
)

MULTITURN_FIRST_SUFFIX = " This is camera view '{cam}' of the scene."
MULTITURN_NEXT_SUFFIX = (
    " This is a DIFFERENT camera view ('{cam}') of the SAME scene at the SAME "
    "moment as the image(s) you just edited. Apply the SAME single physical "
    "change you already made, seen from this viewpoint, so that all views agree "
    "with each other. Keep this view's own camera angle and framing unchanged."
)


# --------------------------------------------------------------------------- #
# provider call shapes
# --------------------------------------------------------------------------- #

def _gemini_image_part(data: bytes):
    from google.genai import types

    return types.Part.from_bytes(data=data, mime_type="image/png")


def _gemini_extract(resp) -> bytes:
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        raise RuntimeError("no candidates (refusal?)")
    parts = getattr(cands[0].content, "parts", None) or []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is not None and inline.data:
            return inline.data
    finish = getattr(cands[0], "finish_reason", None)
    txt = " ".join(getattr(p, "text", "") or "" for p in parts).strip()
    raise RuntimeError(f"no image part (finish={finish}); text={txt[:160]!r}")


class GeminiConversation:
    """Real multi-turn image editing: history carries the returned images."""

    route = "chats"

    def __init__(self, be):
        from google.genai import types

        self.be = be
        self._chat = be._get_client().chats.create(
            model=be.model,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )

    def turn(self, prompt: str, image: bytes):
        resp = self._chat.send_message([prompt, _gemini_image_part(image)])
        img = _gemini_extract(resp)
        usage, cost = self.be._usage(resp)
        return img, "png", cost, {"usage": usage, "route": self.route}


class OpenAIConversation:
    """Multi-turn on OpenAI, preferring a real conversation.

    Route 1 (`responses`): the Responses API's `image_generation` tool threaded
    with `previous_response_id` — genuinely stateful. Route 2 (`multi_image`):
    if route 1 errors on the first turn, fall back to `images.edit` with the
    previously edited views passed as extra reference images. Which route
    produced a sample is recorded, because the two are not the same claim.
    """

    def __init__(self, be, responses_model: str):
        self.be = be
        self.responses_model = responses_model
        self.route = "responses"
        self._prev_id: str | None = None
        self._edited: list[bytes] = []

    def _tool(self) -> dict:
        tool = {"type": "image_generation", "quality": self.be.quality,
                "size": self.be.size, "output_format": "png"}
        if self.be.sends_input_fidelity:
            tool["input_fidelity"] = self.be.input_fidelity
        return tool

    def _responses_turn(self, prompt: str, image: bytes):
        client = self.be._get_client()
        content = [
            {"type": "input_text", "text": prompt},
            {"type": "input_image",
             "image_url": "data:image/png;base64," + base64.b64encode(image).decode()},
        ]
        kwargs = {"model": self.responses_model,
                  "input": [{"role": "user", "content": content}],
                  "tools": [self._tool()]}
        if self._prev_id:
            kwargs["previous_response_id"] = self._prev_id
        resp = client.responses.create(**kwargs)
        imgs = [o for o in (resp.output or []) if getattr(o, "type", "") == "image_generation_call"]
        if not imgs or not getattr(imgs[0], "result", None):
            raise RuntimeError("responses: no image_generation_call result")
        self._prev_id = resp.id
        # The Responses usage payload does not break out image-output tokens the
        # way images.edit does, so the static per-image bound is recorded and
        # flagged rather than a derived number that would be wrong.
        return (base64.b64decode(imgs[0].result), "png", self.be.estimate_cost(),
                {"route": "responses", "response_id": resp.id, "cost_is_static_estimate": True})

    def turn(self, prompt: str, image: bytes):
        if self.route == "responses":
            try:
                out = self._responses_turn(prompt, image)
                self._edited.append(out[0])
                return out
            except Exception as exc:  # noqa: BLE001
                if self._prev_id is not None:
                    raise  # mid-conversation failure is a real result, not a route problem
                print(f"    responses route unavailable ({type(exc).__name__}: "
                      f"{str(exc)[:120]}); falling back to multi_image")
                self.route = "multi_image"

        img, ext, cost, meta = _canvas_single_edit(
            "openai_image", self.be, prompt, [*self._edited, image]
        )
        self._edited.append(img)
        meta["route"] = "multi_image"
        return img, ext, cost, meta


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def view_metrics(subgoal: np.ndarray, src: np.ndarray, real: np.ndarray | None) -> dict:
    """The harness's three standard numbers, per view.

    Same units as `run_experiment.py` / `analyze.py`: mean absolute pixel
    difference on 0-255 RGB, computed on the smaller grid.
    """
    return {
        "vs_src": round(diff(subgoal, src), 2),
        "vs_real": round(diff(subgoal, real), 2) if real is not None else None,
        "real_motion": round(diff(src, real), 2) if real is not None else None,
    }


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #

def strategy_independent(*, be, backend_name, sid, prompt, srcs, reals, out_dir,
                         cache, tracker, use_cache, sample_id,
                         tag: str = "independent") -> dict:
    """The control: exactly the path `pipeline/subgoal_image` runs today.

    `tag` names the variant on disk so the "change nothing" twin does not
    overwrite the real edit's images.
    """
    from pipeline.subgoal_image.edit import edit_camera

    rec = {"strategy": tag, "views": {}, "cost_usd": 0.0,
           "cached": True, "errors": [], "n_calls": 0}
    for cam in CAMERAS:
        outcome = edit_camera(
            backend_name=backend_name, backend=be, camera=cam,
            source_bytes=srcs[cam], instruction="", prompt=prompt,
            future_bytes=reals.get(cam), k=None, cache=cache, tracker=tracker,
            use_cache=use_cache, example_id=sample_id,
        )
        rec["n_calls"] += 1
        if outcome.over_budget:
            raise Aborted(outcome.error or "budget")
        if not outcome.ok:
            rec["errors"].append(f"{cam}: {outcome.error}")
            continue
        path = out_dir / f"{tag.replace(':', '_')}__{cam}.png"
        path.write_bytes(outcome.image_bytes)
        arr = load_rgb_bytes(outcome.image_bytes)
        rec["views"][cam] = {
            "file": path.name,
            "native_size": [arr.shape[1], arr.shape[0]],
            **view_metrics(arr, load_rgb_bytes(srcs[cam]),
                           load_rgb_bytes(reals[cam]) if reals.get(cam) else None),
        }
        rec["cost_usd"] = round(rec["cost_usd"] + outcome.cost_usd, 6)
        rec["cached"] = rec["cached"] and outcome.cached
    return rec


def strategy_canvas(*, be, backend_name, layout, sid, template_text, instruction,
                    srcs, reals, out_dir, cache, tracker, use_cache, sample_id,
                    noop: bool = False) -> dict:
    """One request over a stitched composite, split back into three views.

    With `noop=True` this becomes the canvas's own re-render floor: the same
    composite, asked to change nothing. Needed because a canvas edit magnitude
    is otherwise unreadable — "the canvas edits more" and "the canvas is
    re-rendered harder" predict the same number.
    """
    canvas_bytes, canvas_im, src_panels = build_canvas(layout, srcs)
    geom = LAYOUTS[layout][2]
    prompt = canvas_prompt(prompts.build_prompt(template_text, instruction),
                           noop_prompt=NOOP_PROMPT if noop else None)

    name = f"canvas:{layout}:noop" if noop else f"canvas:{layout}"
    rec: dict = {"strategy": name, "views": {}, "cost_usd": 0.0, "cached": False,
                 "errors": [], "n_calls": 1,
                 "canvas": {"layout": layout, "source_geometry": geom}}

    tag = f"{layout}_noop" if noop else layout
    src_path = out_dir / f"{tag}__canvas_source.png"
    src_path.write_bytes(canvas_bytes)
    rec["canvas"]["source_file"] = src_path.name

    res = edit_canvas(
        backend_name=backend_name, backend=be, layout=layout, prompt=prompt,
        canvas_bytes=canvas_bytes, cache=cache, tracker=tracker,
        use_cache=use_cache, example_id=sample_id,
    )
    if res["error"]:
        rec["errors"].append(res["error"])
        return rec
    rec.update(cost_usd=res["cost_usd"], cached=res["cached"])
    rec["latency_s"] = res["meta"].get("latency_s")

    out_path = out_dir / f"{tag}__canvas_output.png"
    out_path.write_bytes(res["image"])
    out_arr = load_rgb_bytes(res["image"])
    rec["canvas"].update(
        output_file=out_path.name,
        source_size=[canvas_im.width, canvas_im.height],
        output_size=[out_arr.shape[1], out_arr.shape[0]],
        source_aspect=round(canvas_im.width / canvas_im.height, 3),
        output_aspect=round(out_arr.shape[1] / out_arr.shape[0], 3),
    )

    out_panels = LAYOUTS[layout][1](out_arr)
    rec["canvas"]["layout_check"] = layout_check(out_panels, src_panels)
    for cam in CAMERAS:
        panel = out_panels[cam]
        p_path = out_dir / f"{tag}__panel__{cam}.png"
        Image.fromarray(panel.astype(np.uint8)).save(p_path)
        rec["views"][cam] = {
            "file": p_path.name,
            "native_size": [panel.shape[1], panel.shape[0]],
            **view_metrics(panel, src_panels[cam],
                           load_rgb_bytes(reals[cam]) if reals.get(cam) else None),
        }
    return rec


def multiturn_plan(*, be, backend_name, template_text, instruction, srcs) -> list[tuple]:
    """(turn, camera, prompt, cache_key) per turn.

    Each key folds in a rolling hash of every earlier turn's prompt and source
    frame, so a key identifies a position in a *specific* conversation. Without
    the prefix, turn 2 of one conversation would collide with turn 2 of another
    that happened to ask the same thing after a different turn 1.
    """
    plan, prefix = [], ""
    for i, cam in enumerate(CAMERAS):
        suffix = (MULTITURN_FIRST_SUFFIX if i == 0 else MULTITURN_NEXT_SUFFIX).format(cam=cam)
        prompt = prompts.build_prompt(template_text, instruction) + suffix
        prefix = sha256_hex(f"{prefix}|{prompt}|{sha256_hex(srcs[cam])}")
        key = request_key(backend_name, be.model, getattr(be, "quality", ""),
                          getattr(be, "size", ""), getattr(be, "input_fidelity", ""),
                          "multiturn", i, cam, prefix)
        plan.append((i, cam, prompt, key))
    return plan


def strategy_multiturn(*, be, backend_name, sid, template_text, instruction, srcs,
                       reals, out_dir, cache, tracker, use_cache, sample_id,
                       responses_model) -> dict:
    """Sequential edits in one conversation, each conditioned on the previous.

    Caching is **all-or-nothing across the conversation**. Turn 2's output
    depends on turn 1 having actually been sent, so reading turn 2 from cache
    while turn 1 was skipped would build a conversation whose history has holes
    in it — the model would be answering "the view you just edited" having edited
    nothing. So: every turn hits, or the whole conversation is replayed live.
    """
    plan = multiturn_plan(be=be, backend_name=backend_name,
                          template_text=template_text, instruction=instruction, srcs=srcs)
    hits = ([cache.lookup(k) for *_, k in plan] if (cache is not None and use_cache)
            else [None] * len(plan))
    replay = any(h is None for h in hits)

    rec: dict = {"strategy": "multiturn", "views": {}, "cost_usd": 0.0,
                 "cached": not replay, "errors": [], "n_calls": len(plan), "routes": []}
    conv = None
    if replay:
        conv = (GeminiConversation(be) if backend_name == "gemini_image"
                else OpenAIConversation(be, responses_model))

    for (i, cam, prompt, key), hit in zip(plan, hits):
        if not replay:
            img = cache.get_blob(hit["blob"])
            meta = hit.get("meta", {})
            tracker.record(backend=backend_name, model=be.model, cost_usd=0.0,
                           cached=True, example_id=sample_id,
                           camera=f"multiturn/{cam}", note="cache hit")
        else:
            try:
                tracker.precheck(be.estimate_cost(), what=f"{backend_name}/multiturn/{cam}")
            except BudgetExceeded as exc:
                raise Aborted(str(exc)) from exc
            t0 = time.time()
            try:
                img, ext, cost, meta = conv.turn(prompt, srcs[cam])
            except Exception as exc:  # noqa: BLE001
                tracker.record(backend=backend_name, model=be.model, cost_usd=0.0,
                               cached=False, example_id=sample_id,
                               camera=f"multiturn/{cam}",
                               note=f"ERROR: {type(exc).__name__}: {exc}")
                rec["errors"].append(f"{cam}: {type(exc).__name__}: {exc}")
                break  # a broken turn invalidates every later turn's context
            meta = dict(meta or {})
            meta["latency_s"] = round(time.time() - t0, 1)
            tracker.record(backend=backend_name, model=be.model, cost_usd=cost,
                           cached=False, example_id=sample_id, camera=f"multiturn/{cam}")
            rec["cost_usd"] = round(rec["cost_usd"] + cost, 6)
            if cache is not None and use_cache:
                cache.store(key, {"blob": cache.put_blob(img, ext), "kind": "edited",
                                  "meta": meta, "cost_usd_est": cost})

        if meta.get("route"):
            rec["routes"].append(meta["route"])
        path = out_dir / f"multiturn__{i}__{cam}.png"
        path.write_bytes(img)
        arr = load_rgb_bytes(img)
        rec["views"][cam] = {
            "file": path.name, "turn": i,
            "native_size": [arr.shape[1], arr.shape[0]],
            **view_metrics(arr, load_rgb_bytes(srcs[cam]),
                           load_rgb_bytes(reals[cam]) if reals.get(cam) else None),
        }
    return rec


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

def load_situation(sdir: Path, k: int) -> tuple[dict, dict, dict]:
    """(situation, source frames per camera, real t+k frames per camera)."""
    sit = json.loads((sdir / "situation.json").read_text())
    srcs, reals = {}, {}
    for cam in CAMERAS:
        src = sdir / "history" / f"{cam}_0.png"
        if src.exists():
            srcs[cam] = src.read_bytes()
        fut = sdir / "future" / f"{cam}_f{k:02d}.png"
        if fut.exists():
            reals[cam] = fut.read_bytes()
    return sit, srcs, reals


def plan_strategies(args) -> list[str]:
    """Strategy names to run, in report order.

    With `--noop` each strategy is paired with its own "change nothing" twin.
    The floor has to be measured per strategy rather than borrowed from the
    single-frame table: the re-render tax on a stitched 640x360 composite is not
    the same quantity as the tax on one 320x180 frame, and without the matching
    floor "the canvas edits more" is indistinguishable from "the canvas gets
    re-rendered harder".
    """
    out = []
    for s in args.strategies:
        names = [f"canvas:{lay}" for lay in args.layouts] if s == "canvas" else [s]
        for n in names:
            out.append(n)
            if args.noop:
                out.append(f"{n}:noop")
    return out


def calls_per_situation(strategies: list[str]) -> int:
    """canvas is one call for all three views; the other two are one per view."""
    return sum(1 if s.startswith("canvas") else len(CAMERAS) for s in strategies)


def project_spend(*, be, backend_name, sit_dir, situations, strategies, tpl_text,
                  cache, use_cache, k) -> tuple[int, int]:
    """(calls, calls that would actually be paid for) -> the honest projection.

    A projection that counts cached calls is worse than no projection: it inflates
    the number the `--ceiling` decision is made from, so a run whose real cost is
    $2.66 gets refused at a $7.20 ceiling that would never have been reached.
    This walks the exact plan and asks the cache, using the same key builders the
    run uses.
    """
    from pipeline.subgoal_image.edit import edit_cache_key

    total = paid = 0
    for sid in situations:
        sdir = sit_dir / sid
        sit = json.loads((sdir / "situation.json").read_text())
        srcs = {c: (sdir / "history" / f"{c}_0.png").read_bytes()
                for c in CAMERAS if (sdir / "history" / f"{c}_0.png").exists()}
        if len(srcs) < len(CAMERAS):
            continue
        edit_prompt = prompts.build_prompt(tpl_text, sit["instruction"])
        for strat in strategies:
            is_noop = strat.endswith(":noop")
            base = strat[: -len(":noop")] if is_noop else strat
            if base.startswith("canvas:"):
                layout = base.split(":", 1)[1]
                canvas_bytes = build_canvas(layout, srcs)[0]
                prompt = canvas_prompt(edit_prompt,
                                       noop_prompt=NOOP_PROMPT if is_noop else None)
                keys = [canvas_cache_key(backend_name, be, layout=layout,
                                        prompt=prompt, canvas_bytes=canvas_bytes)]
            elif base == "independent":
                prompt = NOOP_PROMPT if is_noop else edit_prompt
                keys = [edit_cache_key(backend_name, be, camera=c, prompt=prompt,
                                       source_bytes=srcs[c]) for c in CAMERAS]
            else:  # multiturn: all-or-nothing, and counted conservatively
                keys = [None] * len(CAMERAS)
            total += len(keys)
            for key in keys:
                hit = (cache.lookup(key) is not None
                       if (key and cache is not None and use_cache) else False)
                paid += not hit
    return total, paid


def run(args) -> None:
    sit_dir = Path(args.situations).resolve()
    meta = json.loads((sit_dir / "meta.json").read_text())
    lim = args.limit_situations if args.limit_situations and args.limit_situations > 0 else None
    situations = meta["situations"][:lim]
    k = args.k if args.k is not None else int(meta.get("horizon") or 0)
    strategies = plan_strategies(args)

    backend_kwargs: dict = {}
    if args.model:
        backend_kwargs["model"] = args.model
    if args.backend == "openai_image":
        backend_kwargs["quality"] = args.openai_quality
        backend_kwargs["size"] = args.openai_size
    be = build_image_backend(args.backend, **backend_kwargs)
    tpl_name, tpl_text = prompts.resolve_template(args.prompt_template)

    cache = BlobCache(Path(args.cache_dir))
    n_calls, n_paid = project_spend(
        be=be, backend_name=args.backend, sit_dir=sit_dir, situations=situations,
        strategies=strategies, tpl_text=tpl_text, cache=cache,
        use_cache=not args.no_cache, k=k)
    per_call = be.estimate_cost()
    projected = n_paid * per_call
    print(f"{args.backend} / {be.model}   template={tpl_name}   k={k}")
    print(f"  {len(situations)} situations x {strategies} = {n_calls} calls")
    print(f"  {n_calls - n_paid} already cached, {n_paid} to pay for")
    print(f"  projected spend <= ${projected:.2f} (${per_call:.4f}/image upper bound)")
    if args.dry_run:
        print("  --dry-run: nothing called.")
        return
    if projected > args.ceiling:
        raise SystemExit(
            f"ABORT: projected ${projected:.2f} exceeds --ceiling ${args.ceiling:.2f}.\n"
            f"       Raise the ceiling or cut the grid (--limit-situations / "
            f"--strategies / --layouts).\n"
            f"       Refusing up front rather than aborting mid-run, which would "
            f"leave a run that cannot be reported on."
        )

    run_dir = HERE / "results" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = CostTracker(ceiling_usd=args.ceiling, costs_path=run_dir / "costs.jsonl")

    samples: list[dict] = []
    aborted = False
    for sid in situations:
        sdir = sit_dir / sid
        sit, srcs, reals = load_situation(sdir, k)
        if len(srcs) < len(CAMERAS):
            print(f"  skip {sid}: only {len(srcs)}/{len(CAMERAS)} cameras on disk")
            continue
        instruction = sit["instruction"]
        out_dir = run_dir / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        for cam, data in srcs.items():
            (out_dir / f"source__{cam}.png").write_bytes(data)
            if reals.get(cam):
                (out_dir / f"real__{cam}.png").write_bytes(reals[cam])

        print(f"\n[{sid}] {instruction!r}")
        for strat in strategies:
            sample_id = f"{sid}__{strat}"
            common = dict(be=be, backend_name=args.backend, sid=sid, srcs=srcs,
                          reals=reals, out_dir=out_dir, cache=cache, tracker=tracker,
                          use_cache=not args.no_cache, sample_id=sample_id)
            is_noop = strat.endswith(":noop")
            base = strat[: -len(":noop")] if is_noop else strat
            try:
                if base == "independent":
                    rec = strategy_independent(
                        prompt=NOOP_PROMPT if is_noop
                        else prompts.build_prompt(tpl_text, instruction),
                        tag=strat, **common)
                elif base.startswith("canvas:"):
                    rec = strategy_canvas(layout=base.split(":", 1)[1],
                                          template_text=tpl_text,
                                          instruction=instruction, noop=is_noop,
                                          **common)
                else:
                    rec = strategy_multiturn(template_text=tpl_text,
                                             instruction=instruction,
                                             responses_model=args.responses_model,
                                             **common)
            except Aborted as exc:
                print(f"  BUDGET CEILING HIT: {exc}")
                aborted = True
                break

            rec.update(situation_id=sid, instruction=instruction,
                       prompt_template=tpl_name)
            samples.append(rec)
            flag = " (cached)" if rec.get("cached") else ""
            lay = rec.get("canvas", {}).get("layout_check")
            extra = ""
            if lay is not None:
                extra = f"  layout {lay['panels_matched']}/{lay['n_panels']} panels in place"
            if rec.get("routes"):
                extra += f"  route={rec['routes'][0]}"
            mags = "/".join(f"{v['vs_src']:.0f}" for v in rec["views"].values()) or "-"
            print(f"  {strat:<18} ${rec['cost_usd']:.3f}{flag}  vs_src {mags}{extra}"
                  + (f"  ERR {rec['errors']}" if rec["errors"] else ""))
        if aborted:
            break

    index = {
        "kind": "multiview_probe",
        "backend": args.backend, "model": be.model,
        "openai": {"quality": args.openai_quality, "size": args.openai_size}
        if args.backend == "openai_image" else None,
        "responses_model": args.responses_model if args.backend == "openai_image" else None,
        "prompt_template": tpl_name,
        "situations_dir": str(sit_dir), "k": k, "strategies": strategies,
        "cameras": list(CAMERAS),
        "canvas_geometry": {lay: LAYOUTS[lay][2] for lay in args.layouts},
        "aborted_on_budget": aborted,
        "cost": tracker.summary(),
        "ts": datetime.now(timezone.utc).isoformat(),
        "samples": samples,
    }
    (run_dir / "results.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {run_dir}/results.json")
    print(f"cost: {json.dumps(tracker.summary())}")
    if aborted:
        print("** ABORTED ON BUDGET CEILING — partial results written **")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--situations", default=str(DEFAULT_SITUATIONS))
    p.add_argument("--backend", default="gemini_image",
                   choices=["gemini_image", "openai_image"],
                   help="Only the paid editors have a multi-view question to answer.")
    p.add_argument("--model", default=None, help="Model override.")
    p.add_argument("--run-name", default=None, help="Output dir under results/.")
    p.add_argument("--strategies", nargs="+", default=["independent", "canvas", "multiturn"],
                   choices=["independent", "canvas", "multiturn"])
    p.add_argument("--layouts", nargs="+", default=list(LAYOUTS), choices=list(LAYOUTS),
                   help="Canvas layouts to try (one call each).")
    p.add_argument("--limit-situations", type=int, default=2,
                   help="Situations to probe; 0 or less means the whole set.")
    p.add_argument("--noop", action="store_true",
                   help="Also run each strategy's own 'change nothing' twin, to "
                        "measure that strategy's re-render tax. One extra call per "
                        "canvas, three per independent/multiturn.")
    p.add_argument("--prompt-template", default=prompts.DEFAULT_TEMPLATE,
                   help=f"Registered: {', '.join(prompts.TEMPLATES)}.")
    p.add_argument("--k", type=int, default=None,
                   help="Future offset for the real reference (default: set horizon).")
    p.add_argument("--responses-model", default="gpt-5.6-sol",
                   help="Model hosting the image_generation tool on the OpenAI "
                        "Responses API (the true multi-turn route).")
    p.add_argument("--openai-quality", default="low",
                   choices=["low", "medium", "high", "auto"])
    p.add_argument("--openai-size", default="auto")
    p.add_argument("--ceiling", type=float, default=2.0, help="Hard $ spend ceiling.")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--no-cache", action="store_true",
                   help="Never read or write the edit cache.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the call count and spend projection; call nothing.")
    args = p.parse_args(argv)
    if not args.run_name:
        args.run_name = f"multiview_{args.backend}"
    return args


if __name__ == "__main__":
    run(parse_args())
