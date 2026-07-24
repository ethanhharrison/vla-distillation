"""Generate subgoal images for an example set (Stage B), runnable by itself.

Given an example set (a handful of trajectory steps, each with three camera
frames, a real t+k future frame, and an instruction — see
`scripts/prepare_subgoal_examples.py`), produce a **subgoal** frame per camera
via one or more backends and one or more prompt templates:

- edited (`gemini_image`, `openai_image`) : the scene a few moments into the
  instruction — scene-level change, robot pose roughly unchanged.
- `real_future` : the REAL frame at t+k (no API call), tagged accordingly.
- `dummy_image` : an offline placeholder edit, for plumbing.

Both subgoal sources yield identically-shaped samples (tagged `subgoal_source`).
Every edited camera is cached (a rerun of the same config costs $0) and a hard
`ceiling_usd` aborts before overspending. Output is a self-describing
`results.json` + images that `scripts/summarize_subgoal_images.py` renders.

CLI examples
------------
    # offline plumbing (no spend): real future frames + a dummy edit
    python -m pipeline.subgoal_image.generate --examples outputs/subgoal_examples/<ep> \
        --no-spend --backend real_future dummy_image

    # real edits, both providers, two prompt variants, under a $5 ceiling
    python -m pipeline.subgoal_image.generate --examples outputs/subgoal_examples/<ep> \
        --ceiling 5.0 --backend gemini_image openai_image real_future \
        --prompt-template default minimal --limit 10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from . import prompts
from .backends import (
    SubgoalRequest,
    available_image_backends,
    build_image_backend,
    is_paid_backend,
)
from .cache import BlobCache
from .cost import BudgetExceeded, CostTracker
from .imaging import downscale, image_size, is_png, phash_delta, request_key, sha256_hex

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "subgoal_images"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"


@dataclass
class SubgoalConfig:
    examples_dir: Path
    backends: list[str] = field(default_factory=lambda: ["real_future", "dummy_image"])
    prompt_templates: list[str] = field(default_factory=lambda: [prompts.DEFAULT_TEMPLATE])
    cameras: list[str] | None = None
    limit: int | None = None
    instruction: str | None = None  # override each example's own instruction
    ceiling_usd: float = 5.0
    no_spend: bool = False
    gemini_model: str | None = None
    openai_model: str | None = None
    openai_quality: str = "low"
    openai_size: str = "auto"
    cache_dir: Path = DEFAULT_CACHE_DIR
    output_dir: Path | None = None


@dataclass
class SubgoalRun:
    config: dict
    output_dir: Path
    samples: list[dict]
    cost_summary: dict
    aborted: bool


def _variant_key(backend: str, model: str, tpl_id: str | None, k: int | None) -> str:
    if backend == "real_future":
        return f"real_future:k{k}"
    if tpl_id:
        return f"{backend}:{model}:{tpl_id}"
    return f"{backend}:{model}"


def generate_subgoals(config: SubgoalConfig) -> SubgoalRun:
    ex_dir = Path(config.examples_dir)
    meta = json.loads((ex_dir / "meta.json").read_text())
    examples = meta["examples"][: config.limit] if config.limit else meta["examples"]
    cameras = config.cameras or meta.get("cameras")

    out = Path(config.output_dir) if config.output_dir else (
        DEFAULT_OUTPUT_DIR / f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    cache = BlobCache(Path(config.cache_dir))
    tracker = CostTracker(ceiling_usd=config.ceiling_usd, costs_path=out / "costs.jsonl")

    # resolve backends, dropping paid ones under no_spend
    backends: list[str] = []
    for b in config.backends:
        if config.no_spend and is_paid_backend(b):
            print(f"[--no-spend] skipping paid backend {b!r}")
            continue
        backends.append(b)
    if not backends:
        raise SystemExit("no backends to run (all filtered by --no-spend?)")

    built = {}
    for b in backends:
        kwargs: dict = {}
        if b == "gemini_image" and config.gemini_model:
            kwargs["model"] = config.gemini_model
        if b == "openai_image":
            if config.openai_model:
                kwargs["model"] = config.openai_model
            kwargs["quality"] = config.openai_quality
            kwargs["size"] = config.openai_size
        built[b] = build_image_backend(b, **kwargs)

    tpls = [prompts.resolve_template(t) for t in config.prompt_templates]

    samples: list[dict] = []
    aborted = False

    for ex in examples:
        instruction = config.instruction or ex.get("instruction") or ""
        variants: list[tuple[str, str | None, str | None]] = []
        for b in backends:
            if b in ("real_future", "dummy_image"):
                variants.append((b, None, None))
            else:
                for name, text in tpls:
                    variants.append((b, name, text))

        for backend, tpl_name, tpl_text in variants:
            be = built[backend]
            tpl_id = prompts.template_id(tpl_text) if tpl_text else None
            prompt_text = prompts.build_prompt(tpl_text, instruction) if tpl_text else ""
            k = ex.get("future", {}).get("k")
            vkey = _variant_key(backend, be.model, tpl_id, k)

            sample = {
                "example_id": ex["id"],
                "step_index": ex.get("step_index"),
                "instruction": instruction,
                "variant_key": vkey,
                "subgoal_source": (
                    "real_future" if backend == "real_future"
                    else "dummy" if backend == "dummy_image" else "edited"
                ),
                "backend": backend,
                "model": be.model,
                "prompt_template": tpl_name,
                "prompt_template_id": tpl_id,
                "prompt": prompt_text,
                "k": k if backend == "real_future" else None,
                "cameras": {},
                "cost_usd": 0.0,
                "ts": datetime.now(timezone.utc).isoformat(),
            }

            for cam in cameras:
                src_rel = ex["cameras"].get(cam)
                if not src_rel:
                    continue
                src_bytes = (ex_dir / src_rel).read_bytes()
                fut_bytes = None
                fut = ex.get("future", {})
                if fut.get("cameras", {}).get(cam):
                    fut_bytes = (ex_dir / fut["cameras"][cam]).read_bytes()

                cam_rec = _produce_camera(
                    backend=backend, be=be, cam=cam, src_bytes=src_bytes,
                    fut_bytes=fut_bytes, instruction=instruction,
                    prompt_text=prompt_text, k=k, cache=cache, tracker=tracker,
                    img_out=img_out, ex_id=ex["id"], vkey=vkey,
                )
                if cam_rec is None:  # ceiling hit
                    aborted = True
                    break
                sample["cameras"][cam] = cam_rec
                sample["cost_usd"] = round(sample["cost_usd"] + cam_rec.get("cost_usd", 0.0), 6)

            samples.append(sample)
            if aborted:
                break
        if aborted:
            break

    run_config = {
        "examples_dir": str(ex_dir),
        "episode_id": meta.get("episode_id"),
        "dataset": meta.get("dataset"),
        "backends": backends,
        "prompt_templates": [
            {"name": n, "id": prompts.template_id(t), "text": t} for n, t in tpls
        ],
        "cameras": cameras,
        "ceiling_usd": config.ceiling_usd,
        "no_spend": config.no_spend,
        "limit": config.limit,
        "instruction_override": config.instruction,
        "openai": {"model": config.openai_model, "quality": config.openai_quality,
                   "size": config.openai_size},
        "gemini": {"model": config.gemini_model},
        "aborted_on_budget": aborted,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2))
    (out / "results.json").write_text(
        json.dumps({"run": run_config, "samples": samples}, indent=2)
    )

    summary = tracker.summary()
    _print_summary(samples, summary, out, aborted)
    return SubgoalRun(config=run_config, output_dir=out, samples=samples,
                      cost_summary=summary, aborted=aborted)


def _produce_camera(
    *, backend, be, cam, src_bytes, fut_bytes, instruction, prompt_text, k,
    cache, tracker, img_out, ex_id, vkey,
) -> dict | None:
    """Produce one subgoal camera image (cached or paid). Returns the record, or
    None if the budget ceiling was hit (signals the caller to stop)."""
    src_sha = sha256_hex(src_bytes)
    src_native = img_out / f"{ex_id}__{cam}__source.jpg"
    if not src_native.exists():
        src_native.write_bytes(src_bytes)
    src_224 = img_out / f"{ex_id}__{cam}__source224.jpg"
    if not src_224.exists():
        src_224.write_bytes(downscale(src_bytes))

    rec: dict = {
        "source": src_native.name, "source_224": src_224.name,
        "subgoal": None, "subgoal_224": None, "kind": None, "cached": False,
        "cost_usd": 0.0, "error": None, "phash": None, "native_size": None, "meta": {},
    }

    if backend == "real_future":
        key = request_key("real_future", cam,
                          sha256_hex(fut_bytes) if fut_bytes else "none", k)
    else:
        key = request_key(backend, be.model, getattr(be, "quality", ""),
                          getattr(be, "size", ""), getattr(be, "input_fidelity", ""),
                          cam, prompt_text, src_sha)

    cached = cache.lookup(key) if backend != "dummy_image" else None
    if cached is not None:
        img_bytes = cache.get_blob(cached["blob"])
        rec.update(kind=cached.get("kind"), cached=True, cost_usd=0.0,
                   meta=cached.get("meta", {}))
        tracker.record(backend=backend, model=be.model, cost_usd=0.0, cached=True,
                       example_id=ex_id, camera=cam, note="cache hit")
    else:
        est = be.estimate_cost()
        if be.is_paid:
            try:
                tracker.precheck(est, what=f"{backend}/{cam}")
            except BudgetExceeded as e:
                print(f"BUDGET CEILING HIT: {e}")
                return None
        result = be.edit(SubgoalRequest(
            source_bytes=src_bytes, camera=cam, instruction=instruction,
            prompt=prompt_text, future_bytes=fut_bytes, k=k,
        ))
        if result.error or result.image_bytes is None:
            rec.update(kind=result.kind, error=result.error or "no image", cost_usd=0.0)
            tracker.record(backend=backend, model=be.model, cost_usd=0.0, cached=False,
                           example_id=ex_id, camera=cam, note=f"ERROR: {result.error}")
            print(f"  ! {ex_id} {vkey} {cam}: {result.error}")
            return rec
        img_bytes = result.image_bytes
        paid = be.is_paid
        tracker.record(backend=backend, model=be.model, cost_usd=result.cost_usd_est,
                       cached=(not paid), example_id=ex_id, camera=cam)
        rec.update(kind=result.kind, cost_usd=(result.cost_usd_est if paid else 0.0),
                   meta=result.meta)
        if backend != "dummy_image":
            blob = cache.put_blob(img_bytes, result.ext)
            cache.store(key, {"blob": blob, "kind": result.kind, "meta": result.meta,
                              "cost_usd_est": result.cost_usd_est})

    ext = "png" if is_png(img_bytes) else "jpg"
    tag = vkey.replace(":", "_")
    sg_native = img_out / f"{ex_id}__{tag}__{cam}__subgoal.{ext}"
    sg_native.write_bytes(img_bytes)
    sg_224 = img_out / f"{ex_id}__{tag}__{cam}__subgoal224.jpg"
    sg_224.write_bytes(downscale(img_bytes))
    rec["subgoal"] = sg_native.name
    rec["subgoal_224"] = sg_224.name
    rec["native_size"] = image_size(img_bytes)
    rec["phash"] = phash_delta(src_bytes, img_bytes)
    return rec


def _print_summary(samples, cost_summary, out, aborted):
    print("\n=== Stage B (subgoal_image) summary ===")
    print(f"samples: {len(samples)}   out: {out}")
    if aborted:
        print("** RUN ABORTED ON BUDGET CEILING — partial results written **")
    by_backend: dict[str, list[float]] = {}
    errors = 0
    for s in samples:
        for r in s["cameras"].values():
            if r.get("error"):
                errors += 1
            elif r.get("phash"):
                by_backend.setdefault(s["backend"], []).append(r["phash"]["norm"])
    for b, deltas in sorted(by_backend.items()):
        ds = sorted(deltas)
        med = ds[len(ds) // 2]
        print(f"  {b}: {len(ds)} images  phash_norm min/med/max = "
              f"{min(ds):.3f}/{med:.3f}/{max(ds):.3f}")
    if errors:
        print(f"  errors: {errors} camera(s) failed (see results.json)")
    print(f"cost: {json.dumps(cost_summary)}")


def build_config_from_args(args: argparse.Namespace) -> SubgoalConfig:
    return SubgoalConfig(
        examples_dir=Path(args.examples),
        backends=args.backend,
        prompt_templates=args.prompt_template,
        cameras=args.cameras,
        limit=args.limit,
        instruction=args.instruction,
        ceiling_usd=args.ceiling,
        no_spend=args.no_spend,
        gemini_model=args.gemini_model,
        openai_model=args.openai_model,
        openai_quality=args.openai_quality,
        openai_size=args.openai_size,
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output) if args.output else None,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--examples", required=True, help="Example set dir (contains meta.json).")
    p.add_argument("--backend", nargs="+", default=["real_future", "dummy_image"],
                   help=f"Backends. Available: {', '.join(available_image_backends())}.")
    p.add_argument("--prompt-template", nargs="+", default=[prompts.DEFAULT_TEMPLATE],
                   help=f"Template name(s) or literal(s). Registered: {', '.join(prompts.TEMPLATES)}.")
    p.add_argument("--cameras", nargs="+", default=None,
                   help="Cameras to edit (default: all in the example set).")
    p.add_argument("--limit", type=int, default=None, help="Only the first N examples.")
    p.add_argument("--instruction", default=None,
                   help="Override the instruction for all examples (default: each example's own).")
    p.add_argument("--ceiling", type=float, default=5.0, help="Hard $ spend ceiling.")
    p.add_argument("--no-spend", action="store_true",
                   help="Drop paid backends; run only real_future/dummy_image.")
    p.add_argument("--gemini-model", default=None)
    p.add_argument("--openai-model", default=None)
    p.add_argument("--openai-quality", default="low", choices=["low", "medium", "high", "auto"])
    p.add_argument("--openai-size", default="auto")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                   help="Content-addressed edit cache (shared across runs; $0 reruns).")
    p.add_argument("--output", default=None, help="Run output dir.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    generate_subgoals(build_config_from_args(parse_args(argv)))


if __name__ == "__main__":
    main()
