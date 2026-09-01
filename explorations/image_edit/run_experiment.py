"""Run the hosted image-editor A/B/C experiment over a shared situation set.

Same situations, same conditions and same units as `explorations/cosmos3` and
`explorations/dreamzero`, so hosted image editing lands in tables that can be
read next to the world models instead of only next to itself. The situation sets
and their `conditions.json` are reused verbatim — nothing here re-derives them.

Backends come from `pipeline.subgoal_image`, so this measures the code Stage B
would actually run, not a reimplementation of it.

Differences forced by the modality, none of them cosmetic:

- **One image, not a clip.** Every per-clip metric the video harnesses report
  (jitter, drift, per-timestep fidelity) has no analogue. What survives is
  `vs src` (how far the edit moved), `vs real` (distance from what actually
  happened at t+k) and `vs A` (how much swapping the instruction mattered).
- **No seed.** Hosted APIs resample on every call, so the control is the
  *resample* null rather than a seed null — see `resample_null.py`. It plays the
  identical role: an instruction effect below it is indistinguishable from
  asking the same question twice.
- **No free lunch.** Every call is billed, so this refuses to start when the
  projected spend exceeds `--ceiling`, rather than aborting halfway and leaving
  a run that cannot be analysed. Reruns are $0 (the Stage B edit cache).
- **No reconstruction floor** unless you ask for one. A diffusion world model
  pays a measurable VAE tax on frame 0 that bounds how good its numbers can be;
  an editor's equivalent is how much it perturbs the scene when told to change
  nothing, which costs one extra call per camera. `--noop` measures it as
  condition N. Without it, `vs src` includes an unmeasured re-render tax and the
  report says so.

One run = one backend + model + prompt template, mirroring the video harnesses'
one-run-per-checkpoint layout. Compare runs with `analyze.py`.

    ../../.venv/bin/python run_experiment.py \
        --situations ../cosmos3/results/situations_multitraj \
        --backend openai_image --run-name openai_gpt_image_2 --ceiling 5

    # what would it cost, without calling anything:
    ../../.venv/bin/python run_experiment.py --situations ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))  # to import the Stage B package

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from pipeline.subgoal_image import prompts  # noqa: E402
from pipeline.subgoal_image.backends import (  # noqa: E402
    available_image_backends,
    build_image_backend,
    is_paid_backend,
)
from pipeline.subgoal_image.cache import BlobCache  # noqa: E402
from pipeline.subgoal_image.cost import CostTracker  # noqa: E402
from pipeline.subgoal_image.edit import edit_camera  # noqa: E402
from pipeline.subgoal_image.imaging import phash_delta  # noqa: E402

DEFAULT_SITUATIONS = HERE.parent / "cosmos3/results/situations_multitraj"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs" / "subgoal_images" / "cache"
CAMERAS = ("exterior_1", "exterior_2", "wrist")

#: Condition N — the editor's analogue of the video models' reconstruction
#: floor. Sent as a raw prompt rather than through a subgoal template, because
#: wrapping "change nothing" in "show the scene a few moments later" asks for two
#: contradictory things and would measure the contradiction instead of the tax.
NOOP_CONDITION = "N"
NOOP_PROMPT = (
    "Robot camera view. Reproduce this exact image unchanged: the same scene, the "
    "same objects in the same positions, the same robot arm pose, the same "
    "viewpoint and the same lighting. Do not add, remove, move or restyle anything."
)


def load_situations(sit_dir: Path, limit: int | None) -> tuple[dict, list[str]]:
    meta = json.loads((sit_dir / "meta.json").read_text())
    sits = meta["situations"]
    return meta, (sits[:limit] if limit else sits)


def conditions_for(sdir: Path, keep: list[str] | None, noop: bool) -> dict[str, list[str]]:
    """Instructions per condition for one situation, in report order.

    Falls back to condition A alone when a situation set was built without
    `prepare_instructions.py` having been run over it.
    """
    sit = json.loads((sdir / "situation.json").read_text())
    path = sdir / "conditions.json"
    conds: dict[str, list[str]] = (
        json.loads(path.read_text()) if path.exists() else {"A": [sit["instruction"]]}
    )
    conds = {k: v for k, v in conds.items() if v}
    if keep:
        conds = {k: v for k, v in conds.items() if k in keep}
    if noop:
        conds[NOOP_CONDITION] = [NOOP_PROMPT]
    return conds


def count_calls(sit_dir: Path, situations: list[str], cameras: list[str],
                keep: list[str] | None, noop: bool) -> int:
    total = 0
    for sid in situations:
        conds = conditions_for(sit_dir / sid, keep, noop)
        present = [c for c in cameras if (sit_dir / sid / "history" / f"{c}_0.png").exists()]
        total += sum(len(v) for v in conds.values()) * len(present)
    return total


def run(args) -> None:
    sit_dir = Path(args.situations).resolve()
    meta, situations = load_situations(sit_dir, args.limit_situations)
    cameras = [c for c in args.cameras if c in CAMERAS] or list(CAMERAS)
    horizon = int(meta.get("horizon") or 0)
    k = args.k if args.k is not None else horizon

    backend_kwargs: dict = {}
    if args.model:
        backend_kwargs["model"] = args.model
    if args.backend == "openai_image":
        backend_kwargs["quality"] = args.openai_quality
        backend_kwargs["size"] = args.openai_size
    backend = build_image_backend(args.backend, **backend_kwargs)
    tpl_name, tpl_text = prompts.resolve_template(args.prompt_template)

    # --- spend projection, before anything is called ------------------------ #
    n_calls = count_calls(sit_dir, situations, cameras, args.conditions, args.noop)
    per_call = backend.estimate_cost()
    projected = n_calls * per_call
    paid = is_paid_backend(args.backend)
    print(f"{args.backend} / {backend.model}: {len(situations)} situations x "
          f"{len(cameras)} cameras = {n_calls} edits")
    if paid:
        print(f"  projected spend <= ${projected:.2f} "
              f"(${per_call:.4f}/image upper bound; cached edits are free)")
    if args.dry_run:
        print("  --dry-run: nothing called.")
        return
    if paid and projected > args.ceiling:
        raise SystemExit(
            f"ABORT: projected ${projected:.2f} exceeds --ceiling ${args.ceiling:.2f}.\n"
            f"       Raise the ceiling, or cut the grid with --limit-situations / "
            f"--cameras / --conditions.\n"
            f"       Refusing up front rather than aborting mid-run, which would "
            f"leave a run that cannot be analysed."
        )

    run_dir = HERE / "results" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = BlobCache(Path(args.cache_dir))
    tracker = CostTracker(ceiling_usd=args.ceiling, costs_path=run_dir / "costs.jsonl")

    samples: list[dict] = []
    aborted = False
    for sid in situations:
        sdir = sit_dir / sid
        sit = json.loads((sdir / "situation.json").read_text())
        for cond, instructions in conditions_for(sdir, args.conditions, args.noop).items():
            for j, instruction in enumerate(instructions):
                is_noop = cond == NOOP_CONDITION
                prompt_text = NOOP_PROMPT if is_noop else prompts.build_prompt(tpl_text, instruction)
                sample_id = f"{sid}__{cond}{j}"
                out = run_dir / sample_id
                rec = {
                    "sample_id": sample_id, "situation_id": sid, "anchor": sit["anchor"],
                    "condition": cond, "instruction": instruction,
                    "prompt_template": None if is_noop else tpl_name,
                    "cameras": {},
                }

                for cam in cameras:
                    src = sdir / "history" / f"{cam}_0.png"
                    if not src.exists():
                        continue
                    src_bytes = src.read_bytes()
                    fut = sdir / "future" / f"{cam}_f{k:02d}.png"
                    fut_bytes = fut.read_bytes() if fut.exists() else None

                    t0 = time.time()
                    outcome = edit_camera(
                        backend_name=args.backend, backend=backend, camera=cam,
                        source_bytes=src_bytes, instruction=instruction,
                        prompt=prompt_text, future_bytes=fut_bytes, k=k,
                        cache=cache, tracker=tracker, use_cache=not args.no_cache,
                        example_id=sample_id,
                    )
                    dt = time.time() - t0

                    if outcome.over_budget:
                        print(f"BUDGET CEILING HIT: {outcome.error}")
                        aborted = True
                        break
                    if not outcome.ok:
                        rec["cameras"][cam] = {"image": None, "error": outcome.error}
                        print(f"  ! {sample_id} [{cam}]: {outcome.error}")
                        continue

                    out.mkdir(parents=True, exist_ok=True)
                    name = f"subgoal_{cam}.{outcome.ext}"
                    (out / name).write_bytes(outcome.image_bytes)
                    rec["cameras"][cam] = {
                        "image": name,
                        "cached": outcome.cached,
                        "cost_usd": round(outcome.cost_usd, 6),
                        "latency_s": round(dt, 2),
                        "phash_norm": phash_delta(src_bytes, outcome.image_bytes)["norm"],
                        "usage": outcome.meta.get("usage"),
                    }
                    flag = "cached" if outcome.cached else f"${outcome.cost_usd:.4f}"
                    print(f"  {sample_id} [{cam}] '{instruction[:38]}' -> {flag} {dt:.1f}s")

                if rec["cameras"]:
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "sample.json").write_text(json.dumps(rec, indent=2))
                    samples.append(rec)
                if aborted:
                    break
            if aborted:
                break
        if aborted:
            break

    index = {
        "situations_dir": str(sit_dir),
        "episode_id": meta.get("episode_id") or meta.get("layout", "multiple episodes"),
        "backend": args.backend,
        # Derived from the built backend, never hardcoded: this harness runs
        # against several models and a fixed string would mislabel every run but
        # the first (the mistake the cosmos3 harness made and had to fix).
        "model": backend.model,
        "prompt_template": tpl_name,
        "prompt_template_id": prompts.template_id(tpl_text),
        "prompt_template_text": tpl_text,
        "cameras": cameras,
        "k": k,
        "horizon": horizon,
        "fps": meta.get("fps"),
        "noop_floor": args.noop,
        "openai": ({"quality": args.openai_quality, "size": args.openai_size}
                   if args.backend == "openai_image" else None),
        "cost": tracker.summary(),
        "aborted_on_budget": aborted,
        "resample_null": None,   # filled in by resample_null.py
        "ts": datetime.now(timezone.utc).isoformat(),
        "samples": samples,
    }
    (run_dir / "index.json").write_text(json.dumps(index, indent=2))

    print(f"\nWrote {len(samples)} samples to {run_dir}")
    if aborted:
        print("** RUN ABORTED ON BUDGET CEILING — partial results written **")
    print(f"cost: {json.dumps(tracker.summary())}")
    print(f"next: resample_null.py --run {args.run_name}   "
          f"(the control every sensitivity number is read against)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--situations", default=str(DEFAULT_SITUATIONS),
                   help="Situation set built by cosmos3/dreamzero prepare_situations.py.")
    p.add_argument("--backend", default="gemini_image",
                   help=f"One image backend. Available: {', '.join(available_image_backends())}.")
    p.add_argument("--model", default=None, help="Model override (default: backend's default).")
    p.add_argument("--run-name", default="runs")
    p.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    p.add_argument("--conditions", nargs="+", default=None,
                   help="Subset of the situation set's conditions (default: all).")
    p.add_argument("--limit-situations", type=int, default=None)
    p.add_argument("--prompt-template", default=prompts.DEFAULT_TEMPLATE,
                   help=f"Stage B template name or literal. Registered: {', '.join(prompts.TEMPLATES)}.")
    p.add_argument("--k", type=int, default=None,
                   help="Future frame index used as the real yardstick "
                        "(default: the situation set's horizon).")
    p.add_argument("--noop", action="store_true",
                   help="Also run condition N (a 'change nothing' edit) to measure the "
                        "model's re-render tax — the analogue of the video models' "
                        "reconstruction floor. Costs one extra call per camera.")
    p.add_argument("--ceiling", type=float, default=5.0,
                   help="Hard $ ceiling. The run refuses to start if the projection exceeds it.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the call count and projected spend, then exit.")
    p.add_argument("--openai-quality", default="low", choices=["low", "medium", "high", "auto"])
    p.add_argument("--openai-size", default="auto")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                   help="Stage B edit cache, shared with the pipeline ($0 reruns).")
    p.add_argument("--no-cache", action="store_true",
                   help="Never read or write the edit cache (always re-call, always pay).")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
