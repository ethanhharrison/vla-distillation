"""Rebuild a run's index.json from the per-sample sample.json files.

run_experiment.py writes index.json only at the very end, so a crash after the
generations are already on disk loses the manifest but not the (expensive) work.
This reconstructs it in place rather than re-running the model.

    .venv/bin/python rebuild_index.py --run runs_multitraj \\
        --situations results/situations_multitraj --chunk-size 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--situations", required=True)
    p.add_argument("--model-path", default="/home/tiger/proj/staging/vla/models/Cosmos3-Nano-Policy-DROID")
    p.add_argument("--cameras", nargs="+", default=["exterior_1", "exterior_2", "wrist"])
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--resolution-tier", type=int, default=256)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    run_dir = HERE / "results" / args.run
    sit_dir = Path(args.situations)
    if not sit_dir.is_absolute():
        sit_dir = HERE / sit_dir
    meta = json.loads((sit_dir / "meta.json").read_text())

    samples = []
    for sd in sorted(run_dir.iterdir()):
        f = sd / "sample.json"
        if sd.is_dir() and f.is_file():
            samples.append(json.loads(f.read_text()))

    index = {
        "situations_dir": str(sit_dir),
        "episode_id": meta.get("episode_id") or meta.get("layout", "multiple episodes"),
        "model_path": args.model_path, "model": "Cosmos3-Nano-Policy-DROID",
        "action_mode": "policy", "domain_name": "droid_lerobot",
        "action_space": "10D EEF pose delta (3D translation + 6D rotation) + gripper, model-normalized",
        "cameras": args.cameras, "fps": args.fps, "chunk_size": args.chunk_size,
        "resolution_tier": args.resolution_tier, "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale, "seed": args.seed,
        "guardrails": False, "prompt_upsampling": False,
        "rebuilt_from_samples": True,
        "samples": samples,
    }
    (run_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"rebuilt {run_dir}/index.json from {len(samples)} sample.json files")


if __name__ == "__main__":
    main()
