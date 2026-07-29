"""Write per-situation A/B/C instructions for the DreamZero experiment.

Runs in the MAIN vla-distillation venv (has google-genai). Reuses our Stage A
VLM backend (`pipeline.language_instruction.vlm`) to propose counterfactual
instructions from the anchor frame. Writes conditions.json into each situation
dir, read later by run_experiment.py.

  A = the episode's original instruction (sanity / calibration)
  B = counterfactual: NEW plausible instructions proposed by the VLM (Stage A)
  C = null/stress: hardcoded impossible / irrelevant instructions

Spend is tiny (a few Gemini-flash calls); ceiling guard via --max-situations.

    python explorations/dreamzero/prepare_instructions.py \
        --situations explorations/dreamzero/results/situations/<episode>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# NOTE: we call google-genai directly rather than reusing
# pipeline.language_instruction.vlm, because that module now hard-imports torch
# (partner's local-HF backend) which isn't installed in this venv — and this
# session must not touch pipeline/. This is the same Gemini call it would make.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


def gemini_generate(prompt: str, images: list[bytes], model: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
    parts = [types.Part.from_text(text=prompt)]
    parts += [types.Part.from_bytes(data=img, mime_type="image/png") for img in images]
    resp = client.models.generate_content(model=model, contents=parts)
    return resp.text or ""

# Deliberately impossible/irrelevant for a DROID tabletop scene (null/stress).
NULL_INSTRUCTIONS = ["fold the laundry", "pour a glass of orange juice"]

B_PROMPT = (
    "These are camera views of a robot arm at one moment during a tabletop "
    "manipulation task. Propose {n} DISTINCT, short imperative instructions for "
    "NEW tasks the robot could plausibly begin from this exact scene, using only "
    "objects visible here. They must be different from the original task: "
    "\"{original}\". Output one instruction per line, no numbering or extra text."
)


def parse_lines(text: str, n: int) -> list[str]:
    out = []
    for line in text.splitlines():
        s = line.strip().lstrip("-*0123456789. ").strip().strip('"').strip()
        if s:
            out.append(s)
    return out[:n]


def run(args):
    sit_dir = Path(args.situations)
    meta = json.loads((sit_dir / "meta.json").read_text())
    sits = meta["situations"][: args.max_situations]
    model = args.model or DEFAULT_GEMINI_MODEL

    for sid in sits:
        sdir = sit_dir / sid
        sit = json.loads((sdir / "situation.json").read_text())
        original = sit.get("instruction", "")
        # anchor frame (offset 0) for the two exterior cameras
        imgs = []
        for cam in ("exterior_1", "exterior_2"):
            p = sdir / "history" / f"{cam}_0.png"
            if p.exists():
                imgs.append(p.read_bytes())

        b_instr = []
        if not args.no_spend and imgs:
            raw = gemini_generate(B_PROMPT.format(n=args.num_b, original=original), imgs, model)
            b_instr = parse_lines(raw, args.num_b)

        conditions = {
            "A": [original] if original else [],
            "B": b_instr,
            "C": NULL_INSTRUCTIONS[: args.num_c],
        }
        (sdir / "conditions.json").write_text(json.dumps(conditions, indent=2))
        print(f"  {sid}: A={conditions['A']}  B={b_instr}  C={conditions['C']}")

    print(f"Wrote conditions.json for {len(sits)} situations.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--situations", required=True)
    p.add_argument("--model", default=None, help="Gemini model (default: provider default).")
    p.add_argument("--num-b", type=int, default=2, help="Counterfactual instructions per situation.")
    p.add_argument("--num-c", type=int, default=2, help="Null/stress instructions per situation.")
    p.add_argument("--max-situations", type=int, default=5, help="Spend guard.")
    p.add_argument("--no-spend", action="store_true", help="Skip VLM (B empty); A+C only.")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
