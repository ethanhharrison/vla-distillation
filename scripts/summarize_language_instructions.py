"""Visualize VLM-generated language instructions for a record's trajectory.

Reads an output file produced by `pipeline.language_instruction.generate` and
renders a self-contained HTML report showing, at every queried step, the camera
frame(s) alongside the instructions the VLM proposed. The report header also
lists the task's original instructions, the VLM used, and the system prompt.

Usage:
    uv run python scripts/summarize_language_instructions.py <run.txt | record-name> [--open]

You can pass either the path to a generated `.txt` run file, or just a record
name (e.g. `success-00188`), in which case the most recent matching run in
outputs/language_instructions/ is used.
"""

from __future__ import annotations

import argparse
import base64
import html
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.language_instruction.prompts import (  # noqa: E402
    INSTRUCTION_PROMPT,
    build_prompt,
)

RUNS_DIR = PROJECT_ROOT / "outputs" / "language_instructions"
DEFAULT_VIZ_DIR = PROJECT_ROOT / "outputs" / "visualizations"

ORIGINAL_INSTRUCTION_KEYS = (
    "language_instruction1",
    "language_instruction2",
    "language_instruction3",
)


def _split_score(text: str) -> tuple[str, str | None]:
    """Split an instruction line into its text and optional ' | score: N' tag."""
    instruction, sep, score = text.partition(" | score: ")
    if sep:
        return instruction.strip(), score.strip()
    return instruction.strip(), None


def resolve_run_file(target: str) -> Path:
    """Accept a direct .txt path or a record name to look up in RUNS_DIR."""
    path = Path(target)
    if path.is_file():
        return path
    matches = sorted(
        RUNS_DIR.glob(f"*{target}*.txt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not matches:
        raise FileNotFoundError(
            f"No run file found for {target!r}. Pass a .txt path or a record name "
            f"present in {RUNS_DIR}."
        )
    return matches[0]


def parse_run(path: Path) -> dict:
    """Parse a generated run .txt into a structured dict."""
    lines = path.read_text().splitlines()
    separator = next(
        (i for i, line in enumerate(lines) if set(line) == {"="} and len(line) >= 10),
        len(lines),
    )
    header, body = lines[:separator], lines[separator + 1 :]

    info: dict[str, str] = {}
    metadata: dict[str, str] = {}
    in_metadata = False
    for line in header:
        if not line.strip():
            continue
        if line.startswith("metadata:"):
            in_metadata = True
            continue
        key, _, value = line.strip().partition(": ")
        if in_metadata and line.startswith("  "):
            metadata[key] = value
        else:
            info[key] = value

    steps: list[dict] = []
    current: dict | None = None
    for line in body:
        stripped = line.strip()
        if stripped.startswith("[step ") and stripped.endswith("]"):
            if current is not None:
                steps.append(current)
            current = {
                "step": int(stripped[len("[step ") : -1]),
                "instructions": [],
                "scores": [],
                "rejected": [],
                "images": {},
            }
        elif stripped.startswith("- ") and current is not None:
            text, score = _split_score(stripped[2:])
            current["instructions"].append(text)
            current["scores"].append(score)
        elif stripped.startswith("(rejected) ") and current is not None:
            text, score = _split_score(stripped[len("(rejected) ") :])
            current["rejected"].append({"text": text, "score": score})
        elif stripped.startswith("(image) ") and current is not None:
            camera, _, image_path = stripped[len("(image) ") :].partition(": ")
            current["images"][camera] = image_path
    if current is not None:
        steps.append(current)

    original = []
    for key in ORIGINAL_INSTRUCTION_KEYS:
        value = metadata.get(key)
        if value and value not in original:
            original.append(value)

    return {"info": info, "metadata": metadata, "original": original, "steps": steps}


def _embed_image(image_path: str) -> str:
    """Return an <img> tag with the JPEG embedded as base64, or a placeholder."""
    path = Path(image_path)
    if not path.is_file():
        return f'<div class="missing">missing: {html.escape(image_path)}</div>'
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/jpeg;base64,{encoded}" alt="{html.escape(path.name)}">'


def _score_badge(score: str | None, rejected: bool = False) -> str:
    """Render a small score badge, or '' when there is no score."""
    if score is None:
        return ""
    cls = "badge rejected-badge" if rejected else "badge"
    return f' <span class="{cls}">score {html.escape(str(score))}</span>'


def render_html(run: dict, source: Path) -> str:
    info = run["info"]
    provider = info.get("provider", "?")
    model = info.get("model", "?")

    original_items = "".join(f"<li>{html.escape(o)}</li>" for o in run["original"]) or (
        "<li><em>none recorded</em></li>"
    )

    try:
        total = int(info.get("trajectory_length", 0))
    except ValueError:
        total = 0
    try:
        num_instructions = int(info.get("num_instructions", len(run["original"]) or 3))
    except ValueError:
        num_instructions = 3

    step_sections = []
    previous_instructions: list[str] = []
    for step in run["steps"]:
        images = "".join(
            f'<figure>{_embed_image(p)}<figcaption>{html.escape(cam)}</figcaption></figure>'
            for cam, p in step["images"].items()
        )
        scores = step.get("scores") or [None] * len(step["instructions"])
        instructions = "".join(
            f"<li>{html.escape(text)}{_score_badge(score)}</li>"
            for text, score in zip(step["instructions"], scores)
        )
        rejected_items = "".join(
            f'<li>{html.escape(r["text"])}{_score_badge(r["score"], rejected=True)}</li>'
            for r in step.get("rejected", [])
        )
        rejected_block = (
            f'<details class="rejected"><summary>Rejected by judge '
            f'({len(step["rejected"])})</summary><ul>{rejected_items}</ul></details>'
            if step.get("rejected")
            else ""
        )

        # Reconstruct the exact system prompt sent at this step: the avoid list
        # grows as it accumulates instructions suggested at earlier steps.
        step_prompt = build_prompt(
            step=step["step"],
            total=total,
            num_instructions=num_instructions,
            original_instructions=run["original"],
            previous_instructions=previous_instructions,
        )
        step_sections.append(
            f"""
            <section class="step">
              <h3>Step {step['step']}</h3>
              <div class="frames">{images or '<em>no images saved</em>'}</div>
              <ol class="instructions">{instructions}</ol>
              {rejected_block}
              <details class="prompt">
                <summary>System prompt used at this step
                  ({len(previous_instructions)} prior instruction(s) to avoid)</summary>
                <pre>{html.escape(step_prompt)}</pre>
              </details>
            </section>
            """
        )

        for text in step["instructions"]:
            if text not in previous_instructions:
                previous_instructions.append(text)

    meta_rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in info.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Language instructions — {html.escape(info.get('record', source.name))}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
         max-width: 1100px; padding: 24px; line-height: 1.5; }}
  h1 {{ margin-bottom: 4px; }}
  .subtitle {{ color: #888; margin-top: 0; }}
  .card {{ border: 1px solid #8883; border-radius: 10px; padding: 16px 20px;
          margin: 16px 0; }}
  table {{ border-collapse: collapse; }}
  th {{ text-align: left; padding-right: 16px; vertical-align: top;
       color: #888; font-weight: 600; }}
  td {{ padding: 2px 0; }}
  details pre {{ white-space: pre-wrap; background: #8881; padding: 12px;
                border-radius: 8px; }}
  .original li {{ font-weight: 600; }}
  .step {{ border-top: 1px solid #8883; padding-top: 12px; margin-top: 24px; }}
  .frames {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  figure {{ margin: 0; }}
  figure img {{ width: 260px; height: auto; border-radius: 6px; display: block; }}
  figcaption {{ font-size: 12px; color: #888; text-align: center; margin-top: 4px; }}
  .instructions li {{ margin: 4px 0; }}
  .badge {{ display: inline-block; font-size: 11px; font-weight: 600;
           background: #2e7d3222; color: #2e7d32; border-radius: 6px;
           padding: 1px 6px; margin-left: 6px; vertical-align: middle; }}
  .rejected-badge {{ background: #c6282822; color: #c62828; }}
  details.rejected {{ margin: 6px 0 8px; }}
  details.rejected summary {{ cursor: pointer; color: #c62828; font-size: 13px; }}
  details.rejected li {{ color: #888; text-decoration: line-through; }}
  details.prompt {{ margin-top: 8px; }}
  details.prompt summary {{ cursor: pointer; color: #888; font-size: 13px; }}
  .missing {{ width: 260px; height: 146px; display: flex; align-items: center;
             justify-content: center; background: #8881; border-radius: 6px;
             font-size: 12px; color: #c33; padding: 8px; text-align: center; }}
</style>
</head>
<body>
  <h1>Generated language instructions</h1>
  <p class="subtitle">{html.escape(info.get('record', ''))} &mdash;
     <strong>{html.escape(provider)}</strong> / {html.escape(model)}</p>

  <div class="card">
    <h2>Run configuration</h2>
    <table>{meta_rows}</table>
  </div>

  <div class="card">
    <h2>Original task instructions</h2>
    <ul class="original">{original_items}</ul>
  </div>

  <div class="card">
    <h2>System prompt template</h2>
    <details>
      <summary>Base template (per-step values and the growing avoid list are
        substituted in — see each step below for the exact prompt used)</summary>
      <pre>{html.escape(INSTRUCTION_PROMPT)}</pre>
    </details>
  </div>

  <h2>Per-step instructions ({len(run['steps'])} steps)</h2>
  {''.join(step_sections)}

  <footer class="subtitle">
    Rendered from {html.escape(str(source))} on
    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </footer>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run",
        help="Path to a generated .txt run file, or a record name to look up.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .html path (defaults to outputs/visualizations/).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the rendered report in the default browser.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_file = resolve_run_file(args.run)
    run = parse_run(run_file)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = DEFAULT_VIZ_DIR / f"{run_file.stem}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(run, run_file))

    print(f"Summarized {len(run['steps'])} steps from {run_file}")
    print(f"Wrote report to {output_path}")
    if args.open:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
