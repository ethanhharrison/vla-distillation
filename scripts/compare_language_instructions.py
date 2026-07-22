"""Side-by-side comparison of multiple instruction-generation runs.

Given two or more run files produced by `pipeline.language_instruction.generate`
(typically the same record generated with different VLMs), this renders a single
HTML report that aligns the runs by trajectory step: each step shows the camera
frame(s) once, followed by one column per run listing that run's instructions.

Usage:
    uv run python scripts/compare_language_instructions.py <run1.txt> <run2.txt> [...] [--open]

Each argument may be a path to a generated `.txt` run file or a record name to
look up in outputs/language_instructions/.
"""

from __future__ import annotations

import argparse
import html
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from summarize_language_instructions import (  # noqa: E402
    DEFAULT_VIZ_DIR,
    _embed_image,
    parse_run,
    resolve_run_file,
)


def _load_runs(targets: list[str]) -> list[dict]:
    runs = []
    for target in targets:
        run_file = resolve_run_file(target)
        run = parse_run(run_file)
        run["source"] = run_file
        run["label"] = run["info"].get("model") or run_file.stem
        runs.append(run)
    return runs


def _step_index(run: dict) -> dict[int, dict]:
    return {step["step"]: step for step in run["steps"]}


def render_html(runs: list[dict]) -> str:
    base = runs[0]
    record = base["info"].get("record", "")
    original_items = "".join(f"<li>{html.escape(o)}</li>" for o in base["original"]) or (
        "<li><em>none recorded</em></li>"
    )

    indexed = [_step_index(run) for run in runs]
    all_steps = sorted({step["step"] for run in runs for step in run["steps"]})

    labels = [run["label"] for run in runs]
    header_cells = "".join(f"<div class='mhead'>{html.escape(l)}</div>" for l in labels)

    step_blocks = []
    for step_num in all_steps:
        # Images are identical across runs of the same record; use the first
        # run that has them for this step.
        images_html = ""
        for run_steps in indexed:
            step = run_steps.get(step_num)
            if step and step["images"]:
                images_html = "".join(
                    f"<figure>{_embed_image(p)}<figcaption>{html.escape(cam)}</figcaption></figure>"
                    for cam, p in step["images"].items()
                )
                break

        columns = []
        for run_steps in indexed:
            step = run_steps.get(step_num)
            if step and step["instructions"]:
                items = "".join(
                    f"<li>{html.escape(t)}</li>" for t in step["instructions"]
                )
            else:
                items = "<li><em>no instructions</em></li>"
            columns.append(f"<ol class='instructions'>{items}</ol>")
        columns_html = "".join(f"<div class='mcol'>{c}</div>" for c in columns)

        step_blocks.append(
            f"""
            <section class="step">
              <h3>Step {step_num}</h3>
              <div class="frames">{images_html or '<em>no images saved</em>'}</div>
              <div class="grid" style="--cols: {len(runs)}">
                <div class="mhead-row">{header_cells}</div>
                {columns_html}
              </div>
            </section>
            """
        )

    models_table = "".join(
        f"<tr><th>{html.escape(run['label'])}</th>"
        f"<td>{html.escape(run['info'].get('provider', '?'))}</td>"
        f"<td>{html.escape(str(run['source']))}</td></tr>"
        for run in runs
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Model comparison — {html.escape(record)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
         max-width: 1400px; padding: 24px; line-height: 1.5; }}
  h1 {{ margin-bottom: 4px; }}
  .subtitle {{ color: #888; margin-top: 0; }}
  .card {{ border: 1px solid #8883; border-radius: 10px; padding: 16px 20px; margin: 16px 0; }}
  table {{ border-collapse: collapse; }}
  th {{ text-align: left; padding-right: 16px; vertical-align: top; color: #888; }}
  td {{ padding: 2px 12px 2px 0; }}
  .original li {{ font-weight: 600; }}
  .step {{ border-top: 1px solid #8883; padding-top: 12px; margin-top: 24px; }}
  .frames {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
  figure {{ margin: 0; }}
  figure img {{ width: 240px; height: auto; border-radius: 6px; display: block; }}
  figcaption {{ font-size: 12px; color: #888; text-align: center; margin-top: 4px; }}
  .grid {{ display: grid; grid-template-columns: repeat(var(--cols), 1fr); gap: 12px; }}
  .mhead-row {{ display: contents; }}
  .mhead {{ font-weight: 700; padding: 6px 10px; background: #8882; border-radius: 6px;
           position: sticky; top: 0; }}
  .mcol {{ border: 1px solid #8883; border-radius: 8px; padding: 8px 10px; }}
  .instructions {{ margin: 0; padding-left: 20px; }}
  .instructions li {{ margin: 4px 0; }}
  .missing {{ width: 240px; height: 135px; display: flex; align-items: center;
             justify-content: center; background: #8881; border-radius: 6px;
             font-size: 12px; color: #c33; text-align: center; }}
</style>
</head>
<body>
  <h1>VLM comparison</h1>
  <p class="subtitle">{html.escape(record)} &mdash; {len(runs)} models, {len(all_steps)} steps</p>

  <div class="card">
    <h2>Models compared</h2>
    <table>{models_table}</table>
  </div>

  <div class="card">
    <h2>Original task instructions</h2>
    <ul class="original">{original_items}</ul>
  </div>

  {''.join(step_blocks)}

  <footer class="subtitle">Rendered {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</footer>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        help="Two or more run .txt paths (or record names) to compare.",
    )
    parser.add_argument("--output", default=None, help="Output .html path.")
    parser.add_argument(
        "--open", action="store_true", help="Open the report in the default browser."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if len(args.runs) < 2:
        raise SystemExit("Provide at least two runs to compare.")

    runs = _load_runs(args.runs)

    if args.output:
        output_path = Path(args.output)
    else:
        record_stem = Path(runs[0]["info"].get("record", "comparison")).stem
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = DEFAULT_VIZ_DIR / f"{record_stem}_comparison_{timestamp}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(runs))

    print(f"Compared {len(runs)} runs: {', '.join(r['label'] for r in runs)}")
    print(f"Wrote comparison to {output_path}")
    if args.open:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
