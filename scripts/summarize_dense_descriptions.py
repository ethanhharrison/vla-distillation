"""Visualize dense change-of-state descriptions for trajectory clips.

Renders a self-contained HTML report pairing each clip's start/end camera
frames with the VLM dense description.

    uv run python scripts/summarize_dense_descriptions.py \
        outputs/dense_descriptions/success-00285_gemini_20260821-134039.txt \
        --open
"""

from __future__ import annotations

import argparse
import base64
import html
import re
import webbrowser
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "outputs" / "dense_descriptions"
DEFAULT_VIZ_DIR = PROJECT_ROOT / "outputs" / "visualizations"

CLIP_HEADER = re.compile(
    r"^\[clip\s+(\d+)\]\s+steps\s+(\d+)\s*-\s*(\d+)\s*$", re.IGNORECASE
)


def resolve_run_file(target: str) -> Path:
    """Accept a direct .txt path or a record/stem name under RUNS_DIR."""
    path = Path(target)
    if path.is_file():
        return path
    matches = sorted(
        RUNS_DIR.glob(f"*{target}*.txt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not matches:
        raise FileNotFoundError(
            f"No dense-description run found for {target!r}. Pass a .txt path "
            f"or a record name present in {RUNS_DIR}."
        )
    return matches[0]


def parse_run(path: Path) -> dict:
    """Parse a dense-description .txt into {info, metadata, clips}."""
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

    clips: list[dict] = []
    current: dict | None = None
    for line in body:
        stripped = line.strip()
        match = CLIP_HEADER.match(stripped)
        if match:
            if current is not None:
                clips.append(current)
            current = {
                "clip_index": int(match.group(1)),
                "start_step": int(match.group(2)),
                "end_step": int(match.group(3)),
                "language": "",
                "description": "",
                "start_images": {},
                "end_images": {},
            }
            continue
        if current is None or not stripped:
            continue
        if stripped.startswith("language:"):
            current["language"] = stripped[len("language:") :].strip()
        elif stripped.startswith("description:"):
            current["description"] = stripped[len("description:") :].strip()
        elif stripped.startswith("(image) "):
            rest = stripped[len("(image) ") :]
            label, _, image_path = rest.partition(": ")
            role, _, camera = label.partition("/")
            if role == "start":
                current["start_images"][camera or label] = image_path
            elif role == "end":
                current["end_images"][camera or label] = image_path
            else:
                current["start_images"][label] = image_path
    if current is not None:
        clips.append(current)

    return {"info": info, "metadata": metadata, "clips": clips}


def _embed_image(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        return f'<div class="missing">missing: {html.escape(image_path)}</div>'
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/jpeg;base64,{encoded}" alt="{html.escape(path.name)}">'


def _fmt_cost(value: str | None) -> str:
    if value is None:
        return "unknown"
    try:
        return f"${float(value):.6f}"
    except ValueError:
        return html.escape(value)


def cost_card(info: dict) -> str:
    per_clip = info.get("estimated_cost_per_clip_usd")
    total = info.get("estimated_cost_total_usd")
    if per_clip is None and total is None:
        return ""
    rows = [
        ("Cost per clip", _fmt_cost(per_clip)),
        ("Total cost", _fmt_cost(total)),
    ]
    if "generation_cost_usd" in info:
        rows.append(("Generation", _fmt_cost(info.get("generation_cost_usd"))))
    body = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{value}</td></tr>" for label, value in rows
    )
    return f"""
  <div class="card">
    <h2>Estimated cost</h2>
    <table>{body}</table>
    <p class="subtitle">Per-clip = total cost / number of clips. Approximate; based on recorded token usage.</p>
  </div>"""


def _frame_row(label: str, images: dict[str, str]) -> str:
    if not images:
        return f'<div class="frame-row"><h4>{html.escape(label)}</h4><em>no images saved</em></div>'
    figures = "".join(
        f"<figure>{_embed_image(path)}<figcaption>{html.escape(cam)}</figcaption></figure>"
        for cam, path in images.items()
    )
    return (
        f'<div class="frame-row"><h4>{html.escape(label)}</h4>'
        f'<div class="frames">{figures}</div></div>'
    )


def render_html(run: dict, source: Path) -> str:
    info = run["info"]
    provider = info.get("provider", "?")
    model = info.get("model", "?")
    language = info.get("language_instruction", "")

    meta_rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in info.items()
    )
    episode_rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in run["metadata"].items()
    )
    episode_card = (
        f"""
  <div class="card">
    <h2>Episode metadata</h2>
    <table>{episode_rows}</table>
  </div>"""
        if run["metadata"]
        else ""
    )

    clip_sections = []
    for clip in run["clips"]:
        clip_language = clip.get("language") or language
        clip_sections.append(
            f"""
            <section class="clip">
              <h3>Clip {clip['clip_index']}
                <span class="subtitle">steps {clip['start_step']}–{clip['end_step']}</span>
              </h3>
              <p class="language"><strong>Language:</strong> {html.escape(clip_language)}</p>
              <div class="compare">
                {_frame_row("Start", clip.get("start_images") or {})}
                {_frame_row("End", clip.get("end_images") or {})}
              </div>
              <div class="description">
                <h4>Dense description</h4>
                <p>{html.escape(clip.get("description") or "(empty)")}</p>
              </div>
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dense descriptions — {html.escape(info.get('record', source.name))}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
         max-width: 1100px; padding: 24px; line-height: 1.5; }}
  h1 {{ margin-bottom: 4px; }}
  .subtitle {{ color: #888; margin-top: 0; font-weight: 400; font-size: 0.9em; }}
  .card {{ border: 1px solid #8883; border-radius: 10px; padding: 16px 20px;
          margin: 16px 0; }}
  table {{ border-collapse: collapse; }}
  th {{ text-align: left; padding-right: 16px; vertical-align: top;
       color: #888; font-weight: 600; }}
  td {{ padding: 2px 0; }}
  .language {{ margin: 8px 0 12px; }}
  .clip {{ border-top: 1px solid #8883; padding-top: 12px; margin-top: 28px; }}
  .compare {{ display: grid; gap: 16px; }}
  .frame-row h4 {{ margin: 0 0 6px; color: #555; font-size: 13px;
                  text-transform: uppercase; letter-spacing: 0.04em; }}
  .frames {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  figure {{ margin: 0; }}
  figure img {{ width: 240px; height: auto; border-radius: 6px; display: block; }}
  figcaption {{ font-size: 12px; color: #888; text-align: center; margin-top: 4px; }}
  .description {{ margin-top: 14px; padding: 12px 14px; background: #8881;
                 border-radius: 8px; }}
  .description h4 {{ margin: 0 0 6px; font-size: 13px; color: #555;
                    text-transform: uppercase; letter-spacing: 0.04em; }}
  .description p {{ margin: 0; }}
  .missing {{ width: 240px; height: 135px; display: flex; align-items: center;
             justify-content: center; background: #8881; border-radius: 6px;
             font-size: 12px; color: #c33; padding: 8px; text-align: center; }}
  @media (min-width: 900px) {{
    .compare {{ grid-template-columns: 1fr 1fr; }}
  }}
</style>
</head>
<body>
  <h1>Dense descriptions</h1>
  <p class="subtitle">{html.escape(info.get('record', ''))} &mdash;
     <strong>{html.escape(provider)}</strong> / {html.escape(model)}</p>

  <div class="card">
    <h2>Run configuration</h2>
    <table>{meta_rows}</table>
  </div>
{cost_card(info)}
{episode_card}
  <div class="card">
    <h2>Language instruction</h2>
    <p>{html.escape(language) if language else "<em>none recorded</em>"}</p>
  </div>

  <h2>Per-clip descriptions ({len(run['clips'])} clips)</h2>
  {''.join(clip_sections)}

  <footer class="subtitle">
    Rendered from {html.escape(str(source))} on
    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </footer>
</body>
</html>
"""  # noqa: DTZ005


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run",
        help="Path to a dense-description .txt run file, or a record name to look up.",
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
        output_path = DEFAULT_VIZ_DIR / f"dense_{run_file.stem}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(run, run_file))

    print(f"Summarized {len(run['clips'])} clips from {run_file}")
    print(f"Wrote report to {output_path}")
    if args.open:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
