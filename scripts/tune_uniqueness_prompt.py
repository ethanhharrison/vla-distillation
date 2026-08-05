"""Compare duplicate definitions on the same instructions, side by side.

Manual-testing harness for the thing most worth tuning: what we tell the judge a
duplicate *is*. One clustering call per definition, so comparing all four costs
about a cent.

    .venv/bin/python scripts/tune_uniqueness_prompt.py --frames-from <run_dir>
    .venv/bin/python scripts/tune_uniqueness_prompt.py \
        --instructions tests/fixtures/long_horizon_25.txt \
        --definitions end_state same_actions same_actions_lenient

Instructions come from a plain text file (one per line, `#` comments ignored) or
from a run's `.txt` output via `--from-run`. Frames are optional but strongly
recommended: without them the judge cannot tell whether "the black mug" and "the
black cup" are one object.

Output is per-definition groups plus an agreement table, because the interesting
signal is not how many merges each found but *which* — a definition that merges
more is only better if the extra merges are right.
"""

from __future__ import annotations

import argparse
import base64
import html
import itertools
import re
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from pipeline.language_instruction.uniqueness import (  # noqa: E402
    DUPLICATE_DEFINITIONS,
    Clustering,
    build_uniqueness_judge,
    cluster_instructions,
    resolve_definition,
)

DEFAULT_VIZ_DIR = PROJECT_ROOT / "outputs" / "visualizations"

DEFAULT_INSTRUCTIONS = PROJECT_ROOT / "tests" / "fixtures" / "long_horizon_25.txt"
CAMERA_ORDER = ("shoulder_image_1", "shoulder_image_2", "wrist_image")


EXPECT_LINE = re.compile(r"#\s*EXPECT\s+(DUPLICATE|DISTINCT)\s*:\s*(.+?)\s*\|\|\s*(.+?)\s*$", re.I)


def read_instructions(path: Path) -> list[str]:
    """One instruction per line; blank lines and `#` comments ignored."""
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def read_expectations(path: Path, instructions: list[str]) -> dict[tuple[int, int], str]:
    """Parse `# EXPECT DUPLICATE/DISTINCT: <a> || <b>` lines into index pairs.

    These are hand labels for the handful of pairs worth being sure about — the
    ones a definition must merge, and the traps it must not. They are scored, and
    never go anywhere near a prompt. An EXPECT naming an instruction that is not
    in the set is a typo, so it raises rather than being skipped.
    """
    index_of = {text: i for i, text in enumerate(instructions)}
    expectations: dict[tuple[int, int], str] = {}
    for line in path.read_text().splitlines():
        match = EXPECT_LINE.match(line.strip())
        if not match:
            continue
        verdict, left, right = match.group(1).upper(), match.group(2), match.group(3)
        for text in (left, right):
            if text not in index_of:
                raise SystemExit(f"EXPECT line names an instruction not in {path.name}: {text!r}")
        expectations[tuple(sorted((index_of[left], index_of[right])))] = verdict
    return expectations


def score(expectations: dict[tuple[int, int], str], merged: set[tuple[int, int]]) -> dict[str, int]:
    """How a definition did on the labelled pairs: caught, missed, false merges."""
    wanted = {pair for pair, verdict in expectations.items() if verdict == "DUPLICATE"}
    traps = {pair for pair, verdict in expectations.items() if verdict == "DISTINCT"}
    return {
        "caught": len(wanted & merged),
        "missed": len(wanted - merged),
        "false_merges": len(traps & merged),
        "traps_held": len(traps - merged),
    }


def read_run_instructions(path: Path) -> list[str]:
    """Pull the accepted instructions out of a generate/pipeline `.txt` run file."""
    instructions: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            instructions.append(stripped[2:].split(" | ")[0].strip())
    return instructions


def find_frames(run_dir: Path, step: int) -> list[bytes]:
    """Read a step's saved camera frames, in a stable camera order."""
    frames: list[bytes] = []
    for camera in CAMERA_ORDER:
        matches = sorted(run_dir.rglob(f"step{step:04d}_{camera}.jpeg"))
        if matches:
            frames.append(matches[0].read_bytes())
    return frames


def pairs_of(clustering: Clustering) -> set[tuple[int, int]]:
    """The index pairs this clustering merged, for comparing two clusterings."""
    return {
        pair
        for cluster in clustering.clusters
        for pair in itertools.combinations(sorted(cluster), 2)
    }


def report(instructions: list[str], results: dict[str, Clustering]) -> None:
    names = list(results)
    all_pairs = {name: pairs_of(clustering) for name, clustering in results.items()}
    shared = set.intersection(*all_pairs.values()) if all_pairs else set()

    for name, clustering in results.items():
        print(f"\n{'=' * 78}\n{name}: {len(instructions)} -> {len(clustering.clusters)} "
              f"({clustering.num_duplicates} folded)\n{'=' * 78}")
        groups = [c for c in clustering.clusters if len(c) > 1]
        if not groups:
            print("  (no merges)")
        for cluster in groups:
            solo = "" if pairs_of(Clustering([cluster])) <= shared else "   *"
            print(f"  KEEP  {instructions[cluster[0]]}{solo}")
            for member in cluster[1:]:
                print(f"   fold   {instructions[member]}")

    print(f"\n{'=' * 78}\nagreement  (* above = not merged by every definition)\n{'=' * 78}")
    print(f"  merged by all {len(names)}: {len(shared)} pairs")
    for name in names:
        only = all_pairs[name] - set.union(*(all_pairs[n] for n in names if n != name))
        print(f"  only {name}: {len(only)} pairs")
        for a, b in sorted(only):
            print(f"      {instructions[a]}\n        + {instructions[b]}")


def render_html(
    instructions: list[str],
    results: dict[str, Clustering],
    frames: list[bytes],
    model: str,
    expectations: dict[tuple[int, int], str] | None = None,
    source: str = "",
) -> str:
    """A side-by-side page: what each definition merged, and where they disagree.

    The per-instruction table is the debugging view — one row per instruction, one
    column per definition, so a disagreement is a row that is not all one colour.
    """
    names = list(results)
    expectations = expectations or {}
    all_pairs = {name: pairs_of(clustering) for name, clustering in results.items()}
    shared = set.intersection(*all_pairs.values()) if all_pairs else set()
    scores = {name: score(expectations, all_pairs[name]) for name in names}
    labelled = {i for pair in expectations for i in pair}

    frame_imgs = "".join(
        '<figure><img src="data:image/jpeg;base64,{}"><figcaption>{}</figcaption></figure>'.format(
            base64.b64encode(frame).decode("ascii"), html.escape(camera)
        )
        for camera, frame in zip(CAMERA_ORDER, frames)
    )

    # --- one column of merged groups per definition --------------------------- #
    columns = []
    for name, clustering in results.items():
        groups = [c for c in clustering.clusters if len(c) > 1]
        blocks = []
        for cluster in groups:
            cluster_pairs = pairs_of(Clustering([cluster]))
            unanimous = cluster_pairs <= shared
            badge = ('<span class="badge">all definitions</span>' if unanimous
                     else '<span class="badge solo">not unanimous</span>')
            if any(expectations.get(p) == "DISTINCT" for p in cluster_pairs):
                badge += '<span class="badge bad">merges a pair labelled DISTINCT</span>'
            if any(expectations.get(p) == "DUPLICATE" for p in cluster_pairs):
                badge += '<span class="badge good">finds a pair labelled DUPLICATE</span>'

            items = "".join(
                f'<li class="{"rep" if i == 0 else "dup"}">{html.escape(instructions[m])}</li>'
                for i, m in enumerate(cluster)
            )
            blocks.append(f'<div class="group">{badge}<ul>{items}</ul></div>')
        body = "".join(blocks) or '<p class="none">no merges</p>'
        raw = html.escape(clustering.raw_response or "(no call — fewer than two instructions)")
        columns.append(
            f'<div class="col"><h3>{html.escape(name)}</h3>'
            f'<p class="count">{len(instructions)} &rarr; {len(clustering.clusters)} '
            f'&middot; {clustering.num_duplicates} folded</p>{body}'
            f'<details><summary>definition</summary><pre>'
            f'{html.escape(resolve_definition(name)[1])}</pre></details>'
            f'<details><summary>raw response</summary><pre>{raw}</pre></details></div>'
        )

    # --- one row per instruction, one column per definition ------------------- #
    rows = []
    for index, text in enumerate(instructions):
        cells = []
        verdicts = set()
        for name in names:
            cluster = next(c for c in results[name].clusters if index in c)
            if cluster[0] == index:
                cells.append('<td class="kept">kept</td>')
                verdicts.add("kept")
            else:
                cells.append(f'<td class="folded">&rarr; {html.escape(instructions[cluster[0]])}</td>')
                verdicts.add(instructions[cluster[0]])
        disagree = ' class="disagree"' if len(verdicts) > 1 else ""
        mark = ' <span class="badge tiny">labelled</span>' if index in labelled else ""
        rows.append(f'<tr{disagree}><td class="instr">{html.escape(text)}{mark}</td>{"".join(cells)}</tr>')
    headers = "".join(f"<th>{html.escape(n)}</th>" for n in names)

    def cell(value: int, good: bool) -> str:
        if not expectations:
            return "<td>&mdash;</td>"
        klass = "kept" if good else "folded"
        return f'<td class="{klass}"><strong>{value}</strong></td>'

    summary_rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{len(results[n].clusters)}</td>"
        f"<td>{results[n].num_duplicates}</td>"
        f"<td>{len(all_pairs[n] - set.union(*(all_pairs[o] for o in names if o != n))) if len(names) > 1 else 0}</td>"
        f"{cell(scores[n]['caught'], True)}{cell(scores[n]['missed'], False)}"
        f"{cell(scores[n]['false_merges'], False)}</tr>"
        for n in names
    )

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")  # noqa: DTZ005
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Uniqueness definitions &mdash; comparison</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
         max-width: 1500px; padding: 24px; line-height: 1.5; }}
  h1 {{ margin-bottom: 4px; }}
  h2 {{ margin-top: 0; }}
  h3 {{ margin: 0 0 2px; font-size: 15px; }}
  .subtitle {{ color: #888; margin-top: 0; }}
  .card {{ border: 1px solid #8883; border-radius: 10px; padding: 16px 20px; margin: 16px 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ text-align: left; padding-right: 16px; color: #888; font-weight: 600; }}
  td {{ padding: 3px 12px 3px 0; vertical-align: top; }}
  .cols {{ display: flex; gap: 20px; align-items: flex-start; }}
  .col {{ flex: 1; min-width: 0; }}
  .count {{ color: #888; font-size: 13px; margin: 0 0 10px; }}
  .group {{ border: 1px solid #8883; border-radius: 8px; padding: 8px 12px; margin: 8px 0; }}
  .group ul {{ margin: 6px 0 0; padding-left: 18px; }}
  li.rep {{ font-weight: 600; }}
  li.dup {{ color: #888; }}
  .badge {{ display: inline-block; font-size: 11px; font-weight: 600;
           background: #2e7d3222; color: #2e7d32; border-radius: 6px; padding: 1px 6px; }}
  .badge.solo {{ background: #ef6c0022; color: #ef6c00; }}
  .badge.bad {{ background: #c6282822; color: #c62828; margin-left: 4px; }}
  .badge.good {{ background: #2e7d3222; color: #2e7d32; margin-left: 4px; }}
  .badge.tiny {{ background: #8882; color: #888; font-weight: 400; }}
  .none {{ color: #888; font-size: 13px; }}
  details {{ margin-top: 8px; }}
  details summary {{ cursor: pointer; color: #888; font-size: 13px; }}
  details pre {{ white-space: pre-wrap; background: #8881; padding: 12px;
                border-radius: 8px; font-size: 12px; }}
  .frames {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  figure {{ margin: 0; }}
  figure img {{ width: 260px; height: auto; border-radius: 6px; display: block; }}
  figcaption {{ font-size: 12px; color: #888; text-align: center; margin-top: 4px; }}
  table.full {{ font-size: 13px; margin-top: 10px; }}
  table.full td.instr {{ width: 34%; }}
  tr.disagree {{ background: #ef6c0011; }}
  td.kept {{ color: #2e7d32; }}
  td.folded {{ color: #ef6c00; }}
</style>
</head>
<body>
  <h1>Uniqueness definitions &mdash; comparison</h1>
  <p class="subtitle">{stamp} &middot; {len(instructions)} instructions &middot;
     judge <strong>{html.escape(model)}</strong> at temperature 0 &middot;
     {len(frames)} frame(s) shown{(" &middot; " + html.escape(source)) if source else ""}</p>

  <div class="card">
    <h2>Summary</h2>
    <table><tr><th>definition</th><th>kept</th><th>folded</th><th>only it found</th>
      <th>labelled DUPLICATE found</th><th>missed</th><th>labelled DISTINCT wrongly merged</th></tr>
    {summary_rows}</table>
    <p class="count">{len(expectations)} hand-labelled pair(s) in this set.
       {len(shared)} pair(s) merged by every definition.</p>
    <div class="frames">{frame_imgs}</div>
  </div>

  <div class="card">
    <h2>What each definition merged</h2>
    <div class="cols">{"".join(columns)}</div>
  </div>

  <div class="card">
    <h2>Every instruction</h2>
    <p class="count">Highlighted rows are where the definitions disagree.</p>
    <table class="full"><tr><th>instruction</th>{headers}</tr>{"".join(rows)}</table>
  </div>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--instructions", default=str(DEFAULT_INSTRUCTIONS), help="Text file, one instruction per line.")
    source.add_argument("--from-run", default=None, help="A generate/pipeline .txt to take accepted instructions from.")
    parser.add_argument("--frames-from", default=None, help="Directory to search for stepNNNN_<camera>.jpeg frames.")
    parser.add_argument("--step", type=int, default=0, help="Which step's frames to show the judge.")
    parser.add_argument("--total", type=int, default=439, help="Trajectory length, for the prompt's step context.")
    parser.add_argument("--definitions", nargs="+", default=list(DUPLICATE_DEFINITIONS),
                        help=f"Definitions to compare. Available: {', '.join(DUPLICATE_DEFINITIONS)}.")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--html", default=None,
                        help="Write an HTML comparison here "
                             "(defaults to outputs/visualizations/uniqueness_definitions_<stamp>.html).")
    parser.add_argument("--open", action="store_true", help="Open the page in a browser when done.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.from_run:
        source_path = Path(args.from_run)
        instructions = read_run_instructions(source_path)
        expectations: dict[tuple[int, int], str] = {}
    else:
        source_path = Path(args.instructions)
        instructions = read_instructions(source_path)
        expectations = read_expectations(source_path, instructions)

    frames = find_frames(Path(args.frames_from), args.step) if args.frames_from else []
    print(f"{len(instructions)} instructions, {len(expectations)} labelled pair(s), {len(frames)} frame(s)"
          + ("" if frames else "  — text only; object references cannot be resolved"))

    judge = build_uniqueness_judge(args.provider, args.model)
    results = {
        name: cluster_instructions(
            judge, instructions, frames, step=args.step, total=args.total, definition=name
        )
        for name in args.definitions
    }
    report(instructions, results)
    if expectations:
        print(f"\n{'=' * 78}\nscored against {len(expectations)} hand-labelled pair(s)\n{'=' * 78}")
        for name, clustering in results.items():
            marks = score(expectations, pairs_of(clustering))
            print(f"  {name:28} found {marks['caught']}, missed {marks['missed']}, "
                  f"wrongly merged {marks['false_merges']}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
    out = Path(args.html) if args.html else DEFAULT_VIZ_DIR / f"uniqueness_definitions_{stamp}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(instructions, results, frames, judge.model,
                               expectations, source_path.name))
    print(f"\nWrote {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
