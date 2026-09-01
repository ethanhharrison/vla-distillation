"""Compare Wan eval configs on floor, motion, fidelity and instruction sensitivity.

The counterpart of `explorations/cosmos3/analyze_cfg.py`. Reuses
`make_eval_report.build`, so the table and the HTML can never disagree about what
a number means — there is one implementation of each metric.

  floor        generated frame 0 vs the real source. Pure VAE tax; the model
               cannot beat it, so treat anything below it as zero.
  real motion  real future at t+k vs source, averaged over cameras. The scale
               everything else is read against, NOT a target: a counterfactual
               subgoal should not match what actually happened.
  sens         |A - counterfactual| at the matched frame — how much swapping the
               instruction moved the rollout.
  seed null    the same instruction at a different seed. Sensitivity below this
               is indistinguishable from re-rolling the dice, which is the whole
               reason the column exists.

    ../cosmos3/.venv/bin/python analyze_eval.py --configs e_prefix e_explicit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from make_eval_report import build

HERE = Path(__file__).resolve().parent


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", required=True)
    args = p.parse_args(argv)

    rows = []
    for name in args.configs:
        run_dir = HERE / "results" / name
        if not (run_dir / "config.json").exists():
            print(f"  ! {name}: no config.json, skipping")
            continue
        try:
            _, st = build(run_dir)
            rows.append((name, st))
        except SystemExit as exc:
            print(f"  ! {name}: {exc}")

    if not rows:
        raise SystemExit("nothing to compare")

    head = (f"{'config':14} {'sits':>4} {'n':>4} {'floor':>6} {'real':>6} "
            f"{'sens':>6} {'null':>6} {'sens/null':>10}")
    print(head)
    print("-" * len(head))
    for name, st in rows:
        null = st["null"]
        ratio = "n/a" if not null else f"{st['sens'] / null:.0%}"
        print(f"{name:14} {st['situations']:4d} {st['samples']:4d} "
              f"{st['floor']:6.1f} {st['real']:6.1f} {st['sens']:6.1f} "
              f"{'n/a' if null is None else f'{null:6.1f}'} {ratio:>10}")

    print("\nall numbers are mean |a-b| on 0-255 RGB, comparable to the "
          "cosmos3 / dreamzero / image_edit tables")
    print("sens/null > 100% means the instruction moves the rollout more than the seed does")
    for name, st in rows:
        if st["null"] is None:
            print(f"  ! {name}: no seed null — its sensitivity is uninterpretable "
                  f"(run: run_eval.py --config-name {name} --seed-null)")


if __name__ == "__main__":
    main()
