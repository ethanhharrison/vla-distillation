#!/usr/bin/env bash
# Comprehensive hyperparameter sweep for the GENERAL Cosmos3-Nano, mirroring the
# Policy-DROID sweep: same 8 trajectories at frame 0, full condition set
# (original + 2 Gemini counterfactuals + 2 impossible), all three cameras.
#
# Each run is followed by its own seed-null measurement, because we learned the
# null is NOT a constant — it doubled from 22 to 46 when we raised resolution, so
# sensitivity is only interpretable against the null of the same configuration.
#
# Grid choices reflect what earlier sweeps showed actually matters:
#   - guidance_scale  : real effect on text adherence vs drift
#   - upscale         : the only knob that changes output resolution
#   - chunk_size      : horizon, and therefore how much drift accumulates
#   - concat_view     : the 3-view canvas; halved drift on the Policy-DROID model
#   - steps           : included for parity even though it moved nothing before
# Deliberately dropped: flow_shift (no effect on either model), tier alone
# (a ceiling, not a resolution knob), karras (confounded, no effect).
#
#   bash sweep_general_nano.sh [gpu_group]     # 0, 1 or 2; run the three in parallel
set -u
cd "$(dirname "$0")" || exit 1
P="$PWD/.venv/bin/python"
MODEL=/home/tiger/proj/staging/vla/models/Cosmos3-Nano
OUT=results/sweep_general_nano
COMMON="--situations results/situations_multitraj --fps 15 --cameras exterior_1 exterior_2 wrist --model-path $MODEL"
export HF_HOME=/home/tiger/proj/staging/vla/models/hf_cache
export TMPDIR=/home/tiger/proj/staging/vla/tmp
mkdir -p "$OUT"

# one_run <report-name> <run-name> <extra flags...>
one_run() {
  local report="$1" name="$2"; shift 2
  echo "=== $report ($name) ==="
  nice -n 5 "$P" -u run_experiment.py $COMMON --chunk-size 32 --guidance-scale 3.0 "$@" --run-name "$name" \
      >> "sweep_$name.log" 2>&1 || { echo "  RUN FAILED: $name"; return; }
  nice -n 5 "$P" -u seed_null.py --run "$name" >> "sweep_$name.log" 2>&1 || echo "  seed-null failed: $name"
  "$P" make_report.py --run "$name" --output "$OUT/$report.html" 2>&1 | head -2
}

case "${1:-0}" in
0)  # low-resolution guidance axis + the cheap steps arm
    one_run "cfg=1"     ngs_cfg1     --guidance-scale 1.0
    one_run "cfg=2"     ngs_cfg2     --guidance-scale 2.0
    one_run "cfg=7"     ngs_cfg7     --guidance-scale 7.0
    one_run "steps=4"   ngs_steps4   --steps 4
    ;;
1)  # structure / horizon axis
    one_run "chunk=16"     ngs_chunk16 --chunk-size 16
    one_run "concat3view"  ngs_concat  --resolution-tier 480 --concat-view
    one_run "steps=50"     ngs_steps50 --steps 50
    ;;
2)  # high-resolution axis: can upscaling's instability be tamed?
    one_run "upscale=2_chunk=16" ngs_up2_chunk16 --upscale 2 --resolution-tier 480 --chunk-size 16
    one_run "upscale=2_cfg=2"    ngs_up2_cfg2    --upscale 2 --resolution-tier 480 --guidance-scale 2.0
    ;;
esac
echo "GROUP ${1:-0} COMPLETE"
