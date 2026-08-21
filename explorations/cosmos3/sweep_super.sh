#!/usr/bin/env bash
# Hyperparameter sweep for Cosmos3-SUPER (64B), mirroring sweep_general_nano.sh:
# same 8 trajectories at frame 0, same A/B/C conditions, same three cameras, so
# every number lands in the same tables as the two Nano checkpoints.
#
# Super is the general (not DROID-post-trained) model, exactly like the general
# Nano — so this is a clean capacity-scaling comparison, 16B vs 64B, with the
# specialist Policy-DROID run as the third reference column.
#
# Two differences from the Nano sweep, both forced by the model:
#
#  1. SEQUENTIAL, NOT THREE PARALLEL GROUPS. Super peaks at ~135 GB of a 143.7 GB
#     H200, so one run owns an entire card. Do not launch groups concurrently
#     unless that many cards are genuinely idle — we share this box.
#  2. SMALLER GRID. The two Nano sweeps already settled several axes; repeating
#     them at 4x the parameter count buys nothing. Dropped as measured-dead on
#     BOTH Nano models: flow_shift, steps 30-vs-50, resolution_tier on its own,
#     karras sigmas. Dropped as lower value for now: cfg=2 (the 1/3/7 spread
#     already resolves the trend), chunk=16 (measures against a different real
#     baseline, so it is not comparable to the 32-frame rows), steps=4 (a real
#     but unexplained Nano effect — worth a visual check, not GPU-hours here).
#
# Each run is followed by its own seed-null, because the null is NOT a constant
# (it doubled 22 -> 46 when resolution changed on Nano), so sensitivity is only
# interpretable against the null of the same configuration.
#
#   bash sweep_super.sh              # all six on one GPU, sequential, ~2.5-3.5 h
#   GPU=0 bash sweep_super.sh 0      # or split across cards, one FULL card each:
#   GPU=1 bash sweep_super.sh 1      #   group 0 ~35 min, group 1 ~1.2 h,
#   GPU=5 bash sweep_super.sh 2      #   group 2 ~1 h  => ~1.2 h wall clock
set -u
cd "$(dirname "$0")" || exit 1

# One run owns a whole card. Refuse to start if this GPU cannot actually hold the
# model — better to bail now than to OOM an hour in, or to push a co-tenant over.
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
_g="${CUDA_VISIBLE_DEVICES:-0}"
_free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$_g" \
        | awk -F', ' '{print $1-$2}')
if [ "${_free:-0}" -lt 138000 ]; then
  echo "ABORT: GPU $_g has only ${_free} MiB free; Super peaks at ~138000 MiB (135 GB)."
  echo "       Pick an emptier card with: nvidia-smi --query-gpu=index,memory.used --format=csv"
  exit 1
fi
echo "using GPU $_g (${_free} MiB free)"
P="$PWD/.venv/bin/python"
MODEL=/raid/users/tiger/vla-distillation/models/Cosmos3-Super
OUT=results/sweep_super
COMMON="--situations results/situations_multitraj --fps 15 --cameras exterior_1 exterior_2 wrist --model-path $MODEL"
# Weights and caches live on /raid: the home fs is at 98% and cannot take them.
export HF_HOME=/raid/users/tiger/cache/hf
export TMPDIR=/raid/users/tiger/tmp
mkdir -p "$OUT"

one_run() {
  local report="$1" name="$2"; shift 2
  echo "=== $report ($name) === $(date +%H:%M:%S)"
  nice -n 5 "$P" -u run_experiment.py $COMMON --chunk-size 32 --guidance-scale 3.0 "$@" --run-name "$name" \
      >> "sweep_$name.log" 2>&1 || { echo "  RUN FAILED: $name"; return; }
  # seed_null rebuilds its config from index.json and does NOT inherit new flags
  # automatically — it has silently mismatched geometry before. Check it if you
  # add a flag to the grid.
  nice -n 5 "$P" -u seed_null.py --run "$name" >> "sweep_$name.log" 2>&1 || echo "  seed-null failed: $name"
  "$P" make_report.py --run "$name" --output "$OUT/$report.html" 2>&1 | head -2
}

cheap() {
  # Guidance axis at the baseline resolution. On the general Nano CFG was a clean
  # monotonic dial (1->7 drove jitter 1.16->1.42, sensitivity 16.7->31.8) while on
  # the specialist it did nothing. Which way does 64B behave?
  one_run "super_cfg=3_baseline" sup_cfg3                        # anchors to the Nano baseline row
  one_run "super_cfg=1"          sup_cfg1 --guidance-scale 1.0
  one_run "super_cfg=7"          sup_cfg7 --guidance-scale 7.0
}

big() {
  # concat3view is the best lever found on BOTH Nano models (roughly halves drift,
  # closest-to-real motion). A third model either confirms it or breaks it.
  one_run "super_concat3view" sup_concat --resolution-tier 480 --concat-view
  # upscale was the only image-quality knob and it cost stability on Nano (best
  # floor 3.1, worst drift, seed null doubled). Does 4x capacity absorb that?
  one_run "super_upscale=2"   sup_up2    --resolution-tier 480 --upscale 2
  # The #1 untested item on the pick-up list: the two winners are on independent
  # axes (stability from layout, sharpness from resolution). Tier 720 is required
  # to exceed the concat canvas ceiling, since the tier never upscales.
  one_run "super_concat+upscale=2" sup_concat_up2 --resolution-tier 720 --concat-view --upscale 2
}

case "${1:-all}" in
  cheap|0) cheap ;;
  big)     big ;;
  # Groups for fan-out across cards, balanced by cost. One FULL card each --
  # these are NOT three processes sharing a GPU.
  1) one_run "super_concat3view" sup_concat --resolution-tier 480 --concat-view
     one_run "super_upscale=2"   sup_up2    --resolution-tier 480 --upscale 2 ;;
  2) one_run "super_concat+upscale=2" sup_concat_up2 --resolution-tier 720 --concat-view --upscale 2 ;;
  all)     cheap; big ;;
  # Re-run of the four configs corrupted on 2026-08-13 by an export bug: frames
  # were handed to export_to_video as ndarray, which it treats as float [0,1] and
  # multiplies by 255, wrapping uint8 into garbage (reconstruction floor ~132
  # instead of ~5). cfg=3 and concat3view predate the bug and are still valid.
  redo0) one_run "super_cfg=1" sup_cfg1 --guidance-scale 1.0
         one_run "super_cfg=7" sup_cfg7 --guidance-scale 7.0 ;;
  redo1) one_run "super_upscale=2" sup_up2 --resolution-tier 480 --upscale 2 ;;
  redo2) one_run "super_concat+upscale=2" sup_concat_up2 --resolution-tier 720 --concat-view --upscale 2 ;;
  *) echo "usage: [GPU=n] $0 [all|cheap|big|0|1|2|redo0|redo1|redo2]"; exit 1 ;;
esac
echo "SWEEP COMPLETE $(date +%H:%M:%S)"
