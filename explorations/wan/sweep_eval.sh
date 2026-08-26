#!/usr/bin/env bash
# The A/B/C eval grid for Wan, on the four configs that survived manual judging.
#
# Grid per config: 8 situations x 5 instructions (A + 2xB + 2xC) x 3 cameras
# = 120 generations, ~260 s each. Four configs = 480 generations = ~35 GPU-hours.
# Plus one seed-null job per config, without which no sensitivity number means
# anything.
#
# Designed to be left alone overnight:
#   - one job = one (config, situation); jobs never write a shared file, so they
#     run concurrently with no locking and no race
#   - every job is RESUMABLE — re-running skips samples already on disk, so a
#     kill, a crash or a second launch costs only unfinished work
#   - a failed generation is recorded and the job continues
#   - reports are built at the end, and can be rebuilt any time from the results
#
# DETACHABLE:
#   setsid bash sweep_eval.sh </dev/null >sweep_eval.log 2>&1 &
#   tail -f sweep_eval.log
#
#   GPUS="1 4 5 6" bash sweep_eval.sh     # default; leaves a card for other work
set -u
cd "$(dirname "$0")" || exit 1

P="../cosmos3/.venv/bin/python"
export HF_HOME=/raid/users/tiger/cache/hf
export TMPDIR=/raid/users/tiger/tmp
NEED_MIB=88000
SITUATIONS="${SITUATIONS:-../cosmos3/results/situations_multitraj}"

# config-name|args   — all four are independent-view; concat is handled by the
# follow-up sweep, since it did nothing for the exterior views.
CONFIGS=(
  "e_prefix|--prompt-template robot_prefix"
  "e_explicit|--prompt-template robot_explicit"
  "e_static|--prompt-template static_scene"
  "e_cfg7|--prompt-template robot_explicit --guidance-scale 7.0"
)

mapfile -t SITS < <("$P" -c "
import json,sys; print('\n'.join(json.load(open('$SITUATIONS/meta.json'))['situations']))")
if [ "${#SITS[@]}" -eq 0 ]; then echo "ABORT: no situations found in $SITUATIONS"; exit 1; fi

# Build the flat job list: every (config, situation), then the seed nulls last
# so the grid is finished before the control spends anything.
JOBS=()
for cfg in "${CONFIGS[@]}"; do
  name="${cfg%%|*}"; args="${cfg#*|}"
  for sid in "${SITS[@]}"; do
    JOBS+=("$name|$sid|$args --situation $sid")
  done
done
for cfg in "${CONFIGS[@]}"; do
  name="${cfg%%|*}"; args="${cfg#*|}"
  JOBS+=("$name|seednull|$args --seed-null")
done

pick_gpus() {
  local out=()
  while IFS=', ' read -r idx used total; do
    if [ $((total - used)) -ge "$NEED_MIB" ]; then out+=("$idx"); fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total \
             --format=csv,noheader,nounits)
  echo "${out[@]}"
}

read -r -a GPU_LIST <<< "${GPUS:-$(pick_gpus)}"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "ABORT: no GPU has ${NEED_MIB} MiB free (Wan A14B peaks at ~73.7 GB)."
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
  exit 1
fi
echo "$(date +%F\ %H:%M:%S) · GPUs ${GPU_LIST[*]} · ${#JOBS[@]} jobs · "\
"${#SITS[@]} situations x ${#CONFIGS[@]} configs · expect ~$(( ${#JOBS[@]} * 65 / ${#GPU_LIST[@]} / 60 ))h"

for slot in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$slot]}"
  (
    for i in "${!JOBS[@]}"; do
      [ $((i % ${#GPU_LIST[@]})) -eq "$slot" ] || continue
      IFS='|' read -r name sid args <<< "${JOBS[$i]}"
      log="sweep_eval_${name}_${sid}.log"
      echo "[gpu $gpu] === $name/$sid === $(date +%H:%M:%S)"
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES="$gpu" nice -n 5 "$P" -u run_eval.py \
          --config-name "$name" --situations "$SITUATIONS" $args > "$log" 2>&1 \
        || echo "[gpu $gpu]   JOB FAILED: $name/$sid (see $log)"
      tail -1 "$log" | sed "s/^/[gpu $gpu]   /"
    done
    echo "[gpu $gpu] done $(date +%H:%M:%S)"
  ) &
done
wait

echo
echo "=== grid finished $(date +%F\ %H:%M:%S) ==="
for cfg in "${CONFIGS[@]}"; do
  name="${cfg%%|*}"
  "$P" make_eval_report.py --config "$name" 2>&1 | sed 's/^/  /'
done
NAMES=$(printf '%s\n' "${CONFIGS[@]}" | cut -d'|' -f1 | tr '\n' ' ')
# shellcheck disable=SC2086
"$P" analyze_eval.py --configs $NAMES 2>&1 | tee results/eval_summary.txt

echo
echo "=== chaining to the multi-view follow-up $(date +%H:%M:%S) ==="
bash sweep_multiview.sh || echo "multiview follow-up failed; the eval results above are unaffected"
