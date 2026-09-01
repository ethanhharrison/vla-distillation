#!/usr/bin/env bash
# Follow-up to the eval grid: two more attempts at mutually consistent views.
#
# Where this comes from. Independent per-camera generation gives no cross-view
# guarantee at all. The Cosmos-layout canvas (wrist 640x360 above two 320x180
# exteriors) DID collapse exterior motion to ~6.5 against real motion of 6.8 —
# but it also left the wrist at ~9-13 against real motion of 61, and made the
# output almost prompt-insensitive (6.4 / 6.5 / 6.4 / 6.5 across four quite
# different captions). Two readings fit that equally well:
#
#   (a) the canvas genuinely constrains the model to one coherent scene, or
#   (b) each view got a third of the pixels (344x192) and the model gave up.
#
# These two arms separate them:
#
#   hires   the SAME Cosmos layout at a 720p pixel budget instead of 480p, so
#           each view gets ~2.3x the pixels. If (b) is right this should restore
#           motion and prompt sensitivity; if (a) is right it should change little.
#   grid    a plain 2x2 of native-resolution cells — also the shape DreamZero
#           emits natively, so the likelier layout for a general model to have
#           seen than the Cosmos wrist-on-top arrangement.
#
# Both run on the prompt that won the eyeball test, over the same situations, so
# they drop straight into the same tables. Chained from sweep_eval.sh, but safe
# to run alone.
set -u
cd "$(dirname "$0")" || exit 1

P="../cosmos3/.venv/bin/python"
export HF_HOME=/raid/users/tiger/cache/hf
export TMPDIR=/raid/users/tiger/tmp
NEED_MIB=88000
SITUATIONS="${SITUATIONS:-../cosmos3/results/situations_multitraj}"
TPL="${TPL:-robot_prefix}"
# Concat is one generation for all three views, so a whole config is only ~1/3
# the cost of an independent one. Four situations is enough to see whether the
# effect is real before spending the full eight.
NSIT="${NSIT:-4}"

CONFIGS=(
  "m_cat_hires|--view-mode concat --canvas-layout cosmos  --max-area 921600"
  "m_cat_grid|--view-mode concat --canvas-layout grid2x2 --max-area 921600"
)

mapfile -t SITS < <("$P" -c "
import json; print('\n'.join(json.load(open('$SITUATIONS/meta.json'))['situations'][:$NSIT]))")
[ "${#SITS[@]}" -eq 0 ] && { echo "ABORT: no situations in $SITUATIONS"; exit 1; }

JOBS=()
for cfg in "${CONFIGS[@]}"; do
  name="${cfg%%|*}"; args="${cfg#*|}"
  for sid in "${SITS[@]}"; do JOBS+=("$name|$sid|$args --situation $sid"); done
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
  echo "ABORT: no GPU has ${NEED_MIB} MiB free."; exit 1
fi
echo "$(date +%F\ %H:%M:%S) · multiview · GPUs ${GPU_LIST[*]} · ${#JOBS[@]} jobs · template $TPL"

for slot in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$slot]}"
  (
    for i in "${!JOBS[@]}"; do
      [ $((i % ${#GPU_LIST[@]})) -eq "$slot" ] || continue
      IFS='|' read -r name sid args <<< "${JOBS[$i]}"
      log="sweep_mv_${name}_${sid}.log"
      echo "[gpu $gpu] === $name/$sid === $(date +%H:%M:%S)"
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES="$gpu" nice -n 5 "$P" -u run_eval.py \
          --config-name "$name" --situations "$SITUATIONS" \
          --prompt-template "$TPL" $args > "$log" 2>&1 \
        || echo "[gpu $gpu]   JOB FAILED: $name/$sid (see $log)"
      tail -1 "$log" | sed "s/^/[gpu $gpu]   /"
    done
    echo "[gpu $gpu] done $(date +%H:%M:%S)"
  ) &
done
wait

echo
echo "=== multiview finished $(date +%F\ %H:%M:%S) ==="
NAMES=$(printf '%s\n' "${CONFIGS[@]}" | cut -d'|' -f1 | tr '\n' ' ')
for name in $NAMES; do
  "$P" make_eval_report.py --config "$name" 2>&1 | sed 's/^/  /'
done
# shellcheck disable=SC2086
"$P" analyze_eval.py --configs $NAMES 2>&1 | tee results/multiview_summary.txt
