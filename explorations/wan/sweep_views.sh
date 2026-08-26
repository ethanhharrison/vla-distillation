#!/usr/bin/env bash
# Multi-view sweep: the four prompts that survived manual judging, plus the cfg7
# arm, each generated BOTH ways — three independent per-camera rollouts, and one
# rollout of the stitched three-view canvas split back apart.
#
# The question is not which scores better on pixel difference. It is whether the
# three views agree with each other, which no metric here measures and which the
# report is laid out for you to judge by eye.
#
# Load is balanced by interleaving: an independent run is 3 generations (~13 min)
# and a concat run is 1 (~4.7 min), so round-robin over 5 cards pairs one of each
# and every card finishes around the same time.
#
# DETACHABLE:
#   setsid bash sweep_views.sh </dev/null >sweep_views.log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1

P="../cosmos3/.venv/bin/python"
export HF_HOME=/raid/users/tiger/cache/hf
export TMPDIR=/raid/users/tiger/tmp
COMMON="--situation ep000_t0000 --seed 0 --negative-extra no_human_no_camera"
NEED_MIB=80000

RUNS=(
  "v_prefix_ind|--prompt-template robot_prefix     --view-mode independent"
  "v_prefix_cat|--prompt-template robot_prefix     --view-mode concat"
  "v_explicit_ind|--prompt-template robot_explicit --view-mode independent"
  "v_explicit_cat|--prompt-template robot_explicit --view-mode concat"
  "v_static_ind|--prompt-template static_scene     --view-mode independent"
  "v_static_cat|--prompt-template static_scene     --view-mode concat"
  "v_subgoal_ind|--prompt-template subgoal_partial --view-mode independent"
  "v_subgoal_cat|--prompt-template subgoal_partial --view-mode concat"
  "v_cfg7_ind|--prompt-template robot_explicit     --view-mode independent --guidance-scale 7.0"
  "v_cfg7_cat|--prompt-template robot_explicit     --view-mode concat      --guidance-scale 7.0"
)

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
echo "using GPUs: ${GPU_LIST[*]}  ·  ${#RUNS[@]} runs  ·  ~18 min"

for slot in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$slot]}"
  (
    for i in "${!RUNS[@]}"; do
      [ $((i % ${#GPU_LIST[@]})) -eq "$slot" ] || continue
      name="${RUNS[$i]%%|*}"; args="${RUNS[$i]#*|}"
      echo "[gpu $gpu] === $name === $(date +%H:%M:%S)"
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES="$gpu" nice -n 5 "$P" -u run_views.py \
          $COMMON $args --out "results/$name" > "sweep_$name.log" 2>&1 \
        || { echo "[gpu $gpu]   RUN FAILED: $name (see sweep_$name.log)"; continue; }
      grep -E "^  \[" "sweep_$name.log" | sed "s/^/[gpu $gpu] /"
    done
    echo "[gpu $gpu] done $(date +%H:%M:%S)"
  ) &
done
wait

echo
echo "=== all runs finished $(date +%H:%M:%S) ==="
NAMES=$(printf '%s\n' "${RUNS[@]}" | cut -d'|' -f1 | tr '\n' ' ')
# shellcheck disable=SC2086
"$P" make_report.py --runs $NAMES --output results/report_views.html
echo "report: results/report_views.html"
