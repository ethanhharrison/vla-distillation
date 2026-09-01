#!/usr/bin/env bash
# Prompt + hyperparameter sweep for Wan2.2-I2V-A14B, one situation, one camera,
# one seed. Everything is held fixed except the axis named in each run.
#
# Why prompts dominate this grid: the first two smokes moved the rollout 3x by
# changing the caption alone, which is a larger effect than any sampling knob
# measured on any model in this repo. So five of the eleven runs vary wording,
# and the hyperparameter arms are all run on one fixed caption so they stay
# interpretable.
#
# Two ablations worth naming, because they are the ones that could collapse the
# grid:
#   bare_negonly  - bare instruction + the anti-human NEGATIVE prompt. If this
#                   alone stops the model substituting a human, then the long
#                   positive captions are wasted tokens.
#   gs2_low       - guidance 3.5 on the high-noise expert, 1.0 on the low-noise
#                   one. Wan2.2's MoE splits at boundary_ratio 0.9; every run so
#                   far let the second expert inherit the first's guidance. The
#                   hypothesis is "commit to the instructed change early, stay
#                   faithful to the scene late", which is exactly our failure mode.
#
# DETACHABLE. Survives disconnect and frees each GPU as it goes:
#   setsid bash sweep_prompts.sh </dev/null >sweep.log 2>&1 &
#   tail -f sweep.log
#
#   GPUS="1 4" bash sweep_prompts.sh      # restrict to specific cards
set -u
cd "$(dirname "$0")" || exit 1

P="../cosmos3/.venv/bin/python"
export HF_HOME=/raid/users/tiger/cache/hf
export TMPDIR=/raid/users/tiger/tmp
COMMON="--situation ep000_t0000 --camera exterior_1 --seed 0"
NEED_MIB=80000        # measured peak is 73.7 GB; leave headroom for a co-tenant

# name|extra args   — one run per line, held to one varying axis each
RUNS=(
  # --- prompt axis (guidance 3.5, 40 steps) ---------------------------------
  "p_bare|--prompt-template bare --negative-extra none"
  "p_bare_negonly|--prompt-template bare --negative-extra no_human_no_camera"
  "p_robot_prefix|--prompt-template robot_prefix --negative-extra no_human_no_camera"
  "p_robot_explicit|--prompt-template robot_explicit --negative-extra no_human_no_camera"
  "p_static_scene|--prompt-template static_scene --negative-extra no_human_no_camera"
  "p_subgoal_partial|--prompt-template subgoal_partial --negative-extra no_human_no_camera"
  # --- hyperparameter axis (all on robot_explicit) --------------------------
  "h_cfg1|--prompt-template robot_explicit --negative-extra no_human_no_camera --guidance-scale 1.0"
  "h_cfg2|--prompt-template robot_explicit --negative-extra no_human_no_camera --guidance-scale 2.0"
  "h_cfg7|--prompt-template robot_explicit --negative-extra no_human_no_camera --guidance-scale 7.0"
  "h_gs2_low|--prompt-template robot_explicit --negative-extra no_human_no_camera --guidance-scale-2 1.0"
  "h_steps20|--prompt-template robot_explicit --negative-extra no_human_no_camera --steps 20"
)

# Pick cards that can actually hold a worker. Refuse rather than OOM an hour in,
# and never push a co-tenant over: this box is shared.
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
echo "using GPUs: ${GPU_LIST[*]}  ·  ${#RUNS[@]} runs  ·  ~4.5 min each"

# Round-robin the runs onto the cards; each card works through its share
# sequentially, because generation is compute-bound and two workers on one card
# just halve each other's speed.
for slot in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$slot]}"
  (
    for i in "${!RUNS[@]}"; do
      [ $((i % ${#GPU_LIST[@]})) -eq "$slot" ] || continue
      name="${RUNS[$i]%%|*}"; args="${RUNS[$i]#*|}"
      echo "[gpu $gpu] === $name === $(date +%H:%M:%S)"
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES="$gpu" nice -n 5 "$P" -u smoke_i2v.py \
          $COMMON $args --out "results/$name" > "sweep_$name.log" 2>&1 \
        || { echo "[gpu $gpu]   RUN FAILED: $name (see sweep_$name.log)"; continue; }
      grep -E "gen vs src|gen vs real|recon floor" "sweep_$name.log" | sed "s/^/[gpu $gpu]   /"
    done
    echo "[gpu $gpu] done $(date +%H:%M:%S)"
  ) &
done
wait

echo
echo "=== all runs finished $(date +%H:%M:%S) ==="
NAMES=$(printf '%s\n' "${RUNS[@]}" | cut -d'|' -f1 | tr '\n' ' ')
# shellcheck disable=SC2086
"$P" make_report.py --runs $NAMES --output results/report_sweep.html
echo "report: results/report_sweep.html"
