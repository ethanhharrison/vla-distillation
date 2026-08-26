# Wan 2.2 exploration — general image-to-video as a Stage B subgoal generator

**Status: smoke only (n=1).** One situation, one camera, one seed, two prompts.
Enough to settle "does it run, and what does it do", not enough for any claim
about instruction sensitivity — that needs the A/B/C harness and a seed null.

Evaluates [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
(Alibaba, 27B MoE / 14B active, Apache 2.0) as the first **general, not
robot-post-trained** image-to-video model on our DROID frames. Cosmos 3 set up
the question: post-training on DROID bought physical faithfulness but cost
language responsiveness, and the general checkpoints followed the instruction
more. Wan 2.2 is the strongest open-weight general I2V model available — Wan 2.5
and 2.6 exist but are **API-only**, and there is no Wan 2.7 despite widely-copied
blog claims (checked against the [GitHub org](https://github.com/orgs/Wan-Video/repositories?type=all&sort=updated)
and [HF org](https://huggingface.co/Wan-AI), 2026-08-21).

## TL;DR

- **Runs easily on one H200 with no new venv.** The cosmos3 venv's pinned
  `diffusers 0.40.0.dev0` already ships `WanImageToVideoPipeline`; the only
  missing dependency was `ftfy`. 19 s load, **69.1 GB resident / 73.7 GB peak**,
  **260 s** per 81-frame 832x464 generation at 40 steps. Half a card, so it does
  not need a dedicated GPU the way Cosmos3-Super does.
- **Off the shelf it deletes the robot.** Given the bare DROID instruction
  "Put the marker in the pot", the rollout is faithful at frame 0 and then
  replaces the robot arm with **a human in a white shirt** who reaches in and
  does the task, while the camera pushes in. This is the pretraining bias
  [RoboWM-Bench](https://arxiv.org/html/2604.19092v1) names: human-hand
  manipulation dominates robot manipulation in general video corpora, so asked
  to accomplish a task the model reaches for the most likely agent, and that
  agent is a person.
- **A robot-explicit caption fixes the agent and cuts spurious motion 3x** — but
  does not make the rollout useful. The arm stays, no human appears, yet the
  camera still drifts and the marker vanishes rather than being placed.
- **Neither prompt beats copying the input frame**, the same test the Cosmos
  rollouts failed. Here the margin is far wider.

## Numbers

Situation `ep000_t0000`, camera `exterior_1`, seed 0, 81 frames, 40 steps,
guidance 3.5. Mean absolute pixel difference (0–255), generated frame 34 matched
to real DROID frame 32 (16 fps model vs 15 Hz data).

| | bare instruction | robot-explicit caption |
|---|---|---|
| recon floor (frame 0 vs real source) | 3.68 | **2.60** |
| **gen vs src** at the matched frame | **73.77** | **24.18** |
| gen vs real | 73.83 | 26.90 |
| real motion over the same span | 6.84 | 6.84 |
| over-motion vs reality | **10.8x** | **3.5x** |

- The caption is worth **3x**. That is a larger effect than any sampling knob
  measured on any model in this repo, which says the text path is where the
  leverage is, not `guidance_scale` or `steps`.
- **The recon floor of 3.68 / 2.60 cross-validates the metric.** Cosmos 3's three
  checkpoints all floored near 5 on the same frames, and both families use
  `AutoencoderKLWan` — so two independent harnesses agree on the VAE tax.
- **Copying the input frame still wins by 4x.** The unedited source sits 6.84
  from the real future; the best generation sits 26.90. Same caveat as the
  cosmos3 write-up: for a counterfactual instruction there is no ground truth and
  a correct subgoal *should* diverge, so this scores the sanity condition only.
- **`ep000` is the least favourable situation in the set** — real motion 6.84 is
  the lowest of the 8 (the others run 11–24). A scene that barely moves makes
  any generative over-motion look worst. Do not generalise this row.

## Cost of a real run

A grid matched to the one the hosted image editors got (8 situations x 5
instructions x 3 cameras = 120 generations) is **~8.7 h on one GPU**, plus ~13
min for the seed null. No no-op condition is needed: unlike an image editor, a
video model's floor is generated frame 0, which is free.

Levers, best first:

| lever | effect | cost |
|---|---|---|
| more GPUs (4 cards fit a ~75 GB worker) | ~2.2 h | none; compute-bound, so **do not stack two workers per card** |
| generate 37 frames not 81 | >2x | we only compare at frame 34; 35–80 are waste. Changes what the model plans toward |
| exterior_1 only | 3x | loses the per-camera breakdown |
| 40 -> 20 steps | 2x | unmeasured here; Cosmos found 30-vs-50 dead and steps=4 oddly *better* |

## What's here

| file | what |
|---|---|
| `prompts.py` | prompt templates + negative presets, mirroring `pipeline/subgoal_image/prompts.py` |
| `smoke_i2v.py` | one call: loads, generates, reports the calibration quantities |
| `run_views.py` | all three cameras, independent or stitched-canvas, one situation |
| `run_eval.py` | **the A/B/C grid** — resumable, one job per (config, situation) |
| `make_report.py` | prompt/view-mode comparison; three views per row |
| `make_eval_report.py` | per-config A/B/C report, cosmos3 layout |
| `analyze_eval.py` | cross-config table (floor / real / sens / seed null) |
| `sweep_prompts.sh` · `sweep_views.sh` · `sweep_eval.sh` · `sweep_multiview.sh` | detachable sweeps, GPU preflight, round-robin |

`results/` is a **symlink to `/raid/users/tiger/vla-distillation/wan_results`** — the
home fs is at 100 %. Both gitignore forms are needed, because a trailing-slash
pattern does not match a symlink.

## Reading a finished eval

```sh
P=../cosmos3/.venv/bin/python
$P analyze_eval.py --configs e_prefix e_explicit e_static e_cfg7   # the table
$P make_eval_report.py --config e_prefix --open                    # the pictures
cat results/eval_summary.txt results/multiview_summary.txt
```

Everything is rebuildable from `results/` at any time; the sweeps only produce
data. If a sweep was interrupted, **re-running it is safe and cheap** — every job
skips samples already on disk and re-runs only what is missing.

Runs on the **cosmos3 venv** (`../cosmos3/.venv`), deliberately: it already has a
Wan-capable diffusers on a cu128 torch that our 570 driver can run, and adding a
second 10 GB torch to a home fs at 100 % would be careless. The coupling risk is
real but small — Wan needs no diffusers upgrade, so the Cosmos pin is untouched.
`ftfy` was added to that venv (pure Python, no effect on torch/diffusers).

Weights: `/raid/users/tiger/vla-distillation/models/Wan2.2-I2V-A14B` (126 GB,
Apache 2.0, pulled in 65 s). Git-ignored: `results/`, `*.log`.

## Reproduce

```sh
cd explorations/wan
P=../cosmos3/.venv/bin/python
export CUDA_VISIBLE_DEVICES=<idle-gpu> HF_HOME=/raid/users/tiger/cache/hf

$P smoke_i2v.py                                    # bare instruction — swaps in a human
$P smoke_i2v.py --instruction "<robot-explicit caption>" \
   --negative-extra "人，人手，人的手臂，真人出现，摄像机移动，镜头推近，镜头变焦" \
   --out results/smoke_i2v_robotprompt
```

## Gotchas

- **`ftfy` is an undeclared hard dependency** of the diffusers Wan pipeline —
  `NameError: name 'ftfy' is not defined` inside `prompt_clean`, thrown *after*
  the model has loaded and only when a prompt is encoded.
- **The pipeline returns float ndarrays, not PIL.** `export_to_video` branches on
  type and multiplies an ndarray by 255; float arrays survive that, uint8 arrays
  wrap modulo 256 into garbage that still plays. `to_pil()` converts once,
  explicitly, so the bug cannot appear when a dtype changes.
- **fps is 16**, not TI2V-5B's 24 and not DROID's 15. The generated frame
  matching real frame `k` is `round(k / 15 * 16)`. Getting this wrong silently
  compares the wrong timestamps.
- **The card's negative prompt is Chinese** and is used verbatim; dropping it is
  a deviation, not a simplification.
- Stride rounding makes our exactly-16:9 frames 832x464 (aspect 1.793). Small,
  but it is a real distortion applied before every pixel metric.

## Where to pick up

1. **Decide whether a caption layer is acceptable at all.** The 3x result means
   Wan cannot be fed our raw instructions; something must expand them into
   descriptive, robot-explicit captions. That puts an LLM rewrite between the
   instruction and the model — the same deviation Cosmos I2V needed. It must be
   logged deliberately, because it sits inside the thing we are trying to
   measure: instruction sensitivity.
2. **Then, and only then, the A/B/C grid.** Sensitivity numbers computed while
   the model is swapping in a human would be well-defined and meaningless.
3. **Wan2.2-TI2V-5B** as the cheap arm — 5B, 24 GB, the base that the robotics
   world-model literature keeps choosing. Much faster, so multi-seed becomes
   affordable, which is the gap every experiment here has hit.
4. Sampling sweep (steps, guidance) — lowest expected value, given the caption
   dwarfed every knob measured so far.
