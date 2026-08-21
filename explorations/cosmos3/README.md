# Cosmos 3 exploration — policy-mode subgoal generation

**Status: exploration / evidence, not integrated.** Evaluates
[Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID)
(NVIDIA, 16B, Mixture-of-Transformers omnimodal world model) as a **subgoal
generator**: given one real DROID frame + a language instruction it jointly
denoises a future video and an action chunk. Same protocol, same situations and
same conditions as `explorations/dreamzero/`, so the two are comparable.

Three checkpoints are now measured on that one protocol: the DROID-post-trained
Nano, the general Nano (both 16B) and the general **Super (64B)**. There is no
Super-tier Policy-DROID — the specialist exists only at Nano and Edge — so Super
is compared against the *general* Nano, which makes it a clean capacity test.

Full write-up: `~/proj/notes/vla/SESSION_COSMOS3.md`.

## TL;DR

- **Runs on one GPU here with no Docker and no vLLM** — plain diffusers + torch
  cu128. 7.9 s load, **1.4–3.5 s per call**, **32.5 GB VRAM**, checkpoint 31 GB
  and fully self-contained (no 77 GB surprise like DreamZero).
- **Video quality is good**: scene identity holds and the arm moves plausibly
  toward the referenced object across the 17-frame rollout.
- **How much the instruction steers it depends heavily on the experiment design.**
  On a *single* trajectory (5 anchors, one scene) the instruction moved the
  rollout only 50 % as much as re-rolling the random seed. On *eight different*
  trajectories at their first frame, that roughly doubles and the exterior
  cameras reach **100-111 % of the seed null** — i.e. swapping the instruction
  now moves the output as much as re-rolling the dice. The single-scene number
  was substantially an artifact: every counterfactual there referred to the same
  pot and marker, so it was barely a different instruction at all.
- **Still no clean semantic differentiation *at 16B*.** Plausible (B) and
  impossible (C) instructions perturb the rollout by similar amounts, with no
  consistent ordering across cameras. That is ambiguous rather than damning: a
  semantically-distant instruction producing a more distant rollout is also what
  a model that *did* read the text would do. **Super settles it** — C > B in all
  six of its configs, and the gap widens monotonically with guidance.
- **Capacity buys language, not physics.** Super's sensitivity is **119-186 % of
  its own seed null** (general Nano 103 %, Policy-DROID 77 %), so the instruction
  finally moves the rollout more than the dice do. But it is *more* jittery and
  *less* faithful to the real future than the 16B generalist at every CFG.
- **Classifier-free guidance is a real confound.** NVIDIA's own diffusers example
  passes `guidance_scale=1.0` (no CFG); the tech report uses **3** for this
  policy. Going 1 → 3 raises action sensitivity **9×**. Anyone repeating this
  must not use the example's default.
- **Caveat that limits the whole result:** this path feeds the model a *single*
  frame with no proprioception under the generic `droid_lerobot` domain, whereas
  the checkpoint was post-trained on a 540×640 three-view canvas **with**
  proprioception, predicting 32 absolute joint positions at 15 Hz. It also tops
  RoboLab closed-loop (39.7 % specific vs 20.6 % vague instructions — clear
  language sensitivity there). So this measures *this configuration*, not the
  model's ceiling.

## Numbers

Mean absolute pixel difference (0–255), timestamp-matched at 15 Hz, generated
frame +16 vs the corresponding real DROID frame. Action MAE is in the model's
normalized 10D space (mean |a| ≈ 0.57). CFG 3.0, seed 0, **all three cameras**,
5 situations × 6 conditions = 90 generations (`runs_allcams`).

| calibration constant | value |
|---|---|
| VAE reconstruction floor (gen frame 0 vs real source) | **6.0** |
| real motion over the horizon (real +16 vs source) | **13.9** |
| **seed null** (same instruction, seed 0 vs 1) | **video 12.5 · action 0.203** |

| condition | video vs A | action MAE vs A |
|---|---|---|
| B · counterfactual (plausible, Gemini-proposed) | 5.6 | 0.019 |
| C · null/stress (impossible: "fold the laundry") | 6.6 | 0.023 |
| MP · targeted ("move the pot") | 6.4 | 0.022 |
| **overall (75 comparisons)** | **6.2** (50 % of seed null) | **0.021** (10 % of seed null) |

Per camera — the conclusion holds across all three views:

| camera | recon floor | real motion | video vs A | action vs A | as % of that camera's seed null |
|---|---|---|---|---|---|
| exterior_1 | 7.3 | 8.6 | 6.5 | 0.030 | 67 % |
| exterior_2 | 7.0 | 7.7 | 4.2 | 0.017 | 52 % |
| wrist | 3.7 | **31.5** | 7.7 | 0.017 | 39 % |

The wrist view moves ~4× more than the exteriors over the same horizon (31.5 vs
~8), matching what `real_future` showed in the Stage B session — and it also has
by far the largest seed null (19.7), so it is the noisiest channel to judge.

### Multi-trajectory: 8 scenes, each at its own first frame

The single-trajectory design above confounds instruction sensitivity with scene
diversity. This run takes **8 different DROID episodes at frame 0** (horizon 32,
because DROID episodes idle at the start), so each situation is a distinct scene
with its own task and its own counterfactuals. 40 samples x 3 cameras = 120
generations (`runs_multitraj`).

| camera | recon floor | real motion | seed null | instruction effect | ratio |
|---|---|---|---|---|---|
| exterior_1 | 5.4 | 14.4 | 10.8 | **10.7** | **100 %** |
| exterior_2 | 5.3 | 12.0 | 7.8 | **8.7** | **111 %** |
| wrist | 4.1 | 47.8 | 49.0 | 30.4 | 62 % |
| aggregate | 4.9 | 20.5 | 22.5 | 16.6 | 74 % |

Action MAE vs A: 0.047 against a 0.283 seed null (17 %).
Plausible vs impossible: B 15.8 / C 17.4 aggregate, but per camera the ordering
flips (ext_1 9.9/11.6, ext_2 9.1/8.3, wrist 28.4/32.5) — no reliable signal.

Per scene (video divergence from condition A, both conditions averaged):

| situation | real motion | B | C | task |
|---|---|---|---|---|
| ep000 | 6.8 | 20.4 | 20.0 | Put the marker in the pot |
| ep001 | 12.2 | 16.9 | 26.0 | Put the candy bar on the first shelf |
| ep002 | 11.3 | 19.6 | 15.3 | Put one green sachet in the grey bowl |
| ep003 | 12.4 | 25.5 | 23.8 | Place the doritos inside the sink |
| ep010 | 16.9 | 12.0 | 16.7 | Move the pan to the right |
| ep013 | 24.0 | 14.1 | 15.9 | Pick up the black object from the towel |
| ep020 | 17.1 | 9.8 | 14.0 | Slide the black lid off |
| ep021 | 14.4 | 7.7 | 7.6 | Put the orange rubber duck into the pot |

Note the **wrist camera dominates the aggregate seed null** (49.0 vs ~8-11 for the
exteriors), so aggregate ratios understate the exteriors. Read per camera.

## Sampling parameters — what actually matters

Swept on the 8-scene multi-trajectory set (conditions A + C, all three cameras;
fidelity from the 24 condition-A generations, sensitivity from the A-vs-C pairs).
`fidelity` = mean |gen - real| over the clip, `jitter g/r` = generated
frame-to-frame motion over the real clip's, `sens` = |A - C| at the last frame.

| setting | fidelity | drift | jitter g/r | sensitivity |
|---|---|---|---|---|
| CFG 1.0 | 20.1 | 22.4 | 0.76 | 14.0 |
| CFG 2.0 | 19.3 | 21.9 | 0.64 | 16.8 |
| **CFG 3.0** (chosen) | 19.4 | 22.4 | 0.57 | **17.4** |
| CFG 3 + flow_shift 5 | 19.4 | 22.8 | 0.52 | 16.6 |
| CFG 3 + flow_shift 10 | 19.4 | 22.8 | 0.54 | 16.1 |

- **CFG 3 is correct.** Sensitivity peaks there; fidelity and drift are flat
  across 1-3. Lowering CFG does **not** reduce artifacts — it only costs
  instruction sensitivity. (An earlier single-situation sweep suggested CFG 7
  might add a little more sensitivity; on the multi-scene set 3 is enough.)
- **flow_shift is a non-factor.** The sources conflict — the checkpoint's own
  `rectified_flow_inference_config.shift` and scheduler say **1**, the tech
  report says 5 for the deployed policy, NVIDIA's diffusers example says 10 —
  but measured, all three are identical on fidelity and drift, and the shipped
  default has marginally the best sensitivity. Keep the default; do not override.
- **Jitter is always below 1** (0.52-0.76). The rollouts move *less* than reality,
  not erratically more, so the classic over-guidance signature is absent.
- `use_system_prompt=False` is not a deviation: the checkpoint config sets
  `vlm_config.use_system_prompt = False` itself.

## The rollout never beats copying the input frame

Condition A, all scenes and cameras: mean |generated[t] - real[t]| against the
trivial baseline of leaving the source frame unchanged, |source - real[t]|.

| t | seconds | \|gen - real\| | \|src - real\| |
|---|---|---|---|
| 1 | 0.07 | 6.0 | **1.8** |
| 8 | 0.53 | 11.8 | **6.0** |
| 16 | 1.07 | 20.0 | **16.3** |
| 32 | 2.13 | 28.4 | **24.7** |

There is no crossover at any horizon. About 4.9 of every generated number is
unavoidable VAE reconstruction tax that a real frame does not pay — subtract it
and generation edges ahead by ~1-1.5, which is noise against real motion of ~20.

Two things this does and does not mean. It **does** say the rollout carries
little information about what actually happens next, which should temper any
claim that its counterfactual rollouts are physically meaningful. It does **not**
directly score the pipeline's goal: for a counterfactual instruction there is no
ground truth, and a correct subgoal *should* diverge from what really happened.
Fidelity here is a sanity check on condition A only.

## What's here

| file | what |
|---|---|
| `smoke_policy.py` | one call — proves a checkpoint works through diffusers |
| `prepare_situations.py` | multi-trajectory situation builder (one episode per situation, anchored at its first frame) |
| `rebuild_index.py` | recovers a run's `index.json` from the per-sample files after a late crash |
| `run_experiment.py` | the A/B/C/MP run over the DreamZero situation set (`--fps-sweep` for the fps check) |
| `seed_null.py` | measures the seed null for a finished run and records it in its `index.json` |
| `make_report.py` | HTML contact sheet, DreamZero layout: source + real future + generated per condition, 3 cameras per row |
| `sweep_general_nano.sh` / `sweep_super.sh` | the two parameter sweeps; `sweep_super.sh` adds a GPU preflight check because one Super run owns a whole card |
| `smoke_i2v.py` | one image-to-video call, with the checkpoint's own JSON caption schema built around the instruction |

Git-ignored: `.venv/`, `results/`, and the weights (Nano under
`~/proj/staging/vla/models/`, Super under `/raid/users/tiger/vla-distillation/models/`
— the home fs is at 98 % and cannot hold 133 GB).

## Reproduce

```sh
cd explorations/cosmos3

# 1. env (no Docker; cu128 to match our 570.x driver — CUDA 13 wheels will NOT run here)
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --torch-backend=cu128 torch torchvision
uv pip install --python .venv/bin/python \
    "diffusers @ git+https://github.com/huggingface/diffusers.git" \
    transformers accelerate imageio imageio-ffmpeg pillow numpy

# 2. weights (31 GB, self-contained)
HF_HOME=~/proj/staging/vla/models/hf_cache .venv/bin/python -c "
from huggingface_hub import snapshot_download as d
d('nvidia/Cosmos3-Nano-Policy-DROID', local_dir='$HOME/proj/staging/vla/models/Cosmos3-Nano-Policy-DROID')"

# 3. smoke, then the run + report  (situations come from the DreamZero set)
export CUDA_VISIBLE_DEVICES=<idle-gpu> HF_HOME=~/proj/staging/vla/models/hf_cache
.venv/bin/python smoke_policy.py
.venv/bin/python run_experiment.py --fps 15 --cameras exterior_1 exterior_2 wrist --run-name runs_allcams
.venv/bin/python seed_null.py --run runs_allcams      # the control every number is read against
.venv/bin/python make_report.py --run runs_allcams --open
```

## Plumbing that matters

- **`Cosmos3OmniPipeline.from_pretrained` works** even though `model_index.json`
  names the vLLM-Omni class `Cosmos3OmniDiffusersPipeline`; the vision-encoder
  key is ignored with a warning, which is harmless.
- **Action conditioning** goes through `CosmosActionCondition(mode="policy",
  chunk_size=16, domain_name="droid_lerobot", resolution_tier=256, image=...,
  view_point=...)` — passed as `action=`, *not* the top-level `image`/`height`/
  `width`, which must be `None` for action runs.
- **`droid_lerobot` is 10D**: 3D translation + 6D rotation of the end-effector
  pose **delta** (`ΔT = T⁻¹ₜ₋₁Tₜ`, OpenCV convention) + 1D absolute gripper.
  This is **not** DreamZero's absolute-joint-position space, and the checkpoint
  ships **no denormalization statistics** — so actions here are only ever
  compared *between conditions*, never against logged DROID actions.
- **`resolution_tier=256`** matches our 320×180 frames (the tier's bin is
  320×192; output comes back 320×176). The tier never upscales, so a higher tier
  only pads and wastes compute.
- **`view_point`** fills the caption's framing field: `third_person_view` for the
  DROID exteriors, `wrist_view` for the wrist camera.
- **Guardrails off** via `enable_safety_checker=False` at construction (nothing
  is downloaded, and the gated `nvidia/Cosmos-1.0-Guardrail` repo is never
  needed). Left on, the video guardrail **pixelates faces** — DROID scenes have
  people at the edges, so that would silently corrupt generated frames relative
  to the real ones we compare against. Recorded in every `index.json`.
- **No prompt upsampling.** Action mode builds its own structured JSON caption
  from the plain instruction, and the diffusers docs say action prompts should
  not be LLM-upsampled.

## Not done (deliberately)

- **The DROID OpenPI websocket path** (`vllm serve … --omni`, `/v1/realtime/robot/openpi`)
  — the only route that supplies proprioception + the three-view canvas and
  returns joint-position actions, i.e. the true apples-to-apples with DreamZero.
  Skipped because it returns no video and because vLLM-Omni's current wheels
  target CUDA 13, which our driver cannot run.
- **Image-to-video at scale.** `smoke_i2v.py` runs one I2V call on Super (17
  frames, 832x480 from a 320x180 source, 23.8 s) — scene identity holds and it
  upscales, but the camera drifts despite the caption demanding a locked-off
  tripod, and the instructed manipulation is not visibly executed. One sample and
  one seed proves nothing; I2V has never been run through the A/B/C harness,
  which is the only way to separate "ignores language" from "this seed did that".
- **Multi-seed averaging.** With the seed null above the instruction effect, a
  single seed per condition cannot resolve a real effect; averaging N seeds per
  condition would.
- In-distribution hand-written instructions (long, descriptive, JSON-shaped).

---

# General Cosmos3-Nano (not DROID-post-trained)

Same 8 trajectories at frame 0, same conditions, same cameras — so every number
below is directly comparable to the Policy-DROID tables above. Reports:
`results/sweep_general_nano/` (11 configs). Run with `sweep_general_nano.sh`.

| config | floor | fidelity | jitter g/r | drift | sens B | seed null |
|---|---|---|---|---|---|---|
| cfg=1 | 5.0 | 23.6 | 1.16 | 21.4 | 16.7 | — |
| cfg=2 | 5.1 | 24.1 | 1.21 | 20.6 | 21.9 | — |
| **cfg=3** (baseline) | 5.1 | 24.6 | 1.26 | 20.1 | 24.1 | 22.2 |
| cfg=7 | 5.1 | 27.7 | 1.42 | 22.4 | 31.8 | — |
| steps=4 | 5.0 | 25.5 | **1.09** | 21.6 | 26.3 | — |
| steps=50 | 5.1 | 24.8 | 1.26 | 20.0 | 24.0 | — |
| chunk=16 | 5.1 | 22.2 | 1.57* | **16.1** | 17.6 | — |
| **concat3view** | **4.6** | 24.7 | **1.06** | **16.3** | 17.7 | 21.5 |
| upscale=2 | **3.1** | 28.3 | 1.50 | 27.2 | 29.7 | 46.4 |
| upscale=2 + chunk=16 | 3.1 | 30.5 | 2.81* | 29.6 | 34.0 | — |
| upscale=2 + cfg=2 | 3.1 | 28.4 | 1.49 | 27.3 | 25.7 | — |

\* the two `chunk=16` rows measure jitter against a different real baseline
(4.56 vs 5.76) because a shorter horizon contains less real motion — not
comparable to the 32-frame rows.

---

# Cosmos3-Super (64B, general — no Super-tier Policy-DROID exists)

Same 8 trajectories, same A/B/C conditions, same three cameras as both Nano
tables above, so every row is directly comparable. Reports:
`results/sweep_super/` (6 configs). Run with `sweep_super.sh`.

**It fits on one H200.** ~131 GB of BF16 weights, **peak 135 GB** of 143.7 — no
tensor parallelism, no `torchrun`, and the pinned `diffusers 0.40.0.dev0` runs it
unchanged. The diffusers docs say Super "does not fit on one 96 GB GPU, so it
needs TP"; that is calibrated to 96 GB cards and does not apply here. Load 30 s.
Latency scales with **canvas area, not parameter count**: 6.9 s/gen at tier 256,
27 s at concat 480, 66 s at concat+upscale 720.

The grid is smaller than the Nano sweeps on purpose — `flow_shift`, steps 30-vs-50,
`resolution_tier` alone and karras were already measured dead on both 16B models.

| config | CFG | floor | fidelity | jitter g/r | drift | sens B | seed null | sens / null |
|---|---|---|---|---|---|---|---|---|
| cfg=1 | 1 | 5.1 | 26.2 | 1.33 | 20.2 | 18.2 | 30.8 | 61 % |
| **cfg=3** (baseline) | 3 | 5.1 | 28.5 | 1.45 | 21.8 | 26.7 | 23.9 | **119 %** |
| cfg=7 | 7 | 5.1 | 31.3 | 1.46 | 23.5 | 32.1 | 27.3 | **128 %** |
| concat3view | 3 | 4.7 | **38.9** | 1.45 | **30.1** | 28.4 | 19.9 | **186 %** |
| upscale=2 | 3 | **3.1** | 26.7 | 1.40 | 26.2 | 26.1 | 20.0 | **148 %** |
| concat+upscale=2 | 3 | 4.4 | 28.0 | **1.28** | 27.3 | 24.9 | 21.5 | **123 %** |

- **Sensitivity clears the seed null in five of six configs.** This is the first
  configuration measured here where swapping the instruction moves the rollout
  decisively more than re-rolling the seed.
- **B/C finally separates.** `sens C` > `sens B` in all six configs, and the gap
  widens monotonically with guidance (C/B = 1.07 → 1.13 → 1.17 at CFG 1/3/7). At
  16B the ordering was inconsistent and therefore uninterpretable.
- **`concat3view` inverts** — see "Conclusions that transfer" below.
- **`upscale=2` also inverts:** it *hurt* fidelity on Nano (28.3 vs 24.6) but
  *helps* on Super (26.7 vs 28.5), while giving the same best-in-class floor of
  3.1. `concat+upscale=2`, untested on any model before, gives Super's
  closest-to-real motion (1.28) but beats the baseline on nothing else.
- Floors match the Nano runs to within 0.1 across the board — expected, since all
  three checkpoints share the `AutoencoderKLWan` VAE, and a useful check that the
  metric is measuring what it claims.

## The three models sit on one axis: faithful <-> responsive

| | Policy-DROID 16B | general Nano 16B | general Super 64B |
|---|---|---|---|
| jitter vs reality | **0.52-0.89 — under-moves** | 1.06-1.50 — over-moves | **1.28-1.46 — over-moves** |
| fidelity to real future | **19.4 (best)** | 24.6 | 26.2-31.3 (worst) |
| sensitivity / own seed null | 77 % | 103 % | **119-186 %** |

DROID post-training bought physical faithfulness, not language grounding.
Scaling the generalist 16B -> 64B moves *further* along the same axis: more
responsive to the instruction, less faithful to what actually happened. Nothing
here gives both, so the choice still depends on whether Stage B needs a
plausible image or a responsive one — but Super is the first configuration
measured where the instruction beats the seed decisively.

## Conclusions that transfer

- **`concat3view` wins on both 16B models — and is the WORST config on Super.**
  On the Nanos the stitched three-view canvas (the layout the DROID policy was
  post-trained on) gives the closest-to-real motion measured (jitter 1.06
  general / 0.89 specialist) and roughly halves drift. On Super it inverts:
  worst fidelity (38.9 vs 28.5 baseline) and worst drift (30.1 vs 21.8), with no
  jitter benefit. It buys the highest sensitivity of any run (186 % of null) by
  being unfaithful. **This was written up as "a two-model result, not a fluke" —
  the third model breaks it.** Whatever the canvas does, it is not a property of
  Cosmos 3 in general, and the layout argument does not explain Super, which was
  never DROID-post-trained at all.
- **CFG is a clean monotonic dial on both generalists, inert on the specialist.**
  General Nano: guidance 1→7 drives jitter 1.16→1.42, sensitivity 16.7→31.8.
  Super: sensitivity 18.2→26.7→32.1 at CFG 1/3/7 — nearly the same ladder, at 4x
  the parameters. On Policy-DROID it changed fidelity not at all.
- **`upscale` is the only image-quality knob, and it costs stability.** Output
  resolution is `min(input, tier canvas)` — `resolution_tier` alone does nothing
  because it never upscales. 2x gives the best reconstruction floor of any run
  (3.1) but the worst drift and a seed null that doubles to 46. Two attempts to
  tame it (`+chunk=16`, `+cfg=2`) both failed; `+chunk=16` made it far worse.
- **Sampling knobs are mostly dead ends:** `flow_shift` (no effect at 1/5/10 on
  either model, despite docs recommending 5-10), `steps` 30 vs 50 (nothing),
  `use_karras_sigmas`, `use_system_prompt` (already matches the checkpoint).
  Exception: `steps=4` unexpectedly *reduced* over-motion on the generalist
  (1.09) while raising sensitivity — unexplained, worth a visual check.
- **The seed null is not a constant.** It doubled (22 -> 46) when resolution
  changed, so sensitivity is only interpretable against the null of the *same*
  config. Raw sensitivity comparisons across configs are misleading.

## Where to pick up

Untested, in rough order of expected value:

1. **I2V through the A/B/C harness.** `run_experiment.py` hardcodes
   `mode="policy"`, where video is a by-product of an action model. I2V is what a
   general checkpoint is actually built for and is the honest Stage-B analogue;
   `smoke_i2v.py` shows it runs and what a caption has to look like. Costs: no
   action chunk, and captions must be JSON-upsampled rather than raw imperatives
   (an LLM rewrite between our instruction and the model — log it deliberately).
2. **Why `concat3view` inverts at 64B.** It is the best lever on both 16B models
   and the worst config on Super. Until that is explained, no layout conclusion
   from this harness should be trusted to generalise.
3. **`concat3view` + `steps=4`** — stacks the two jitter reducers on Nano. Cheap.
4. **Multi-seed averaging.** One seed per condition cannot resolve an effect that
   sits at or below the seed null; averaging N seeds would. Less pressing on
   Super, where the effect is comfortably above its null.
5. Seed nulls for the Nano configs that lack them (~1 min each).

## Gotchas that have bitten us

- `seed_null.py` rebuilds its config from `index.json` and does **not** inherit
  new flags automatically. It has silently mismatched geometry (`upscale`) and
  crashed on layout (`concat_view`). Both fixed, but check it when adding a flag.
- Cross-resolution **fidelity is not comparable** — `diff()` resamples onto a
  common grid. Trust the reconstruction floor, jitter ratio, and
  sensitivity-over-own-seed-null instead.
- **Never hand `export_to_video` an ndarray.** It branches on type and treats an
  ndarray as float `[0,1]`, multiplying by 255 — uint8 wraps modulo 256 into
  garbage that still encodes and still plays. Four sweep configs were silently
  corrupted this way. Frames must stay PIL through `save_sample`.
- **The reconstruction floor is a corruption canary.** It should sit near 5 (it
  is VAE loss, shared across all three checkpoints). ~130 means the pixels never
  survived encoding, not that the model did badly.
- **Odd frame dimensions kill ffmpeg.** libx264 + yuv420p needs even width and
  height; `macro_block_size=1` does not cover that. `split_canvas` thirds are
  routinely odd (concat+upscale at 720 gives 496x277) — `_even_dims` crops them.
- Reports embed images, so each is ~20-30 MB; whole folders are ~300 MB.
