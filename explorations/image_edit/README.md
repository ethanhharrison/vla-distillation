# Hosted image-edit exploration — the A/B/C protocol for Stage B's paid backends

**Status: exploration / evidence, not integrated.** Both models have now been run
over the full `situations_multitraj` grid — the same 8 trajectories, conditions
and cameras as the Cosmos 3 and DreamZero tables — so the numbers below are
directly comparable to theirs. Single draw per condition; see the caveats.

Puts `pipeline/subgoal_image`'s hosted editors — Gemini and OpenAI — on the same
protocol as [`../cosmos3`](../cosmos3) and [`../dreamzero`](../dreamzero): the
same situation sets, the same A/B/C conditions, the same units. Until now the
world models had a quantitative harness with controls and the paid editors had
an HTML contact sheet, so "is Gemini's edit better than a Cosmos rollout?" was
not an answerable question. It is now.

This harness **calls the pipeline's own backends**, so it measures the code Stage
B would run in production rather than a reimplementation of it. It needs no venv
of its own and no weights — run it from the main `.venv`.

## Numbers

Mean absolute pixel difference (0–255), 8 situations x 3 cameras, conditions
A + 2xB + 2xC + N, single draw each. 48 samples x 3 cameras = **144 edits per
model** (`results/{gemini_flash_3_1_full,openai_gpt_image_2_full}`).

| | `gemini-3.1-flash-image` | `gpt-image-2` (low) |
|---|---|---|
| **no-op floor** (told to change nothing) | **3.7** | 5.6 |
| **edit magnitude** (A vs source) | 5.7 | **9.2** |
| real motion (real t+k vs source) | 24.7 | 24.7 |
| **fidelity** (A vs real t+k) | **25.8** | 28.7 |
| sensitivity B · plausible counterfactual | 7.1 | 9.8 |
| sensitivity C · impossible | 8.6 | 11.7 |
| **resample null** | 6.0 | 10.1 |
| **sensitivity / own null** | **132 %** | 107 % |
| cost · failures | $11.67 · 3/144 | **$0.89** · 0/144 |

- **Both models read the instruction.** Sensitivity clears the resample null for
  both, which puts them above the two 16B world models (77–103 % of their seed
  nulls) and in the range Cosmos3-Super reached (119–186 %). And **C > B on both**,
  consistently — impossible instructions perturb the output more than plausible
  ones, the ordering that only Super achieved among the video models.
- **But both barely edit.** Against real motion of 24.7 over the same span, the
  subgoals move the scene 5.7 and 9.2 — and their own no-op floors are 3.7 and
  5.6, so only about **2 and 3.6 points of that movement is attributable to the
  instruction at all**. Most of what looks like an edit is re-render tax. This is
  the quantitative version of the note in the main README that `gemini_image`
  under-edits exterior views; it turns out to apply to both providers.
- **Neither beats copying the source frame.** Fidelity (25.8, 28.7) is *worse*
  than real motion (24.7) — which is exactly |source − real|. So the edited
  subgoal is further from what actually happened than the unedited source frame
  is. The Cosmos 3 rollouts failed the same test, and the same caveat applies:
  for a counterfactual instruction there is no ground truth and a correct subgoal
  *should* diverge, so this scores condition A only.
- **The 13x price gap buys nothing here.** `gpt-image-2` at `low` edits more (9.2
  vs 5.7), costs $0.89 against $11.67, and refused nothing, where Gemini returned
  `FinishReason.NO_IMAGE` on 3 of 144 calls. Gemini's advantage is a lower floor
  (3.7 vs 5.6) and a higher sensitivity ratio, both of which follow from it
  perturbing the image less overall.

**Caveats that bound all of this.** One draw per condition, so an effect near the
null cannot be resolved — the same limitation the Cosmos 3 README flags, and the
reason `--repeats` exists on the null. The null itself is measured on one
situation across 3 cameras. The no-op floor uses a different prompt from the
subgoal templates, so it bounds the re-render tax rather than measuring it under
the exact prompt. `gpt-image-2` was run at `quality=low`; medium and high are
6x and 24x the price and untested.

## What changes when the model emits an image instead of a clip

| video harnesses | here | why |
|---|---|---|
| generated frames 0..H | one image | an editor jumps straight to the subgoal |
| jitter (frame-to-frame motion) | — | no clip to be jittery |
| drift (error growth over the rollout) | — | no horizon |
| VAE reconstruction floor | **no-op floor** (`--noop`) | an editor is not an autoencoder round-trip. The analogous tax is how far it moves the scene when told to change *nothing* |
| **seed null** (same instruction, new seed) | **resample null** | hosted APIs expose no seed and resample every call. Identical role: the bar an instruction effect has to clear |
| action MAE vs condition A | — | an editor emits no actions — so unlike DreamZero it can only ever be a Stage B candidate, never a combined B+C one |
| fidelity, `vs src`, `vs A` | unchanged | mean abs pixel difference, 0–255, directly comparable to the sibling tables |

Two things that are *not* differences: the situations and the conditions. A, B
(Gemini-proposed counterfactuals), C (authored impossible tasks) and MP come out
of the same `conditions.json` the video harnesses read, untouched.

## The two controls, and why both are needed

- **No-op floor (condition `N`).** One extra call per camera asking the model to
  reproduce the frame unchanged. Hosted editors re-render the whole image at a
  new resolution, so `vs src` is never zero even for a perfect no-op; without
  this bound you cannot tell a real edit from re-rendering. Opt-in, because it is
  a real 20–25 % surcharge on a run. When it is missing, `analyze.py` and the
  report both say so rather than quietly reporting an inflated edit magnitude.
- **Resample null.** The same request asked twice. `resample_null.py`
  **bypasses the edit cache on purpose** — an identical request is precisely what
  the cache serves for free, so a cached null would read 0.0 and rubber-stamp
  every instruction effect as significant. This is the one part of a rerun that
  is never free.

## Verified against the live APIs (2026-08-20)

Model lists come from each provider's `models.list()`; costs and latencies are
billed measurements on a real 320x180 DROID frame, not list-price arithmetic.

| model | $/image | latency | output size |
|---|---|---|---|
| `gpt-image-2` quality=low | **$0.0060** | 19 s | 1672x941 |
| `gpt-image-2` quality=medium | $0.0369 | 38 s | 1672x941 |
| `gpt-image-2` quality=high | $0.1411 | 100 s | 1672x941 |
| `gemini-3.1-flash-image` | $0.0833 | 8 s | 1365x768 |

Consequences worth knowing before planning a run:

- **`gpt-image-2` at `low` is ~14x cheaper than `gemini-3.1-flash-image`**, which
  inverts the cost assumption in `notes/vla/DESIGN.md` (written when hosted
  editing was ~$0.04/image across the board and the two providers were close).
- **On OpenAI, quality is the dominant cost *and* latency knob** — a 3-camera
  8-situation grid is ~13 min and $0.77 at `low`, but ~3.3 h and $17 at `high`.
- Both models return images 4–5x larger than the DROID source. `metrics.diff`
  therefore compares on the smaller grid and is symmetric; upsampling the
  reference would invent detail and score it as model error.

The full `situations_multitraj` grid is 8 situations x 5 instructions x 3 cameras
= **120 edits** (144 with `--noop`). `--dry-run` prints the projection, and a run
whose projection exceeds `--ceiling` **refuses to start** rather than aborting
halfway and leaving a run that cannot be analysed.

## What's here

| file | what |
|---|---|
| `run_experiment.py` | the A/B/C/(MP)/(N) run over a situation set; one run = one backend + model + template |
| `resample_null.py` | the control every sensitivity number is read against |
| `make_report.py` | HTML contact sheet, cosmos3 layout: source + real future + subgoal per condition, 3 cameras per row |
| `analyze.py` | the cross-run table (`noop / edit / real / fidelity / sensB / sensC / null`) |
| `metrics.py` | the pixel metrics, in one place so the three scripts cannot disagree |

Situation sets and `conditions.json` are **not** rebuilt here — reuse
`../cosmos3/prepare_situations.py` and `../dreamzero/prepare_instructions.py`.

Git-ignored: `results/`. Reports embed their images, so each full-grid report is
~20 MB and the whole `results/` folder is ~380 MB — the same order as the cosmos3
reports, and worth watching given the home fs is at 100 %.

## Reproduce

```sh
cd explorations/image_edit                     # main venv; no weights, no GPU
P=../../.venv/bin/python

# 0. what would it cost?
$P run_experiment.py --backend openai_image --dry-run

# 1. the run  (add --noop for the floor; --limit-situations/--cameras to trim)
$P run_experiment.py --backend openai_image --run-name openai_gpt_image_2 \
    --situations ../cosmos3/results/situations_multitraj --noop --ceiling 2

# 2. the control, then the outputs
$P resample_null.py --run openai_gpt_image_2 --ceiling 0.20
$P make_report.py   --run openai_gpt_image_2 --open
$P analyze.py --runs openai_gpt_image_2 gemini_flash_3_1
```

Reruns are $0: edits are served from the Stage B content-addressed cache
(`outputs/subgoal_images/cache`, shared with the pipeline). Only
`resample_null.py` pays again, by design.

## Sanity checks the harness passes

Both free backends are exact self-tests of the metric, and both are worth
re-running after any change here:

- `--backend real_future` scores **fidelity 0.0** and `edit == real motion`, because
  its "subgoal" *is* the real future frame.
- `--backend dummy_image` scores **sensitivity exactly 0.0** and `noop == edit`,
  because its edit ignores the prompt. Any prompt-blind model should look like
  this — which is the failure mode this whole harness exists to detect.

## Gotchas

- **`gpt-image-2` rejects `input_fidelity`** with a 400 (`the model does not
  support the 'input_fidelity' parameter`) — it always processes inputs at high
  fidelity. `gpt-image-1-mini` never supported it either. `backends.py` gates on
  an explicit model set, not a substring of the name.
- **A cached resample null is a silent 0.0.** See above; `use_cache=False` is
  load-bearing, and `tests/test_subgoal_edit.py` pins it.
- **The ceiling is checked before each call, never after**, so it cannot be
  exceeded even once — but that means a too-low ceiling aborts mid-run. Use
  `--dry-run` first.
- **`estimate_cost()` and the billed cost are different numbers by design.** The
  first is a static per-image upper bound that gates the ceiling; the second is
  derived from the token usage the API reports and is what lands in
  `costs.jsonl`. On the reference call they are $0.010 vs $0.0064.

## Not done

Untested, in rough order of expected value:

1. **Multi-draw conditions.** One draw per condition cannot resolve an effect
   sitting near the null, and here the effects are only 1.1–1.3x it. Averaging N
   draws per condition is the single thing that would firm up every row above.
   `resample_null.py --repeats` already averages the *null*; the conditions do not.
2. **A prompt that actually asks for a bigger change.** Both models land barely
   above their no-op floor, so the headline result may be as much about the Stage B
   template as about the models. Stage B ships `default`, `minimal` and
   `object_centric`; one run per template puts them on this protocol, which is
   exactly what `prompts.py` tuning has never had. Cheap on `gpt-image-2 low`.
3. **`gpt-image-2` at medium/high** — 6x and 24x the price, and the only lever
   that changed anything measurable in the pre-run probes.
4. **`gemini-3-pro-image`** — available, ~$0.18/image, unmeasured.
5. Per-camera breakdown. The wrist view dominates the null in both video
   harnesses; the aggregates above may be hiding the same effect here.
