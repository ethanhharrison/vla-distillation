# DreamZero exploration — combined subgoal-video + action generator

**Status: exploration / evidence, not integrated.** We evaluated
[DreamZero-DROID](https://huggingface.co/GEAR-Dreams/DreamZero-DROID) (NVIDIA GEAR
Lab, 14B, Wan2.1-based World Action Model) as a single model that could replace
Stages **B (image)** and **C (action)** at once: given a DROID observation
(3 cameras + proprioception + a language instruction) it jointly predicts
**future video** and a **24-step action chunk**.

Full write-up (findings, latency, honest assessment): `~/proj/notes/vla/SESSION_DREAMZERO.md`.

## TL;DR findings
- **Runs on one GPU here** (H200, ~44.6 GB VRAM, ~3.8 s/inference steady after warmup).
- **Action calibration is good**: predicted chunk vs the episode's logged future
  joint positions ≈ **0.05 rad MAE**.
- **But actions barely follow the instruction** in an open-loop single-step probe:
  a counterfactual or even impossible instruction ("fold the laundry") yields
  nearly the same action chunk as the real task (~0.02–0.08 rad divergence). The
  **video** follows the instruction somewhat more.
- **Recommendation:** promising as a *video-subgoal* generator; **not yet** a
  trustworthy combined action labeler. Verify **closed-loop** (its native regime)
  before adopting.

---

## What's in this directory
Committed (small, our code):
| file | venv | what |
|---|---|---|
| `serve.sh` | dreamzero | launch the single-GPU inference server |
| `prepare_situations.py` | **main** | extract droid_100 situations via our RLDS adapter |
| `prepare_instructions.py` | **main** | A/B/C instruction conditions (Gemini for B) |
| `run_experiment.py` | dreamzero | drive the server, collect action chunks + video |
| `make_report.py` | **main** | HTML contact sheet + condition-A action calibration |

Git-ignored (large / external — recreate with the steps below):
`repo/` (upstream clone), `.venv/` (server venv), `results/` (situations, runs,
videos, report), and the weights under `~/proj/staging/vla/models/`.

**Two venvs on purpose:** the *main* `vla-distillation/.venv` has
`tensorflow-datasets` + our DROID RLDS adapter (data prep / plotting); the
*server* venv here has the heavy DreamZero/torch stack. They exchange plain files
under `results/`, so neither needs the other's deps.

---

## Reproduce from scratch

Prereqs: an idle GPU with ~50 GB free VRAM (single GPU is fine), ~120 GB disk for
weights, and the public `droid_100` RLDS already downloaded (see the main README /
`~/proj/notes/vla/DROID_RLDS_NOTES.md`).

### 1. Clone the upstream repo (pinned)
```bash
cd explorations/dreamzero
git clone https://github.com/dreamzero0/dreamzero repo
git -C repo checkout ab790c198fbce33503358efbbd4187ce9a89adf3   # the commit we used
```

### 2. Create the server venv and install — with our fixes
The upstream install docs assume conda + CUDA 12.9 + compiled flash-attn. On this
box (driver = **CUDA 12.8**, `nvcc` 12.0, no-compile policy) use:
```bash
uv venv .venv --python 3.11
cd repo
VIRTUAL_ENV=../.venv uv pip install -e . \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
cd ..
```
Three gotchas, already handled by the flags above + this step:
- **cu128, not cu129** — matches the driver; `torch==2.8.0+cu128` installs and runs.
- **`--index-strategy unsafe-best-match`** — otherwise uv pins `requests` from the
  PyTorch index and `datasets==3.6.0` can't resolve.
- **Skip flash-attn entirely.** It's a from-source CUDA compile; the model falls
  back to PyTorch SDPA (guarded imports in `wan_video_dit.py`). Do **not** run the
  repo's `pip install flash-attn` step.
- **Remove deepspeed** (training-only) — it triggers a circular import with
  `transformers 4.51.3` at server startup:
  ```bash
  VIRTUAL_ENV=.venv uv pip uninstall deepspeed
  ```

### 3. Download the checkpoint
```bash
# skip the GB200-only TensorRT/ONNX artifacts (~19 GB)
HF_HOME=~/proj/staging/vla/models/hf_cache \
python -c "from huggingface_hub import snapshot_download as d; \
d('GEAR-Dreams/DreamZero-DROID', repo_type='model', \
  local_dir='$HOME/proj/staging/vla/models/DreamZero-DROID', \
  ignore_patterns=['tensorrt/*','*.onnx','*.onnx_data'])"
```
**Note:** the checkpoint (~43 GB) is *not* self-contained — at first serve it
auto-downloads the full `Wan-AI/Wan2.1-I2V-14B-480P` (~77 GB) + `google/umt5-xxl`
into the HF cache. Set `HF_HOME` (e.g. to staging) *before* serving so it doesn't
fill `~/.cache`. Real inference footprint ≈ **120 GB**.

### 4. Serve (single GPU) and verify
```bash
bash serve.sh <gpu> 8901          # model load + warmup ~2-3 min
# verify with the repo's own test client before our data:
.venv/bin/python repo/test_client_AR.py --host localhost --port 8901 --num-chunks 2
```
`serve.sh` runs `torch.distributed.run --nproc_per_node=1` (single GPU works even
though the docs say "min 2 GPUs"; the server is `world_size`-driven). Stop it when
done to free the GPU: `pkill -f "socket_test_optimized_AR.py --port 8901"`.

### 5. Run the experiment + report
```bash
# (main venv) prepare inputs
source ../../.venv/bin/activate
python prepare_situations.py --episode-index 0 --num-situations 5
SIT=results/situations/<episode-id>
python prepare_instructions.py --situations "$SIT"    # A/B/C (+ edit for more)

# (server venv) drive the model
../.venv/bin/python run_experiment.py --situations "$SIT" --port 8901 \
  --model-path ~/proj/staging/vla/models/DreamZero-DROID

# (main venv) build the contact sheet
python make_report.py --situations "$SIT"             # -> results/runs/report.html
```

---

## Input plumbing that matters (documented so you don't misfeed it)
- **Camera-key remap (server ← DROID)** — note the 0-vs-1 index trap:
  `observation/exterior_image_0_left` ← DROID `exterior_image_1_left`;
  `observation/exterior_image_1_left` ← DROID `exterior_image_2_left`;
  `observation/wrist_image_left` ← `wrist_image_left`.
- The model composes views into a **2×2 grid**: top row = wrist (2× wide),
  bottom-left = exterior_1, bottom-right = exterior_2. `make_report.py` splits the
  generated grid back into the 3 cameras using this layout.
- AR / stateful: `session_id` per call; 4-frame history window at offsets
  `[-23,-16,-8,0]`; images `(180,320,3)` uint8 RGB; proprio
  `joint_position[7]` + `cartesian_position[6]` + `gripper_position[1]`.
- **Output:** `(24, 8)` = 7 **joint positions** + 1 gripper, radians-scale absolute
  (`action_space="joint_position"`). Video is emitted block-by-block (2 latent
  frames/call ≈ ~8 decoded RGB frames, ~5 fps) and saved as MP4 on `reset()`.
