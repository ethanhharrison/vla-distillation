# VLA Distillation

Tooling for building a language-instruction dataset from [DROID](https://droid-dataset.github.io/)
robot trajectories. Given a trajectory (stored as a TFRecord), the pipeline
samples frames along the trajectory and prompts a Vision-Language Model (VLM) to
propose natural-language instructions the robot could accomplish starting at each
sampled step.

## Project layout

```
vla-distillation/
├── scripts/
│   ├── download_dataset.py     # download DROID TFRecords from GCS
│   └── view_dataset.py         # inspect a TFRecord's structure / dump frames
├── pipeline/
│   └── language_instruction/   # instruction-generation pipeline
│       ├── trajectory.py       # decode a TFRecord into per-step camera frames
│       ├── vlm.py              # swappable VLM backends (OpenAI / Gemini / Dummy)
│       ├── prompts.py          # prompt template + response parsing
│       ├── pricing.py          # token accounting + approximate USD cost estimation
│       └── generate.py         # orchestration + CLI
├── datasets/                   # downloaded TFRecords (git-ignored)
└── outputs/                    # generated instructions + saved frames (git-ignored)
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/). Create the environment and
install dependencies:

```bash
uv venv
uv pip install \
  google-cloud-storage tqdm tensorflow \
  openai google-genai python-dotenv
```

Run any command in the environment with `uv run ...` (examples below).

### API keys

The VLM backends read credentials from environment variables. You can export
them directly or place them in a `.env` file at the project root (auto-loaded via
`python-dotenv`):

```bash
# .env
GEMINI_API_KEY=your-gemini-key      # or GOOGLE_API_KEY
OPENAI_API_KEY=your-openai-key
```

> Get a Gemini key from [Google AI Studio](https://aistudio.google.com/apikey).
> If you hit `API_KEY_SERVICE_BLOCKED`, make sure the **Generative Language API**
> is enabled for the key's project and the key has no conflicting API
> restrictions.

## 1. Download TFRecords

DROID TFRecords live in a Google Cloud Storage bucket. Downloading requires GCP
credentials — authenticate once with Application Default Credentials:

```bash
gcloud auth application-default login
```

Then download some records:

```bash
uv run python scripts/download_dataset.py
```

By default this downloads **3 randomly selected** success records from
`gs://pranav-us-east5/datasets/droid/success/` into `datasets/droid/success/`.

To change how many (or which) records are pulled, call `download_droid_records`
directly:

```bash
# Download 10 records (no shuffle = deterministic first 10)
uv run python -c "from scripts.download_dataset import download_droid_records; download_droid_records(10, shuffle=False)"
```

Arguments to `download_droid_records`:

| Argument       | Default                      | Description                              |
| -------------- | ---------------------------- | ---------------------------------------- |
| `num_records`  | (required)                   | Number of TFRecords to download.         |
| `bucket_name`  | `pranav-us-east5`            | GCS bucket to pull from.                 |
| `droid_folder` | `datasets/droid/success/`    | Folder prefix within the bucket.         |
| `shuffle`      | `True`                       | Randomly sample records before slicing.  |

### (Optional) Inspect a record

To see the structure of a TFRecord (feature names, types, sample values) and
dump the first frame of each camera to `images/`:

```bash
uv run python scripts/view_dataset.py
```

## 2. Generate language instructions

Walk a trajectory at a configurable step interval and prompt a VLM at each
sampled step:

```bash
uv run python -m pipeline.language_instruction.generate \
  datasets/droid/success/success-00188.tfrecord \
  --provider gemini \
  --step-interval 25 \
  --num-instructions 3 \
  --save-images
```

This writes a text file to `outputs/language_instructions/` and (with
`--save-images`) the queried frames to `outputs/language_instruction_images/<record>/`.

### CLI options

| Flag                  | Default             | Description                                                        |
| --------------------- | ------------------- | ----------------------------------------------------------------- |
| `record`              | (required)          | Path to a `.tfrecord` file.                                        |
| `--provider`          | `gemini`            | VLM backend: `gemini`, `openai`, or `dummy`.                       |
| `--model`             | provider default    | Model name (e.g. `gpt-4o`, `gemini-2.0-flash`).                    |
| `--step-interval`     | `25`                | Sample and prompt every N steps of the trajectory.                |
| `--num-instructions`  | `3`                 | Number of candidate instructions to request per step.             |
| `--cameras`           | all three cameras   | Which camera image features to send to the VLM.                   |
| `--max-steps`         | `None`              | Only consider steps up to this index (useful for quick runs).     |
| `--example-index`     | `0`                 | Which example within the TFRecord to use.                          |
| `--output`            | auto-named          | Output `.txt` path.                                                |
| `--save-images`       | off                 | Save the camera frame(s) at each queried step.                    |
| `--image-dir`         | auto (per record)   | Where to save queried-step frames.                                 |
| `--judge`             | off                 | Score candidates with a VLM judge and drop low-scoring ones.       |
| `--judge-provider`    | `--provider`        | VLM backend for the judge.                                         |
| `--judge-model`       | provider default    | Model name for the judge.                                          |
| `--judge-threshold`   | `3`                 | Minimum judge score (1-5) required to keep an instruction.         |
| `--estimate-cost`     | off                 | Estimate the run's approximate USD cost from token usage.          |

The two most important knobs:

- **`--provider` / `--model`** — swap which VLM is used.
- **`--step-interval`** — how far apart (in trajectory steps) the sampled frames
  are. Nearby frames look nearly identical, so a larger interval yields more
  distinct scenes.

### Output format

Each run produces a text file summarizing the run configuration followed by the
per-step instructions (and, if `--save-images` is set, the saved frame paths):

```
record: datasets/droid/success/success-00188.tfrecord
provider: gemini
model: gemini-2.0-flash
step_interval: 25
...
============================================================
[step 0]
  - pick up the measuring tape
  - move the arm toward the drawer
  - open the top drawer
  (image) shoulder_image_1: outputs/language_instruction_images/success-00188/step0000_shoulder_image_1.jpeg
  ...
```

## Estimating cost

Pass `--estimate-cost` to record an approximate USD cost for the run (works with
or without `--judge`):

```bash
uv run python -m pipeline.language_instruction.generate \
  datasets/droid/success/success-00285.tfrecord \
  --provider gemini --judge --estimate-cost
```

The generator accumulates the input/output token counts reported by each API
call (image tokens are already included in the input counts) and multiplies them
by the per-model prices in `pipeline/language_instruction/pricing.py`. Both the
generation model and, when enabled, the judge model are counted. The headline
metric is **cost per step** — the total run cost divided by the number of
trajectory steps at which we asked for a new set of instructions:

```
cost per step = total cost / number of generation steps
```

These fields are written into the run `.txt` header:

```
generation_cost_usd: 0.004120
judge_cost_usd: 0.002980
estimated_cost_total_usd: 0.007100
estimated_cost_per_step_usd: 0.000394
```

Prices are approximate list prices and drift over time — edit `MODEL_PRICING`
in `pricing.py` (USD per 1M tokens, `(input, output)`) to keep them current. A
model with no pricing entry (or the offline `dummy`/local `hf` backends) reports
an "unknown" cost rather than a wrong one.

The two viewer scripts surface these numbers automatically:

- `summarize_language_instructions.py` renders an **Estimated cost** card.
- `compare_language_instructions.py` adds a **cost / step** column to the models
  table, making it easy to weigh quality against price when comparing models.

Runs generated without `--estimate-cost` simply omit the cost display.

## Choosing / adding a VLM backend

Backends live in `pipeline/language_instruction/vlm.py`. Each implements a single
method, `generate(prompt, images) -> str`, and registers itself under a provider
name. Built-in backends:

- `openai` — GPT models via the `openai` SDK
- `gemini` — Gemini models via `google-genai`
- `dummy` — offline stub that returns canned instructions

Add a new provider by subclassing `VLM` and decorating it:

```python
from pipeline.language_instruction.vlm import VLM, register_vlm

@register_vlm("myprovider")
class MyVLM(VLM):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        # set up your client here

    def generate(self, prompt: str, images: list[bytes]) -> str:
        ...  # return the model's raw text response
```

It is then selectable via `--provider myprovider`.

## Programmatic use

```python
from pathlib import Path
from pipeline.language_instruction import GenerationConfig, generate_instructions

config = GenerationConfig(
    record_path=Path("datasets/droid/success/success-00188.tfrecord"),
    provider="gemini",
    step_interval=25,
    num_instructions=3,
    save_images=True,
)
result = generate_instructions(config)
for step in result.steps:
    print(step.step, step.instructions, step.image_paths)
```

`generate_instructions` returns structured `StepInstructions(step, instructions,
raw_response, image_paths)` objects rather than only writing text, which makes it
straightforward to feed the results into downstream tooling.

## Roadmap

- **Instruction verification**: prompt a VLM with each generated instruction and
  its corresponding step frames to judge whether the instruction is achievable
  and well-grounded. The structured `(step, instructions, image_paths)` output is
  designed to be consumed directly by this future verification pass.
