# Dense-description prompt-tuning sets (v1)

Scene-disjoint train/test clips for iterating on the dense-description prompt.
Tune against **train**; run **test** only to check a finished prompt.

| split | clips | scenes |
|---|---|---|
| train | 40 | IPRL bedroom, TRI stove kitchen, TRI window kitchen, IRIS wood-table apartment, IRIS granite kitchen |
| test | 30 | CLVR office (whole lab unseen in train), IPRL kitchen, TRI sink kitchen, IRIS white cabinets, IRIS round table |

- **Clips**: 4s (60 frames @ 15fps), one per episode, from the local DROID
  success shards (`datasets/droid/success/*.tfrecord`, labs CLVR/IPRL/IRIS/TRI).
- **Events**: each clip is picked from the robot's gripper/cartesian state to
  contain one clear event, balanced per scene: `grasp`, `release`, `carry`
  (holding, moves >12cm), `reach` (open, moves >12cm), `fine` (holding, 3-12cm).
  Windows where the gripper closes and reopens within the clip are skipped.
- **No leakage**: no scene is in both splits; no instruction or episode repeats
  across the 70 clips; every episode in `fewshot_examples.yaml` (all in the TRI
  stove kitchen) and the earlier ablation query clips are excluded.

## Files

- `scene_groups.json` — episode_id → scene for all 621 episodes. Built by
  clustering DINOv2 embeddings of each episode's first shoulder frame per lab,
  then manually merging clusters that show the same room (a split room could
  otherwise leak across train/test). Two near-black IPRL episodes are unassigned.
- `dense_eval_v1.yaml` — the manifest: clip id, split, scene, event,
  instruction, and `source` (original record/index/steps). `example_index` /
  `start_step` / `end_step` index into the extracted split files.

## Build and use

```bash
# (re)build the manifest (deterministic for a seed) and the extracted clips
python scripts/build_dense_eval_sets.py select   # writes dense_eval_v1.yaml
python scripts/build_dense_eval_sets.py extract  # writes datasets/dense_eval/{train,test}.tfrecord (~570MB)

# score a prompt (generation + rubric judge); results in outputs/dense_prompt_evals/
python scripts/eval_dense_prompt.py --split train --prompt-file my_prompt.txt --name v2
```

Extracted files keep 1s of context either side of each clip (90 frames), so
`load_trajectory` and the existing generators/judge work on them unchanged.
