"""Post-merge paraphrase / noisy-variant augmentation."""

from __future__ import annotations

from pathlib import Path

from pipeline.language_instruction.augment import (
    expand_instructions,
    parse_augment_groups,
    resolve_augment_counts,
)
from pipeline.language_instruction.pipeline import (
    PipelineConfig,
    PipelineResult,
    apply_augment,
    write_pipeline_txt,
)
from pipeline.language_instruction.prompts import build_augment_prompt
from pipeline.language_instruction.uniqueness import Clustering
from pipeline.language_instruction.vlm import VLM

SEEDS = ["Pick up the banana", "Grab the black mug"]


class ScriptedAugmentVLM(VLM):
    """Returns labeled variants per seed, plus one verbatim seed echo to drop."""

    def __init__(self):
        super().__init__("augment-scripted")
        self.calls = 0
        self.last_prompt = ""
        self.last_images: list[bytes] = []

    def generate(self, prompt: str, images: list[bytes]) -> str:
        self.calls += 1
        self.last_prompt = prompt
        self.last_images = images
        return "\n".join(
            [
                "--- seed 1 ---",
                "Pick up the banana",  # exact seed — should be filtered
                "Grab the banana",
                "pik up the bananna",
                "--- seed 2 ---",
                "Take the black mug",
                "grab teh black mug",
            ]
        )


def test_augment_prompt_states_separate_counts():
    prompt = build_augment_prompt(
        SEEDS,
        paraphrases_per_instruction=2,
        noisy_per_instruction=1,
    )
    assert "- 2 clean paraphrase(s)" in prompt
    assert "- 1 noisy rewrite(s)" in prompt
    assert "--- seed N ---" in prompt
    assert "1. Pick up the banana" in prompt
    assert "2. Grab the black mug" in prompt


def test_resolve_augment_counts_defaults_and_rejects_empty():
    assert resolve_augment_counts({}) == (1, 1)
    assert resolve_augment_counts(
        {"paraphrases_per_instruction": 3, "noisy_per_instruction": 0}
    ) == (3, 0)
    try:
        resolve_augment_counts(
            {"paraphrases_per_instruction": 0, "noisy_per_instruction": 0}
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_parse_augment_groups_uses_seed_headers():
    raw = "\n".join(
        [
            "--- seed 1 ---",
            "Grab the banana",
            "pik up the bananna",
            "--- seed 2 ---",
            "Take the black mug",
        ]
    )
    groups = parse_augment_groups(raw, SEEDS, expected_per_seed=2)
    assert groups == {
        "Pick up the banana": ["Grab the banana", "pik up the bananna"],
        "Grab the black mug": ["Take the black mug"],
    }


def test_parse_augment_groups_falls_back_to_chunks_without_headers():
    raw = "\n".join(
        [
            "Grab the banana",
            "pik up the bananna",
            "Take the black mug",
            "grab teh black mug",
        ]
    )
    groups = parse_augment_groups(raw, SEEDS, expected_per_seed=2)
    assert groups["Pick up the banana"] == ["Grab the banana", "pik up the bananna"]
    assert groups["Grab the black mug"] == ["Take the black mug", "grab teh black mug"]


def test_expand_instructions_groups_by_seed_and_is_text_only():
    vlm = ScriptedAugmentVLM()
    groups = expand_instructions(
        vlm,
        SEEDS,
        paraphrases_per_instruction=1,
        noisy_per_instruction=1,
    )

    assert vlm.calls == 1
    assert vlm.last_images == []
    assert "Pick up the banana" not in groups["Pick up the banana"]
    assert groups == {
        "Pick up the banana": ["Grab the banana", "pik up the bananna"],
        "Grab the black mug": ["Take the black mug", "grab teh black mug"],
    }


def test_expand_instructions_skips_empty_input():
    vlm = ScriptedAugmentVLM()
    assert expand_instructions(vlm, []) == {}
    assert vlm.calls == 0


def make_result(*, with_clustering: bool) -> PipelineResult:
    clustering = Clustering(clusters=[[0, 2], [1]]) if with_clustering else None
    config = PipelineConfig(
        config_path=Path("cfg.yaml"),
        record_path=Path("rec.tfrecord"),
        calls=[],
        augment={
            "enabled": True,
            "provider": "gemini",
            "paraphrases_per_instruction": 1,
            "noisy_per_instruction": 1,
        },
        uniqueness={
            "enabled": with_clustering,
            "provider": "gemini",
            "when": "after",
        },
    )
    instructions = [
        "Pick up the banana",
        "Grab the black mug",
        "Lift the banana",
    ]
    return PipelineResult(
        config=config,
        calls=[],
        merged_by_step={0: list(instructions)},
        provenance={
            0: {
                "Pick up the banana": ["default"],
                "Lift the banana": ["precision"],
            }
        },
        clustering_by_step={0: clustering} if clustering is not None else {},
    )


def test_apply_augment_uses_representatives_only():
    """With after-merge clustering, folded duplicates are not seeded."""
    result = make_result(with_clustering=True)
    assert result.representatives(0) == SEEDS

    vlm = ScriptedAugmentVLM()
    apply_augment(
        result,
        vlm=vlm,
        paraphrases_per_instruction=1,
        noisy_per_instruction=1,
    )

    assert vlm.calls == 1
    assert "Lift the banana" not in vlm.last_prompt
    assert result.augmented_by_step[0] == {
        "Pick up the banana": ["Grab the banana", "pik up the bananna"],
        "Grab the black mug": ["Take the black mug", "grab teh black mug"],
    }
    assert result.final_instructions(0) == [
        "Pick up the banana",
        "Grab the black mug",
        "Grab the banana",
        "pik up the bananna",
        "Take the black mug",
        "grab teh black mug",
    ]
    assert result.provenance[0]["Grab the banana"] == ["augment"]


def test_merged_txt_groups_variants_under_their_seed(tmp_path):
    result = make_result(with_clustering=True)
    apply_augment(
        result,
        vlm=ScriptedAugmentVLM(),
        paraphrases_per_instruction=1,
        noisy_per_instruction=1,
    )

    path = write_pipeline_txt(result, tmp_path / "merged.txt")
    lines = [line.rstrip() for line in path.read_text().splitlines()]

    banana = lines.index("  - Pick up the banana | from: default")
    mug = lines.index("  - Grab the black mug")
    assert lines[banana + 1] == "    - Grab the banana | from: augment"
    assert lines[banana + 2] == "    - pik up the bananna | from: augment"
    assert lines[mug + 1] == "    - Take the black mug | from: augment"
    assert lines[mug + 2] == "    - grab teh black mug | from: augment"
    assert banana < mug
    assert (
        "  (duplicate) Lift the banana | from: precision | duplicate of: Pick up the banana"
        in lines
    )


def test_visualizer_parses_nested_augment_groups(tmp_path):
    """Producer nested indent and consumer parse_run must agree."""
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))
    from summarize_language_instructions import parse_run, render_html

    result = make_result(with_clustering=True)
    apply_augment(
        result,
        vlm=ScriptedAugmentVLM(),
        paraphrases_per_instruction=1,
        noisy_per_instruction=1,
    )
    path = write_pipeline_txt(result, tmp_path / "merged.txt")
    run = parse_run(path)

    assert run["info"]["augment"].startswith("(")
    step = run["steps"][0]
    assert len(step["items"]) == 2
    assert step["items"][0]["text"] == "Pick up the banana"
    assert [v["text"] for v in step["items"][0]["augment"]] == [
        "Grab the banana",
        "pik up the bananna",
    ]
    assert step["items"][1]["text"] == "Grab the black mug"
    assert step["items"][0]["augment"][0]["grades"] == {"from": "augment"}
    assert step["rejected"][0]["text"] == "Lift the banana"

    html = render_html(run, path)
    assert "Grab the banana" in html
    assert 'class="augment"' in html
    assert "from augment" in html
