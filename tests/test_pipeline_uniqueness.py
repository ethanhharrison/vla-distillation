"""Wiring: adherence-only filter, and the merged output format round-trip."""

from __future__ import annotations

import sys
from pathlib import Path

from pipeline.language_instruction.filter import score_instructions
from pipeline.language_instruction.pipeline import (
    PipelineConfig,
    PipelineResult,
    write_pipeline_txt,
)
from pipeline.language_instruction.uniqueness import Clustering
from pipeline.language_instruction.vlm import VLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_language_instructions import _split_score  # noqa: E402

INSTRUCTIONS = ["Pick up the banana", "Grab the black mug", "Lift the banana"]


class CountingVLM(VLM):
    """Returns a full-marks score line per candidate and counts the calls."""

    def __init__(self):
        super().__init__("scripted")
        self.calls = 0

    def generate(self, prompt: str, images: list[bytes]) -> str:
        self.calls += 1
        count = sum(
            1 for line in prompt.splitlines() if line.strip()[:2].rstrip(".").isdigit()
        )
        self.usage.add(input_tokens=100, output_tokens=10)
        return "\n".join(f"{i}. 5" for i in range(1, max(count, 1) + 1))


def test_score_instructions_is_adherence_only():
    """Per-call judge must not run a second uniqueness VLM pass."""
    judge = CountingVLM()

    accepted, scored, raw = score_instructions(
        judge=judge,
        generation_prompt="p",
        images=[],
        instructions=INSTRUCTIONS,
        step=0,
        total=10,
        threshold=3,
    )

    assert judge.calls == 1
    assert accepted == INSTRUCTIONS
    assert all(s.adherence_score == 5 for s in scored)
    assert "--- uniqueness ---" not in raw


# --- merged output --------------------------------------------------------- #


def make_result(clustering: Clustering | None) -> PipelineResult:
    config = PipelineConfig(
        config_path=Path("cfg.yaml"),
        record_path=Path("rec.tfrecord"),
        calls=[],
        uniqueness={"enabled": clustering is not None, "provider": "gemini"},
    )
    return PipelineResult(
        config=config,
        calls=[],
        merged_by_step={0: list(INSTRUCTIONS)},
        provenance={
            0: {
                "Pick up the banana": ["default"],
                "Lift the banana": ["precision"],
            }
        },
        clustering_by_step={0: clustering} if clustering is not None else {},
    )


def test_representatives_and_duplicates_resolve_to_instruction_text():
    result = make_result(Clustering(clusters=[[0, 2], [1]]))

    assert result.representatives(0) == ["Pick up the banana", "Grab the black mug"]
    assert result.duplicates(0) == {"Lift the banana": "Pick up the banana"}


def test_without_clustering_everything_is_kept():
    result = make_result(None)

    assert result.representatives(0) == INSTRUCTIONS
    assert result.duplicates(0) == {}


def test_merged_txt_records_what_each_duplicate_was_folded_into(tmp_path):
    result = make_result(Clustering(clusters=[[0, 2], [1]]))

    path = write_pipeline_txt(result, tmp_path / "merged.txt")
    lines = [line.strip() for line in path.read_text().splitlines()]

    assert "- Pick up the banana | from: default" in lines
    assert "- Grab the black mug" in lines
    assert (
        "(duplicate) Lift the banana | from: precision | duplicate of: Pick up the banana"
        in lines
    )


def test_the_visualizer_can_parse_the_line_the_pipeline_writes(tmp_path):
    """Producer and consumer must agree, or the debug view silently loses tags."""
    result = make_result(Clustering(clusters=[[0, 2], [1]]))
    path = write_pipeline_txt(result, tmp_path / "merged.txt")

    line = next(
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip().startswith("(duplicate) ")
    )
    text, grades = _split_score(line[len("(duplicate) ") :])

    assert text == "Lift the banana"
    assert grades == {"from": "precision", "duplicate of": "Pick up the banana"}
