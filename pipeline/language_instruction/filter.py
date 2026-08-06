"""VLM adherence filtering for candidate language instructions.

Scores each candidate against the generation system prompt (feasibility,
visibility, style). Uniqueness is handled separately by
`pipeline.language_instruction.uniqueness` after the multi-call merge.
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompts import ADHERENCE_JUDGE_PROMPT, build_judge_prompt, parse_scores
from .vlm import VLM, build_vlm


@dataclass
class ScoredInstruction:
    """A candidate with an adherence score and accept/reject verdict."""

    instruction: str
    accepted: bool
    adherence_score: int | None = None


def build_judge(provider: str, model: str | None = None) -> VLM:
    """Build the judge VLM for the given provider/model."""
    return build_vlm(provider, model=model)


def score_instructions(
    judge: VLM,
    generation_prompt: str,
    images: list[bytes],
    instructions: list[str],
    step: int,
    total: int,
    threshold: int,
    template: str = ADHERENCE_JUDGE_PROMPT,
) -> tuple[list[str], list[ScoredInstruction], str]:
    """Score candidates on adherence and keep those at or above `threshold`."""
    if not instructions:
        return [], [], ""

    judge_prompt = build_judge_prompt(
        generation_prompt=generation_prompt,
        instructions=instructions,
        step=step,
        total=total,
        template=template,
    )
    raw_response = judge.generate(judge_prompt, images)
    scores = parse_scores(raw_response, len(instructions))

    accepted: list[str] = []
    scored: list[ScoredInstruction] = []
    for index, instruction in enumerate(instructions):
        score = scores.get(index)
        is_accepted = score is None or score >= threshold
        scored.append(
            ScoredInstruction(
                instruction=instruction,
                accepted=is_accepted,
                adherence_score=score,
            )
        )
        if is_accepted:
            accepted.append(instruction)
    return accepted, scored, raw_response
