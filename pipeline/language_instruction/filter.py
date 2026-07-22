"""VLM-judge filtering for candidate language instructions.

After the generator proposes instructions for a step, an optional (possibly
different) VLM "judge" scores each candidate 1-5 for how well it adheres to the
generation prompt (feasible in the scene, references only visible objects,
genuinely new). Candidates scoring below a threshold are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompts import JUDGE_PROMPT, build_judge_prompt, parse_scores
from .vlm import VLM, build_vlm


@dataclass
class ScoredInstruction:
    """A single candidate instruction with its judge score and verdict."""
    instruction: str
    score: int | None
    accepted: bool


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
    template: str = JUDGE_PROMPT,
) -> tuple[list[str], list[ScoredInstruction], str]:
    """Score candidates with the judge and split them by the threshold.

    Returns `(accepted, scored, raw_response)`. A candidate whose score cannot
    be parsed is kept (score `None`, accepted) so a flaky judge response never
    silently discards otherwise-valid instructions.
    """
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
        scored.append(ScoredInstruction(instruction, score, is_accepted))
        if is_accepted:
            accepted.append(instruction)
    return accepted, scored, raw_response
