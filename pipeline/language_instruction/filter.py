"""Two-stage VLM-judge filtering for candidate language instructions.

Stage 1 (adherence): score system-prompt fit *except* uniqueness.
Stage 2 (uniqueness): only candidates that pass stage 1 are scored for
novelty vs the avoid-list and other survivors.
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompts import (
    ADHERENCE_JUDGE_PROMPT,
    UNIQUENESS_JUDGE_PROMPT,
    build_judge_prompt,
    parse_scores,
)
from .vlm import VLM, build_vlm


@dataclass
class ScoredInstruction:
    """A candidate with stage scores and a final accept/reject verdict."""

    instruction: str
    accepted: bool
    adherence_score: int | None = None
    uniqueness_score: int | None = None
    rejected_stage: str | None = None  # "adherence" | "uniqueness" | None


def build_judge(provider: str, model: str | None = None) -> VLM:
    """Build the judge VLM for the given provider/model."""
    return build_vlm(provider, model=model)


def _score_batch(
    judge: VLM,
    generation_prompt: str,
    images: list[bytes],
    instructions: list[str],
    step: int,
    total: int,
    threshold: int,
    template: str,
) -> tuple[list[str], list[tuple[str, int | None, bool]], str]:
    """Run one judge call; return (accepted_texts, per-item triples, raw)."""
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
    items: list[tuple[str, int | None, bool]] = []
    for index, instruction in enumerate(instructions):
        score = scores.get(index)
        is_accepted = score is None or score >= threshold
        items.append((instruction, score, is_accepted))
        if is_accepted:
            accepted.append(instruction)
    return accepted, items, raw_response


def score_instructions(
    judge: VLM,
    generation_prompt: str,
    images: list[bytes],
    instructions: list[str],
    step: int,
    total: int,
    threshold: int,
    template: str = ADHERENCE_JUDGE_PROMPT,
    uniqueness_template: str = UNIQUENESS_JUDGE_PROMPT,
    uniqueness_threshold: int | None = None,
    uniqueness: bool = True,
) -> tuple[list[str], list[ScoredInstruction], str]:
    """Two-stage filter: adherence (all candidates) then uniqueness (survivors).

    `template` is the stage-1 adherence prompt. `uniqueness_template` is stage 2.
    `uniqueness_threshold` defaults to `threshold` when unset.

    Set `uniqueness=False` to run adherence only. Use that when uniqueness is
    handled elsewhere -- `uniqueness.cluster_instructions` runs once over the
    merged set in pipeline.py -- so candidates are never filtered twice under
    two different definitions of "duplicate".
    """
    uniq_threshold = threshold if uniqueness_threshold is None else uniqueness_threshold

    # --- Stage 1: adherence (ignore uniqueness) ---
    adherence_accepted, adherence_items, adherence_raw = _score_batch(
        judge=judge,
        generation_prompt=generation_prompt,
        images=images,
        instructions=instructions,
        step=step,
        total=total,
        threshold=threshold,
        template=template,
    )

    scored_by_index: list[ScoredInstruction] = []
    for instruction, adh_score, adh_ok in adherence_items:
        if not adh_ok:
            scored_by_index.append(
                ScoredInstruction(
                    instruction=instruction,
                    accepted=False,
                    adherence_score=adh_score,
                    uniqueness_score=None,
                    rejected_stage="adherence",
                )
            )
        else:
            scored_by_index.append(
                ScoredInstruction(
                    instruction=instruction,
                    accepted=True,  # may flip after uniqueness
                    adherence_score=adh_score,
                    uniqueness_score=None,
                    rejected_stage=None,
                )
            )

    if not uniqueness or not adherence_accepted:
        raw = f"--- adherence ---\n{adherence_raw}"
        return adherence_accepted, scored_by_index, raw

    # --- Stage 2: uniqueness on survivors only ---
    _uniqueness_accepted, uniqueness_items, uniqueness_raw = _score_batch(
        judge=judge,
        generation_prompt=generation_prompt,
        images=images,
        instructions=adherence_accepted,
        step=step,
        total=total,
        threshold=uniq_threshold,
        template=uniqueness_template,
    )
    survivor_iter = iter(uniqueness_items)
    for scored in scored_by_index:
        if scored.rejected_stage == "adherence":
            continue
        try:
            _inst, uniq_score, uniq_ok = next(survivor_iter)
        except StopIteration:
            break
        scored.uniqueness_score = uniq_score
        if not uniq_ok:
            scored.accepted = False
            scored.rejected_stage = "uniqueness"
        else:
            scored.accepted = True
            scored.rejected_stage = None

    accepted = [t for t, _, ok in uniqueness_items if ok]
    raw = (
        f"--- adherence ---\n{adherence_raw}\n"
        f"--- uniqueness ---\n{uniqueness_raw}"
    )
    return accepted, scored_by_index, raw
