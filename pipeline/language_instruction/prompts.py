"""Prompt templates and response parsing for instruction generation."""

from __future__ import annotations

import re

# The image(s) shown to the model are the robot's camera views at a single step
# of a trajectory. We ask the model for tasks the robot could plausibly start
# executing from this exact configuration.
INSTRUCTION_PROMPT = """You are labeling a robot manipulation dataset.

The image(s) are the camera view(s) of a robot arm at one moment in time (step \
{step} of {total}). Looking at the current scene and the objects within reach, \
propose {num_instructions} distinct natural-language instructions describing \
tasks the robot could plausibly begin executing *starting from this exact \
configuration*.

Guidelines:
- Each instruction should be a short imperative command (e.g. "pick up the red \
block").
- Make the instructions meaningfully different from one another.
- Only reference objects that are actually visible in the scene.
- Do NOT copy, repeat, or paraphrase any of the instructions listed under \
"Instructions to avoid" below (this includes the dataset's original task \
instructions and anything already suggested at earlier steps). Propose \
genuinely new tasks instead.
- Output exactly one instruction per line, with no numbering, bullets, or extra \
commentary.
{avoid_section}"""


def _build_avoid_section(
    original_instructions: list[str] | None,
    previous_instructions: list[str] | None,
) -> str:
    """Render the 'Instructions to avoid' block, or '' when there's nothing."""
    original = original_instructions or []
    previous = previous_instructions or []
    if not original and not previous:
        return ""

    lines = ["", "Instructions to avoid (do not copy, repeat, or rephrase these):"]
    if original:
        lines.append("Original dataset task instructions:")
        lines.extend(f"- {text}" for text in original)
    if previous:
        lines.append("Instructions you already suggested at earlier steps:")
        lines.extend(f"- {text}" for text in previous)
    return "\n".join(lines)


def build_prompt(
    step: int,
    total: int,
    num_instructions: int,
    original_instructions: list[str] | None = None,
    previous_instructions: list[str] | None = None,
    template: str = INSTRUCTION_PROMPT,
) -> str:
    return template.format(
        step=step,
        total=total,
        num_instructions=num_instructions,
        avoid_section=_build_avoid_section(original_instructions, previous_instructions),
    )

LEADING_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s*")

def parse_instructions(text: str) -> list[str]:
    """Turn a raw model response into a clean list of instruction strings."""
    instructions: list[str] = []
    for line in text.splitlines():
        cleaned = LEADING_MARKER.sub("", line).strip().strip('"').strip()
        if cleaned:
            instructions.append(cleaned)
    return instructions


# The judge is shown the same scene image(s) plus the exact system prompt that
# produced the candidates, so it can score how well each candidate obeys that
# prompt (feasible in the scene, references only visible objects, genuinely new).
JUDGE_PROMPT = """You are a strict reviewer for a robot manipulation dataset.

The image(s) are the robot's camera view(s) at one moment in time (step {step} \
of {total}) — the same view(s) used to generate the candidate instructions below.

Another model was asked to propose instructions using this exact system prompt:
--- BEGIN GENERATION SYSTEM PROMPT ---
{generation_prompt}
--- END GENERATION SYSTEM PROMPT ---

Score how well each candidate instruction below adheres to that system prompt. \
Use an integer from 1 to 5, where:
- 5 = fully adheres: the task is physically plausible starting from the current \
scene, references only objects actually visible, is a clear short imperative \
command, and is genuinely new (not a copy or paraphrase of any instruction the \
system prompt says to avoid).
- 3 = partially adheres (e.g. slightly ambiguous, or references an object that \
is only marginally visible).
- 1 = clearly violates the system prompt (e.g. references objects not present, \
is not achievable from this configuration, or duplicates an instruction to avoid).

Candidate instructions:
{numbered_instructions}

Output exactly one line per candidate, in the same order, formatted as:
<candidate number>. <score>
Output nothing else — no explanations, headers, or extra text."""


def build_judge_prompt(
    generation_prompt: str,
    instructions: list[str],
    step: int,
    total: int,
    template: str = JUDGE_PROMPT,
) -> str:
    """Render the judge prompt for a set of candidate instructions."""
    numbered = "\n".join(
        f"{i}. {text}" for i, text in enumerate(instructions, start=1)
    )
    return template.format(
        step=step,
        total=total,
        generation_prompt=generation_prompt,
        numbered_instructions=numbered,
    )


SCORE_LINE = re.compile(r"(\d+)\D+(\d+)")
SINGLE_INT = re.compile(r"-?\d+")


def parse_scores(text: str, num_expected: int) -> dict[int, int]:
    """Parse a judge response into a {candidate_index: score} mapping.

    Candidate indices are 0-based (matching the order of the instructions list).
    Scores are clamped to the 1-5 range. Unparseable candidates are simply
    omitted from the returned mapping so callers can decide how to treat them.
    """
    scores: dict[int, int] = {}
    for line in text.splitlines():
        match = SCORE_LINE.search(line)
        if not match:
            continue
        candidate = int(match.group(1)) - 1
        score = int(match.group(2))
        if 0 <= candidate < num_expected:
            scores[candidate] = max(1, min(5, score))

    if scores:
        return scores

    # Fallback: the model may have emitted one bare score per line, with no
    # candidate numbering. Assign them to candidates in order.
    for idx, line in enumerate(text.splitlines()):
        if idx >= num_expected:
            break
        found = SINGLE_INT.search(line)
        if found:
            scores[idx] = max(1, min(5, int(found.group())))
    return scores
