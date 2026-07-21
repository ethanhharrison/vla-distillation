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
