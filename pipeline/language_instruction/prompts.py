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
- Output exactly one instruction per line, with no numbering, bullets, or extra \
commentary."""


def build_prompt(step: int, total: int, num_instructions: int, template: str = INSTRUCTION_PROMPT) -> str:
    return template.format(step=step, total=total, num_instructions=num_instructions)

LEADING_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s*")

def parse_instructions(text: str) -> list[str]:
    """Turn a raw model response into a clean list of instruction strings."""
    instructions: list[str] = []
    for line in text.splitlines():
        cleaned = LEADING_MARKER.sub("", line).strip().strip('"').strip()
        if cleaned:
            instructions.append(cleaned)
    return instructions
