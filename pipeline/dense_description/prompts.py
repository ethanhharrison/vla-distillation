"""Prompt templates for dense change-of-state descriptions over trajectory clips."""

from __future__ import annotations

DENSE_DESCRIPTION_PROMPT = """You are labeling a robot manipulation dataset.

You are shown camera views of a robot arm at TWO moments in time that bound a \
short clip of the trajectory:
- The FIRST group of images is the START of the clip (one image per camera).
- The SECOND group of images is the END of the clip (same cameras, same order).

Cameras in each group, in order: {camera_list}.
Clip span: steps {start_step}–{end_step} of {total} (about {clip_seconds:g}s \
at {fps:g} fps).

The language instruction associated with this trajectory is:
"{language_instruction}"

Write a dense natural-language description of how the scene changes from the \
start frame to the end frame. Focus on observable motion and state change that \
relates to carrying out that instruction: which objects move or are contacted, \
how the robot arm and gripper move, and what the environment looks like at the \
end relative to the start.

Guidelines:
- Be concrete and visual (objects, contacts, directions, relative positions).
- Do NOT invent objects that are not visible in the images.
- Do NOT merely restate the language instruction; describe the change you see.
- Prefer one coherent paragraph (or a few short sentences). No bullet lists, \
numbering, or preamble like "Here is the description:".
"""


def build_dense_prompt(
    *,
    language_instruction: str,
    cameras: tuple[str, ...] | list[str],
    start_step: int,
    end_step: int,
    total: int,
    clip_seconds: float,
    fps: float,
    template: str = DENSE_DESCRIPTION_PROMPT,
) -> str:
    """Render the dense-description prompt for one clip."""
    return template.format(
        language_instruction=language_instruction,
        camera_list=", ".join(cameras),
        start_step=start_step,
        end_step=end_step,
        total=total,
        clip_seconds=clip_seconds,
        fps=fps,
    )


def parse_description(text: str) -> str:
    """Normalize a VLM response into a single dense description string."""
    cleaned = text.strip()
    # Drop common wrappers the model sometimes adds.
    for prefix in (
        "Here is the description:",
        "Here's the description:",
        "Dense description:",
        "Description:",
    ):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned.strip().strip('"').strip()
