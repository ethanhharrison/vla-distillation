"""Prompt templates for dense change-of-state descriptions over trajectory clips."""

from __future__ import annotations

DENSE_DESCRIPTION_PROMPT = """You are labeling a robot manipulation dataset.

You are shown camera views of a robot arm at TWO moments in time that bound a \
short clip of the trajectory:
- The FIRST group of images is the START of the clip (one image per camera).
- The SECOND group of images is the END of the clip (same cameras, same order).

Cameras in each group, in order: {camera_list}.
So image layout is: start[{camera_list}], then end[{camera_list}].
Clip span: steps {start_step}–{end_step} of {total} (about {clip_seconds:g}s \
at {fps:g} fps).

The language instruction associated with this trajectory is:
"{language_instruction}"

For EACH camera separately, write a dense description of how the robot and \
environment change from that camera's START frame to its END frame. Relate the \
observed change to carrying out the language instruction when it is visible \
from that view.

Guidelines:
- Cover every camera listed above, in the same order. Start each camera's \
section with its exact camera name as a short header line (e.g. \
"shoulder_image_1:"), then the description for that view.
- Be very specific and visual: name the robot links/gripper, contacted \
objects, directions of motion, and relative positions.
- Anything that moved or changed state between the START and END frames \
must get at least one full sentence describing that change in detail \
(what moved, how it moved, and where it ends relative to the start). Do \
not summarize several motions in a vague phrase — give each notable change \
its own clear sentence.
- Do NOT invent objects that are not visible in that camera's images.
- Do NOT merely restate the language instruction; describe the change you see.
- No preamble like "Here is the description:". No bullet lists within a \
camera section.
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
