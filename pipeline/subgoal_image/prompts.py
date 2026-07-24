"""Subgoal-edit prompt templates (Stage B).

Semantics we want from the image editor: the edit depicts a **subgoal** — the
scene a few moments into executing the instruction. It is a *scene-level* change
(the manipulated objects have moved a little toward the goal) while the robot's
pose stays roughly the same. It must NOT depict task completion or a different
scene.

The template is a first-class config field: its id (`prompt_template_id`) is a
hash folded into each sample's identity, so two templates on the same source
frame are distinct, separately-cached, side-by-side-comparable variants.
"""

from __future__ import annotations

from .imaging import sha256_hex

# name -> template. `{instruction}` is substituted per sample.
TEMPLATES: dict[str, str] = {
    "default": (
        "This is a photo from a robot's camera during a manipulation task. "
        "The robot has just been given the instruction: \"{instruction}\". "
        "Edit the image to show the scene a FEW MOMENTS LATER, just after the "
        "robot begins carrying out this instruction. Show a small, plausible "
        "amount of progress: the relevant object(s) should have moved slightly "
        "toward the goal of the instruction. Keep it the SAME scene from the "
        "SAME camera viewpoint, and keep the robot arm in roughly the same "
        "position. Do NOT show the task as finished. Change only what a couple "
        "of seconds of motion would change."
    ),
    "minimal": (
        "Robot camera view. Instruction: \"{instruction}\". Show the scene a "
        "few moments after the robot starts this instruction — a small change "
        "toward the goal, same viewpoint, robot arm roughly unchanged, task not "
        "yet complete."
    ),
    "object_centric": (
        "A robot is about to perform: \"{instruction}\". Edit this camera image "
        "so that the object(s) referenced by the instruction have just started "
        "moving toward their goal (a subgoal partway through the task). Preserve "
        "the background, lighting, camera angle, and the robot arm's pose. Make "
        "the smallest edit that clearly shows the task has begun but is not done."
    ),
}

DEFAULT_TEMPLATE = "default"


def template_id(template_text: str) -> str:
    """Short stable id for a template's text (first 12 hex of sha256)."""
    return sha256_hex(template_text)[:12]


def resolve_template(name_or_text: str) -> tuple[str, str]:
    """Return (template_name, template_text).

    Accepts a registered template name, or a literal template string (must
    contain `{instruction}`).
    """
    if name_or_text in TEMPLATES:
        return name_or_text, TEMPLATES[name_or_text]
    if "{instruction}" in name_or_text:
        return "custom", name_or_text
    raise ValueError(
        f"Unknown template {name_or_text!r}; not a registered name "
        f"({', '.join(TEMPLATES)}) and not a literal template containing "
        "'{instruction}'."
    )


def build_prompt(template_text: str, instruction: str) -> str:
    return template_text.format(instruction=instruction)
