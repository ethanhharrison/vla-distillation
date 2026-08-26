"""Prompt templates for Wan rollouts, and the negative-prompt presets.

Mirrors `pipeline/subgoal_image/prompts.py`: a name -> template dict, every
template containing `{instruction}`, selectable by name or passed as a literal.

Why this file exists at all: the first two smokes showed the caption moving the
rollout **3x**, more than any sampling knob measured anywhere in this repo. Wan
is trained on long descriptive captions, and a bare DROID imperative ("Put the
marker in the pot") is far off that distribution — far enough that the model
answers it by deleting the robot and having a human do the task instead, which is
the pretraining bias RoboWM-Bench names. So the wording is the primary
experimental variable here, not a formatting detail.

The templates below vary along three axes that the first smoke suggested matter:

  who acts    - is the robot named as the agent, or left implicit?
  how much    - is the whole task requested, or only the beginning of it?
  what holds  - is the camera/scene explicitly pinned as static?

`NEGATIVES` is kept separate from the templates so the two can be ablated
independently. That matters: it is entirely possible the negative prompt alone
suppresses the human, in which case the long positive captions are wasted tokens.
`bare_negonly` in the sweep tests exactly that.
"""

from __future__ import annotations

#: Raw instruction, exactly as DROID stores it. The baseline, and what any
#: production Stage B would send if we did nothing.
BARE = "{instruction}"

#: The cheapest possible fix: name the agent and pin the camera, nothing else.
#: If this matches `robot_explicit`, the long caption is not earning its length.
ROBOT_PREFIX = (
    "A robotic arm at a workbench: {instruction}. "
    "Only the robot arm moves; no person appears. The camera does not move."
)

#: The long descriptive caption Wan's training distribution favours, generalised
#: from the hand-written one that cut spurious motion 3x on the first smoke.
ROBOT_EXPLICIT = (
    "A fixed surveillance camera watches a white robotic arm at a cluttered "
    "laboratory workbench. The robot arm carries out the following task: "
    "{instruction}. Only the robot arm moves. No person enters the frame and no "
    "human hand appears. The camera is locked off on a tripod and does not move, "
    "pan or zoom."
)

#: ROBOT_EXPLICIT plus an explicit statement that everything not involved in the
#: task is unchanged. Targets the failure the metrics actually show: the rollout
#: moved 3.5x more than reality even once the robot was preserved.
STATIC_SCENE = (
    "A fixed surveillance camera watches a white robotic arm at a cluttered "
    "laboratory workbench. The robot arm carries out the following task: "
    "{instruction}. Everything else in the scene stays exactly where it is: the "
    "table, the objects on it, the background clutter and the lighting are all "
    "unchanged. Only the robot arm moves, slowly and smoothly. No person enters "
    "the frame. The camera is locked off on a tripod and does not move, pan or zoom."
)

#: Stage B does not want the task completed — it wants the scene "a few ticks"
#: into the task, which is a much smaller change and a much easier prediction.
#: Every other template asks for the whole task, which may be why the rollouts
#: overshoot reality so badly.
SUBGOAL_PARTIAL = (
    "A fixed surveillance camera watches a white robotic arm at a cluttered "
    "laboratory workbench. The robot arm has just begun the following task: "
    "{instruction}. The clip shows only the first moment of that motion — the arm "
    "starts to move toward the object and the task is not finished. Everything "
    "else stays exactly where it is. No person enters the frame. The camera is "
    "locked off on a tripod and does not move, pan or zoom."
)

TEMPLATES: dict[str, str] = {
    "bare": BARE,
    "robot_prefix": ROBOT_PREFIX,
    "robot_explicit": ROBOT_EXPLICIT,
    "static_scene": STATIC_SCENE,
    "subgoal_partial": SUBGOAL_PARTIAL,
}

DEFAULT_TEMPLATE = "robot_explicit"


#: Appended to the model card's own (Chinese) negative prompt. Kept in Chinese to
#: match it — the card's negative prompt is Chinese and the text encoder is
#: multilingual, so mixing scripts inside one negative prompt is an uncontrolled
#: variable we do not need.
NEGATIVES: dict[str, str] = {
    # person, human hand, human arm, a real person appears
    "no_human": "人，人手，人的手臂，真人出现",
    # the above plus: camera movement, push-in, zoom
    "no_human_no_camera": "人，人手，人的手臂，真人出现，摄像机移动，镜头推近，镜头变焦",
    "none": "",
}


def resolve_template(name_or_text: str) -> tuple[str, str]:
    """Return (name, text) for a registered template name, or a literal.

    A literal must contain `{instruction}`, so a typo'd name fails loudly here
    rather than being sent to the model as the entire caption.
    """
    if name_or_text in TEMPLATES:
        return name_or_text, TEMPLATES[name_or_text]
    if "{instruction}" in name_or_text:
        return "custom", name_or_text
    raise ValueError(
        f"Unknown template {name_or_text!r}; not a registered name "
        f"({', '.join(TEMPLATES)}) and not a literal containing '{{instruction}}'."
    )


def resolve_negative(name_or_text: str) -> tuple[str, str]:
    """Return (name, text) for a registered negative preset, or a literal."""
    if name_or_text in NEGATIVES:
        return name_or_text, NEGATIVES[name_or_text]
    return "custom", name_or_text


def build_prompt(template: str, instruction: str) -> str:
    return template.format(instruction=instruction.rstrip(". "))
