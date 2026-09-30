"""Prompt templates for dense change-of-state descriptions over trajectory clips."""

from __future__ import annotations

import re

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


DENSE_VIDEO_PROMPT = """You are labeling a robot manipulation dataset.

You are shown a single video of a robot arm over a short clip of the \
trajectory. The frame is split into {n_cameras} equal horizontal bands, \
stacked top to bottom, playing forward together at {fps:g} fps for the whole \
clip:
{camera_bands}

Clip span: steps {start_step}-{end_step} of {total} (about {clip_seconds:g}s).

The language instruction associated with this trajectory is:
"{language_instruction}"

Write a dense natural-language description of how the robot and environment \
change over the course of this clip. Track the motion continuously rather \
than only comparing the first and last moment — use what happens in between \
to describe how things move, not just where they end up. Relate the change to \
carrying out the language instruction when it is visible.

Guidelines:
- Be very specific and visual: name the robot links/gripper, contacted \
objects, directions of motion, and relative positions.
- Anything that moves or changes state must get at least one full sentence \
describing that change in detail (what moved, how it moved, and where it \
ends relative to the start). Do not summarize several motions in a vague \
phrase — give each notable change its own clear sentence.
- If the views disagree or one occludes the relevant motion, say so rather \
than guessing.
- Do NOT invent objects that are not visible in the video.
- Do NOT merely restate the language instruction; describe the change you see.
- No preamble like "Here is the description:". No bullet lists.
"""


_BAND_NAMES = ("TOP", "MIDDLE", "BOTTOM")


def _band_label(index: int, n_cameras: int) -> str:
    if n_cameras <= len(_BAND_NAMES):
        return _BAND_NAMES[index] if n_cameras > 1 else "THE"
    return f"band {index + 1} of {n_cameras}"


def build_dense_video_prompt(
    *,
    language_instruction: str,
    cameras: tuple[str, ...],
    start_step: int,
    end_step: int,
    total: int,
    clip_seconds: float,
    fps: float,
    template: str = DENSE_VIDEO_PROMPT,
) -> str:
    """Render the dense-description video prompt for one clip."""
    camera_bands = "\n".join(
        f"- {_band_label(i, len(cameras))}: {cam}" for i, cam in enumerate(cameras)
    )
    return template.format(
        language_instruction=language_instruction,
        n_cameras=len(cameras),
        camera_bands=camera_bands,
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


# --- Few-shot format: a "Shared" summary + a per-camera "Per-view" breakdown,
# taught to the model via hand-written (images, answer) example turns rather
# than a single instruction block. See generate_fewshot.py.

FEWSHOT_DENSE_DESCRIPTION_INSTRUCTIONS = """You are labeling a robot manipulation dataset.

For each clip you will be shown the START and END camera frames (start \
frames first, then end frames, same camera order each time).

Write your answer in exactly this format:

Shared:
Gripper: <what the gripper does across the whole clip - what it holds, when \
it opens/closes/slips, its overall trajectory>
Scene change: <what changes in the scene as a whole, independent of any one \
camera - what moves, what stays put>
Terminal gripper state: <the gripper's state at the END frame - open/closed, \
holding what, roughly where>

Per-view:
{camera_labels}

Guidelines:
- Be concrete and visual: name objects, directions, contacts.
- The per-view lines describe the SAME event but strictly from each camera's \
own framing - the same motion can look like "moves right" in one view and \
"moves left" in another. Describe what that view actually shows, don't just \
restate the shared summary.
- Do NOT invent objects that are not visible in the images.
- No preamble, no bullet lists, no headers other than "Shared:" and "Per-view:"."""


FEWSHOT_TERSE_INSTRUCTIONS = """You are labeling a robot manipulation clip from its START and END camera frames \
(start frames first, then end frames, same camera order each time).

Answer in exactly this format, nothing else:

Shared:
Gripper: ...
Scene change: ...
Terminal gripper state: ...

Per-view:
{camera_labels}

Be concrete. Do not invent objects you can't see."""


FEWSHOT_MOTION_EMPHASIS_INSTRUCTIONS = """You are labeling a robot manipulation dataset from START and END camera \
frames per clip (start frames first, then end frames, same camera order \
each time).

Your job is to describe MOTION and CHANGE, not to list the static scene. \
Every sentence should describe something that moved, opened, closed, \
appeared, or disappeared between START and END - never a static fact (e.g. \
"the counter is white") unless it's needed to say where something ended up \
relative to it.

Write your answer in exactly this format:

Shared:
Gripper: <the gripper's trajectory and state changes across the clip>
Scene change: <what changed in the scene as a whole>
Terminal gripper state: <state at END - open/closed, holding what, where>

Per-view:
{camera_labels}

Guidelines:
- Every per-view line must name at least one concrete motion visible in \
THAT camera specifically - not a restatement of the shared summary.
- Do NOT invent objects that are not visible in the images.
- No preamble, no bullet lists, no headers other than "Shared:" and "Per-view:"."""


FEWSHOT_COMPARE_EXPLICIT_INSTRUCTIONS = """You are labeling a robot manipulation dataset. For each clip you will be \
shown the START and END camera frames (start frames first, then end frames, \
same camera order each time).

Before answering, mentally compare the START and END frame of each camera \
one at a time: what is in a different place, a different pose, or a \
different state? Only once you've done that for every camera, write your \
answer - grounded in those specific differences - in exactly this format:

Shared:
Gripper: <what the gripper does across the whole clip - what it holds, when \
it opens/closes/slips, its overall trajectory>
Scene change: <what changes in the scene as a whole, independent of any one \
camera - what moves, what stays put>
Terminal gripper state: <the gripper's state at the END frame - open/closed, \
holding what, roughly where>

Per-view:
{camera_labels}

Guidelines:
- Be concrete and visual: name objects, directions, contacts.
- The per-view lines describe the SAME event but strictly from each camera's \
own framing - the same motion can look like "moves right" in one view and \
"moves left" in another.
- Do NOT invent objects that are not visible in the images.
- No preamble, no bullet lists, no headers other than "Shared:" and "Per-view:". \
Do not show your comparison step - only the final answer."""


# Tuned on the dense_eval_v1 train split for gpt-6-luna (reasoning_effort=low)
# against the correctness rubric: an explicit checklist the model answers in a
# scratch "Observations:" section (stripped by parse_fewshot_response) before
# the description. See outputs/dense_prompt_evals/luna_low_tuning/README.md.
FEWSHOT_CHECKLIST_INSTRUCTIONS = """You are labeling a robot manipulation dataset.

For each clip you will be shown the START and END camera frames (start frames first, then end frames, same camera order each time). The last camera (wrist_image) is mounted on the gripper, so its view moves with the gripper.

Write your answer in exactly this format. First write a short "Observations:" section answering the seven checks below (one line each, numbered 1-7), then the description:

Observations:
1. <...>
2. <...>
3. <...>
4. <...>
5. <...>
6. <...>
7. <...>

Shared:
Gripper: <what the gripper does between START and END - what it holds, whether it grasps or releases, its overall movement>
Scene change: <the net change in the scene between START and END - what moved, opened, closed, turned or was released, and where it ended up>
Terminal gripper state: <the gripper at the END frame - open or closed, exactly what it is holding (or "empty"), and where it is>

Per-view:
{camera_labels}

How to decide what happened - check these before writing:
1. Wrist START frame: is an object already between the fingers? If yes, the gripper was already holding it - do not say it grasps or picks it up.
2. Wrist END frame: are the fingers closed around an object, or open/empty? This decides the Terminal gripper state. Commit to one answer; do not write "open or closed" or "unclear".
3. Is the object actually lifted at the END (clear gap below it), or still resting on a surface? A gripper near or touching an object has not necessarily grasped or moved it.
4. Compare where each touched object is at START vs END in the shoulder views. Describe only the net change you can see. Never assume the task succeeded or will succeed - describe only what the END frames show.
5. Look for changes to fixed parts: compare how far each door, drawer or lid is open at START vs END, and whether a knob or switch turned. A door that opened further counts as a change. State it.
6. Which object is the gripper interacting with - the object between, touching, or directly under the fingers at END (or at START, if it let go)? Always name it, even if it was not grasped or moved.
7. If the gripper holds an object at END, is the object tilted, or raised even slightly off its surface? Say so.

Naming rules (a wrong name counts as an error):
- Mention only objects that matter: the object the gripper handles and the surface or container it moves to or from. Leave out background objects.
- Use the most specific name you are sure of. If unsure, use a plain generic name that is certainly true ("the white object", "the container", "the appliance", "the cloth") instead of guessing.
- Name the surface an object rests on generically ("the surface", "the seat", "the shelf") unless you are sure what it is (a sofa is not a table; a stovetop is not a counter).
- Only state a color, material, or printed text if it is clearly visible. Do not name a grasp point (handle, rim, knob) unless it is clearly visible.

Per-view rules:
- Each per-view line is ONE short sentence. It says where the gripper and the handled object are in THAT camera at the END, relative to the object it is interacting with or the nearest surface - not relative to distant furniture.
- In the wrist view, the whole scene appears to shift because the camera moves with the gripper - do not describe that as objects moving.
- Do not make claims about what is out of view or how much of the robot is visible.

The Observations section is scratch work and will be removed; the description after "Shared:" must stand on its own and agree with your observations. The example answers you are shown omit Observations - always include it anyway.

Style: short, concrete sentences. No bullet lists, no headers other than "Observations:", "Shared:" and "Per-view:"."""


FEWSHOT_INSTRUCTION_TEMPLATES: dict[str, str] = {
    "default": FEWSHOT_DENSE_DESCRIPTION_INSTRUCTIONS,
    "terse": FEWSHOT_TERSE_INSTRUCTIONS,
    "motion_emphasis": FEWSHOT_MOTION_EMPHASIS_INSTRUCTIONS,
    "compare_explicit": FEWSHOT_COMPARE_EXPLICIT_INSTRUCTIONS,
    "checklist": FEWSHOT_CHECKLIST_INSTRUCTIONS,
}


def fewshot_instructions(cameras: tuple[str, ...], template_name: str = "default") -> str:
    """The one-time task/format instructions, sent as the first chat turn."""
    if template_name not in FEWSHOT_INSTRUCTION_TEMPLATES:
        raise ValueError(
            f"Unknown fewshot prompt template {template_name!r}. "
            f"Available: {', '.join(FEWSHOT_INSTRUCTION_TEMPLATES)}"
        )
    camera_labels = "\n".join(f"{cam}: <...>" for cam in cameras)
    return FEWSHOT_INSTRUCTION_TEMPLATES[template_name].format(camera_labels=camera_labels)


def fewshot_clip_header(
    *,
    language_instruction: str,
    start_step: int,
    end_step: int,
    total: int,
    clip_seconds: float,
    include_instruction: bool = True,
) -> str:
    """The short per-clip context sent alongside a clip's images (example or query)."""
    header = f'Clip: steps {start_step}-{end_step} of {total} (~{clip_seconds:g}s).'
    if include_instruction:
        header += f' Instruction: "{language_instruction}".'
    return header


def format_fewshot_answer(shared: dict, per_view: dict, cameras: tuple[str, ...]) -> str:
    """Render hand-written (or parsed) shared/per_view fields as the exact
    answer text the model is taught to produce."""
    per_view_lines = "\n".join(f"{cam}: {str(per_view.get(cam, '')).strip()}" for cam in cameras)
    return (
        "Shared:\n"
        f"Gripper: {str(shared.get('gripper', '')).strip()}\n"
        f"Scene change: {str(shared.get('scene_change', '')).strip()}\n"
        f"Terminal gripper state: {str(shared.get('terminal_gripper_state', '')).strip()}\n\n"
        "Per-view:\n"
        f"{per_view_lines}"
    )


_SHARED_FIELD_RE = re.compile(r"^(Gripper|Scene change|Terminal gripper state):\s*(.*)$", re.IGNORECASE)
_SHARED_KEY_MAP = {
    "gripper": "gripper",
    "scene change": "scene_change",
    "terminal gripper state": "terminal_gripper_state",
}


def parse_fewshot_response(text: str, cameras: tuple[str, ...]) -> dict:
    """Parse a 'Shared: ...\\n\\nPer-view: ...' response into
    {"shared": {...}, "per_view": {cam: ...}}."""
    shared = {"gripper": "", "scene_change": "", "terminal_gripper_state": ""}
    per_view = {cam: "" for cam in cameras}

    section: str | None = None
    current_key: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if current_key is None:
            return
        value = " ".join(buf).strip()
        if section == "shared" and current_key in shared:
            shared[current_key] = value
        elif section == "per_view" and current_key in per_view:
            per_view[current_key] = value

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower().rstrip(":")
        if lowered == "shared":
            flush()
            section, current_key, buf = "shared", None, []
            continue
        if lowered in ("per-view", "per view"):
            flush()
            section, current_key, buf = "per_view", None, []
            continue
        if section == "shared":
            match = _SHARED_FIELD_RE.match(line)
            if match:
                flush()
                current_key = _SHARED_KEY_MAP[match.group(1).lower()]
                buf = [match.group(2)]
                continue
        elif section == "per_view":
            head, sep, rest = line.partition(":")
            if sep and head.strip() in per_view:
                flush()
                current_key = head.strip()
                buf = [rest.strip()]
                continue
        if current_key is not None:
            buf.append(line)
    flush()
    return {"shared": shared, "per_view": per_view}
