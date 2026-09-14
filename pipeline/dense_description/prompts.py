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


def fewshot_instructions(cameras: tuple[str, ...]) -> str:
    """The one-time task/format instructions, sent as the first chat turn."""
    camera_labels = "\n".join(f"{cam}: <...>" for cam in cameras)
    return FEWSHOT_DENSE_DESCRIPTION_INSTRUCTIONS.format(camera_labels=camera_labels)


def fewshot_clip_header(
    *,
    language_instruction: str,
    start_step: int,
    end_step: int,
    total: int,
    clip_seconds: float,
) -> str:
    """The short per-clip context sent alongside a clip's images (example or query)."""
    return (
        f'Clip: steps {start_step}-{end_step} of {total} (~{clip_seconds:g}s). '
        f'Instruction: "{language_instruction}".'
    )


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
