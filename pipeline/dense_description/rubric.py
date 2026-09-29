"""Correctness rubric for dense descriptions, and a VLM judge that applies it.

A description PASSES only if every criterion passes; any single failed
criterion makes the whole description a failure. The criteria target what a
downstream subgoal-image generator needs to get right: if the description
would lead it to draw something that is not in the real end frame (wrong
object, wrong gripper pose, wrong object placement, an action that never
happened), that is a failure.

The judge sees privileged evidence the describer did not: the task
instruction and intermediate frames from the clip, in addition to the
start/end stills. Use `judge_description` to grade one description, and
`evidence_from_trajectory` / `evidence_from_video` to build its images.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass

from PIL import Image

from pipeline.language_instruction.vlm import VLM, ChatTurn

DEFAULT_JUDGE_PROVIDER = "openai"
DEFAULT_JUDGE_MODEL = "gpt-6-astra"
DEFAULT_JUDGE_EFFORT = "xhigh"


@dataclass(frozen=True)
class Criterion:
    id: str
    name: str
    rule: str


# Order matters only for display. Every criterion is pass/fail.
RUBRIC: tuple[Criterion, ...] = (
    Criterion(
        "object_naming",
        "Every object is correctly named",
        "Every object the description mentions actually exists in the frames and is named with the "
        "correct category (a protein bar is not a 'toothpaste tube'; a spoon is not a 'fork'). Any "
        "attribute it states - color, material, printed text, size, count - matches what is visible. "
        "Hallucinated objects or wrong attributes FAIL. Generic names that are still true ('object', "
        "'item', 'cloth') pass; a more specific name that is wrong fails.",
    ),
    Criterion(
        "manipulated_object",
        "Correct object is manipulated",
        "The object(s) the gripper contacts, grasps, pushes or carries are correctly identified, "
        "including WHICH instance when several similar objects are present (e.g. the silver spoon vs "
        "the white plastic spoon). Saying the gripper interacts with a different object, or never "
        "naming the object it interacts with, FAILS.",
    ),
    Criterion(
        "terminal_gripper_state",
        "Final gripper state",
        "What the description says about the gripper at the END frame is correct: open vs closed, and "
        "what (if anything) it is holding. Saying it released an object it still holds, holds "
        "something it does not, or is empty when it is holding something FAILS; so does not "
        "stating the final state at all.",
    ),
    Criterion(
        "terminal_gripper_position",
        "Final gripper position",
        "Where the description places the gripper at the END frame matches: which object/region it is "
        "over or in, and its rough height (touching the surface, low above it, raised high). E.g. "
        "'raised high above the counter' when it is still inside the bowl FAILS, and so does giving "
        "no final location at all.",
    ),
    Criterion(
        "object_end_states",
        "Final object positions and states",
        "Every object described as moving ends where and how the description says (location, "
        "containment, orientation, folded/draped/upright). Objects described as stationary did not "
        "move noticeably. Any object that visibly moved but is described wrongly FAILS.",
    ),
    Criterion(
        "action_fidelity",
        "Actions actually happened",
        "The net change the description claims from the START frame to the END frame really happened: "
        "no invented outcome (e.g. released, placed, lifted out, scooped, wiped away) and no omitted "
        "outcome that is visible in the end frame (e.g. an object that ends grasped or lifted). The "
        "direction of the gross motion is right. Completing the task instruction early - describing "
        "steps that happen after the clip ends - FAILS. Transient mid-clip events that leave no trace "
        "in the end frame (a brief open-and-regrasp, a lift then lower) are neither required nor "
        "penalized.",
    ),
    Criterion(
        "per_view_consistency",
        "Per-view descriptions match each camera",
        "Each per-view description is consistent with THAT camera's start and end frames: image-space "
        "directions (left/right/up/down), what is visible or out of frame, and where things are in "
        "that view at the end. A clear contradiction in any single view FAILS.",
    ),
    Criterion(
        "completeness",
        "Complete and specific",
        "Gripper, Scene change, Terminal gripper state, and every per-view field are present and "
        "describe THIS clip specifically: the manipulated object, the gripper's motion, and the end "
        "state are all stated concretely enough to picture the end frame. Placeholder, generic, or "
        "vague text that could describe any clip FAILS, even if nothing in it is false.",
    ),
)

JUDGE_INSTRUCTIONS = """You are grading a dense description of a short robot-manipulation clip for factual correctness.
The description will be used to generate an image of the clip's END state, so anything it gets wrong about the scene is a failure.

The describer saw ONLY the START and END frames. You additionally get privileged evidence: the task instruction and frames from the middle of the clip, labeled in order below. Use them only to establish what really happens (e.g. which object is grasped, whether an object is actually held at the end). Grade the description against the START and END frames: what matters is whether its account of the net change and of the end state is true. Do not fail it for missing a transient mid-clip event that leaves no trace in the end frame. The task instruction describes the whole episode; this clip may cover only part of it.

Grade the description against each criterion independently. A criterion FAILS if any statement in the description clearly contradicts the frames under that criterion. Omissions count too: a description that leaves out or is vague about what the gripper holds, where it ends, or which object it manipulates fails the corresponding criterion. A statement that the start/end frames cannot confirm or refute (hedged wording like "appears to", "no release is visible", ambiguous timing of a grasp) does not fail on its own. Be strict: one clear contradiction is enough to fail.

Criteria:
{criteria}

Evidence images, in order:
{image_legend}

Task instruction (privileged): "{instruction}"

Description to grade:
<<<
{description}
>>>

First write down, for yourself, what actually happens in the clip, what the gripper holds at the end and where it is, and every object the description names. Then answer with ONLY a JSON object:
{{
  "ground_truth": "<2-3 sentences: what actually happens, final gripper state and position>",
  "objects": [{{"mentioned": "<name used in description>", "actual": "<what it really is, or 'not present'>", "correct": true/false}}],
  "criteria": {{
    "<criterion id>": {{"pass": true/false, "reason": "<one sentence; quote the offending phrase when failing>"}}
  }}
}}
Include every criterion id: {criterion_ids}."""


def rubric_text() -> str:
    return "\n".join(f"- {c.id} ({c.name}): {c.rule}" for c in RUBRIC)


def build_judge_prompt(description: str, instruction: str, image_labels: list[str]) -> str:
    legend = "\n".join(f"{i + 1}. {label}" for i, label in enumerate(image_labels))
    return JUDGE_INSTRUCTIONS.format(
        criteria=rubric_text(),
        image_legend=legend,
        instruction=instruction,
        description=description.strip(),
        criterion_ids=", ".join(c.id for c in RUBRIC),
    )


def parse_judge_response(text: str) -> dict:
    """Parse the judge's JSON and compute the overall verdict ourselves:
    PASS iff every rubric criterion is present and passes. A criterion the
    judge omitted counts as a failure."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(match.group(0) if match else text)
    raw = data.get("criteria") or {}
    criteria = {}
    for c in RUBRIC:
        entry = raw.get(c.id)
        if not isinstance(entry, dict) or "pass" not in entry:
            criteria[c.id] = {"pass": False, "reason": "judge did not grade this criterion"}
        else:
            criteria[c.id] = {"pass": bool(entry["pass"]), "reason": str(entry.get("reason", ""))}
    failed = [cid for cid, v in criteria.items() if not v["pass"]]
    return {
        "pass": not failed,
        "failed_criteria": failed,
        "criteria": criteria,
        "objects": data.get("objects") or [],
        "ground_truth": data.get("ground_truth", ""),
    }


def judge_description(
    vlm: VLM,
    description: str,
    instruction: str,
    evidence: list[tuple[str, bytes]],
) -> dict:
    """Grade one description. `evidence` is [(label, jpeg)] in display order,
    e.g. from evidence_from_trajectory / evidence_from_video."""
    labels = [label for label, _ in evidence]
    prompt = build_judge_prompt(description, instruction, labels)
    raw = vlm.generate_chat([ChatTurn(role="user", text=prompt, images=[img for _, img in evidence])])
    verdict = parse_judge_response(raw)
    verdict["judge_model"] = vlm.model
    verdict["judge_usage"] = getattr(vlm, "last_usage", None)
    return verdict


def _mid_steps(start_step: int, end_step: int, n_mid: int) -> list[int]:
    span = end_step - start_step
    return [start_step + round(span * (i + 1) / (n_mid + 1)) for i in range(n_mid)]


def evidence_from_stills(stills: dict[str, dict[str, bytes]], cameras: list[str], which: str) -> list[tuple[str, bytes]]:
    return [(f"{which.upper()} frame, camera {cam}", stills[which][cam]) for cam in cameras if cam in stills[which]]


def evidence_from_trajectory(
    trajectory,
    start_step: int,
    end_step: int,
    cameras: list[str],
    stills: dict[str, dict[str, bytes]],
    n_mid: int = 3,
) -> list[tuple[str, bytes]]:
    """Start stills, n_mid intermediate frames per camera, then end stills.
    `stills` are the exact start/end images the describer saw."""
    from .imaging import DEFAULT_ROTATE_180_CAMERAS, maybe_rotate

    evidence = evidence_from_stills(stills, cameras, "start")
    for step in _mid_steps(start_step, end_step, n_mid):
        frame = trajectory.frame(step, tuple(cameras))
        for cam in cameras:
            if cam in frame:
                evidence.append((f"MID frame (step {step}), camera {cam}",
                                 maybe_rotate(frame[cam], cam, DEFAULT_ROTATE_180_CAMERAS)))
    return evidence + evidence_from_stills(stills, cameras, "end")


def evidence_from_video(
    video: bytes,
    cameras: list[str],
    stills: dict[str, dict[str, bytes]],
    n_mid: int = 3,
) -> list[tuple[str, bytes]]:
    """Like evidence_from_trajectory, but intermediate frames come from a
    rendered clip video (views stacked top-to-bottom in `cameras` order)."""
    import imageio.v3 as iio

    frames = iio.imread(video, extension=".mp4")
    evidence = evidence_from_stills(stills, cameras, "start")
    for idx in _mid_steps(0, len(frames) - 1, n_mid):
        buf = io.BytesIO()
        Image.fromarray(frames[idx]).save(buf, format="JPEG", quality=90)
        evidence.append((
            f"MID frame ({idx + 1}/{len(frames)} of the clip), all cameras stacked top-to-bottom: {' / '.join(cameras)}",
            buf.getvalue(),
        ))
    return evidence + evidence_from_stills(stills, cameras, "end")
