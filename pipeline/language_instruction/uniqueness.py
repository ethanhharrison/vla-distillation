"""Behavioural-uniqueness clustering for instruction lists.

Where the per-call adherence judge only checks feasibility/style, this module
partitions an instruction list into **behaviour clusters** and keeps one
representative per cluster.

Two instructions are duplicates iff executing each in this scene would leave the
world in substantially the same end state. That is a stronger claim than
"reworded" — "turn the clock so the face is downward" and "place the clock
face-down on the counter" share no distinctive wording but end identically — and
a weaker one than "identical text".

Why a partition rather than a per-instruction score:

- it says *what* each duplicate was folded into, so a merge can be inspected and
  reversed, instead of a bare score that says "redundant with something";
- there is no threshold to tune — the verdict is binary by construction;
- the response is checkable. A partition of 1..n either covers every number
  exactly once or it does not, so a judge that hallucinates, doubles or drops an
  index fails loudly here rather than silently passing the instruction through.

Pipeline can run this **before** merge (each generate call independently) or
**after** merge (once over the unified list). `cluster_instructions` is the
single-list API; `cluster_by_step` walks a step->instructions map (merge-time).
Mutation of `StepInstructions` lives in `generate.apply_uniqueness_to_steps`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .vlm import VLM, build_vlm

# --------------------------------------------------------------------------- #
# The definitions. These decide what the stage considers a duplicate, and they
# are the main thing worth tuning — compare them with
# `scripts/tune_uniqueness_prompt.py` before changing the default.
# --------------------------------------------------------------------------- #

END_STATE_DEFINITION = """Two instructions are DUPLICATES if and only if \
executing each of them in this scene would leave the world in substantially the \
same end state.

- Manner, speed and style modifiers collapse: "slowly open the drawer" and "open \
the drawer" are DUPLICATES, because both end with the drawer open.
- Sub-goal prefixes are DISTINCT: "reach toward the handle" and "open the drawer" \
are NOT duplicates, because the first stops short of the second's end state.
- Object references resolve against the scene: "pick up the cup" and "pick up the \
object on the left" are DUPLICATES if and only if the cup really is the object on \
the left in this scene.
- The verdict is binary: DUPLICATE or DISTINCT, with no degrees in between."""


#: Criterion is the *trajectory* rather than the end state: would the arm do the
#: same thing? For generating action supervision that is arguably what matters —
#: two instructions producing the same motion are the same training signal,
#: whatever the world looks like afterwards. It also separates cases the
#: end-state wording conflates, e.g. "clear the dishes from the counter" and
#: "put the dishes in the sink" can share an end state while naming different work.
SAME_ACTIONS_DEFINITION = """Two instructions are DUPLICATES if and only if a \
robot carrying out one would perform substantially the same actions as a robot \
carrying out the other: it would grasp or touch the same objects, and move them \
the same way, to the same place.

- Manner, speed and style modifiers collapse: "slowly open the drawer" and "open \
the drawer" are DUPLICATES, because the arm does the same thing to the same drawer.
- Different objects means DISTINCT, even when the wording barely changes: "pick \
up the fork" and "pick up the spoon" are not duplicates.
- Different destinations means DISTINCT: "put the cup in the sink" and "put the \
cup on the counter" are not duplicates.
- Sub-goal prefixes are DISTINCT: "reach toward the handle" and "open the drawer" \
are not duplicates, because the first stops short of the work the second does.
- Object references resolve against the scene: "pick up the cup" and "pick up the \
object on the left" are DUPLICATES if and only if the cup really is the object on \
the left here.
- The verdict is binary: DUPLICATE or DISTINCT, with no degrees in between."""


#: SAME_ACTIONS plus one rule about collective nouns. Motivating case, from a real
#: run: "put the dirty forks and spoons into the sink" vs "place the dirty
#: silverware into the sink" — the same objects under a group name, which the
#: judge kept reading as different objects.
SAME_ACTIONS_COLLECTIVE_DEFINITION = SAME_ACTIONS_DEFINITION.replace(
    "- The verdict is binary:",
    """- A group name and the things it covers are the SAME objects when the scene \
contains nothing else in that group: if the only utensils present are forks and \
spoons, then "the silverware", "the utensils" and "the forks and spoons" all name \
the same objects, and instructions using them are duplicates of each other.
- The verdict is binary:""",
)


#: SAME_ACTIONS with the tie-break flipped toward merging. The other definitions
#: leave an unstated "when unsure, DISTINCT" that the judge follows readily — we
#: measured an earlier version answering DISTINCT on 11/11 probe pairs — so this
#: is the control for "are we under-merging?". Expect it to over-merge; the
#: interesting question is which extra merges it finds and whether they are wrong.
SAME_ACTIONS_LENIENT_DEFINITION = SAME_ACTIONS_DEFINITION.replace(
    "- The verdict is binary: DUPLICATE or DISTINCT, with no degrees in between.",
    """- Do not require the wording to match, or the two descriptions to be equally \
specific. A vague instruction and a detailed one are duplicates when the arm's \
work is the same: "tidy the counter" and "move the mug and the plate to the sink" \
are duplicates if the mug and plate are all there is to tidy.
- When the two instructions would have the arm doing broadly the same work and \
you are unsure, answer DUPLICATE. Reserve DISTINCT for cases where the difference \
would genuinely change which objects are touched or where they end up.
- The verdict is binary: DUPLICATE or DISTINCT, with no degrees in between.""",
)


#: The lenient tie-break finds duplicates the strict ones miss, but it also
#: overrides the object/destination rules it is supposed to sit beneath — it
#: merged "clear the top of the toaster oven" with "put the red container in the
#: kitchen cabinet" (same object, different destination), which plain
#: SAME_ACTIONS correctly refused. This version scopes the leniency to *wording
#: and specificity* and says explicitly that it does not override those rules.
SAME_ACTIONS_LENIENT_WORDING_DEFINITION = SAME_ACTIONS_DEFINITION.replace(
    "- The verdict is binary: DUPLICATE or DISTINCT, with no degrees in between.",
    """- Wording and level of detail do not matter. A vague instruction and a \
detailed one are duplicates when the arm's work is the same: "tidy the counter" \
and "move the mug and the plate to the sink" are duplicates if the mug and plate \
are all there is to tidy. When two instructions describe the same work in \
different words and you are unsure, answer DUPLICATE.
- That leniency is about wording only. It never overrides the rules above: if the \
two instructions would have the arm touching different objects, or leaving them \
in different places, they are DISTINCT however similar they sound.
- The verdict is binary: DUPLICATE or DISTINCT, with no degrees in between.""",
)


# --------------------------------------------------------------------------- #
# The short ones. The definitions above kept growing bullets, and measuring them
# showed extra rules did nothing while the overall stance did everything — so
# these say one thing plainly instead.
#
# They also change the criterion. "Would the arm do the same thing" invites the
# judge to reason about geometry and decide that "move the towel to the left" and
# "move the towel toward the oven" come out in the same place. These ask only
# whether the two instructions *mean* the same thing, and use the image for one
# job: telling whether two descriptions name the same object.
# --------------------------------------------------------------------------- #

SEMANTIC_DEFINITION = """Two instructions are DUPLICATES if they mean the same \
thing: the same action, on the same objects, to the same stated destination — \
the same command in different words.

Use the image for one thing only: deciding whether two descriptions name the same \
object, for example whether "the black mug" and "the black cup" are one cup.

If the two instructions state the destination differently, they are DISTINCT, \
even if the robot might end up doing something similar. "Move the towel to the \
left" and "move the towel next to the oven" are DISTINCT."""


#: SEMANTIC plus one sentence on group words, the case the scoring judge got wrong
#: ("the dirty silverware" vs "the dirty forks and spoons", same sink).
SEMANTIC_GROUPS_DEFINITION = SEMANTIC_DEFINITION + """

A group word and the items it covers name the same objects when the scene holds \
nothing else in that group: with only forks and spoons on the counter, "the \
silverware" and "the forks and spoons" are the same objects."""


#: The strictest reading: duplicates are rewordings, nothing more.
PARAPHRASE_DEFINITION = """Two instructions are DUPLICATES only if one is a \
reworded version of the other — the same command, said differently.

Use the image for one thing only: deciding whether two descriptions name the same \
object.

A different object, a different destination, or a different amount of work all \
mean DISTINCT."""


#: name -> definition text, selectable via the `uniqueness.definition` config key
#: (mirrors INSTRUCTION_TEMPLATES in prompts.py).
DUPLICATE_DEFINITIONS: dict[str, str] = {
    "semantic": SEMANTIC_DEFINITION,
    "semantic_groups": SEMANTIC_GROUPS_DEFINITION,
    "paraphrase": PARAPHRASE_DEFINITION,
    "end_state": END_STATE_DEFINITION,
    "same_actions": SAME_ACTIONS_DEFINITION,
    "same_actions_collective": SAME_ACTIONS_COLLECTIVE_DEFINITION,
    "same_actions_lenient": SAME_ACTIONS_LENIENT_DEFINITION,
    "same_actions_lenient_wording": SAME_ACTIONS_LENIENT_WORDING_DEFINITION,
}

#: Chosen on measured behaviour, not taste. Against hand-labelled pairs in
#: tests/fixtures, `semantic_groups` matched the best recall on the vague set
#: while merging none of the traps -- same object, different *stated*
#: destination -- that `same_actions` and `end_state` fell for. Re-check with
#: scripts/tune_uniqueness_prompt.py before changing it.
DEFAULT_DEFINITION = "semantic_groups"


def resolve_definition(name_or_text: str) -> tuple[str, str]:
    """Return (name, text) for a registered definition name, or a literal definition.

    A literal must mention DUPLICATES, so a typo'd name fails loudly instead of
    being sent to the judge as the whole criterion.
    """
    if name_or_text in DUPLICATE_DEFINITIONS:
        return name_or_text, DUPLICATE_DEFINITIONS[name_or_text]
    if "DUPLICATES" in name_or_text:
        return "custom", name_or_text
    raise ValueError(
        f"Unknown definition {name_or_text!r}; not a registered name "
        f"({', '.join(DUPLICATE_DEFINITIONS)}) and not a literal definition "
        "mentioning DUPLICATES."
    )


UNIQUENESS_CLUSTER_PROMPT = """You are de-duplicating the instruction labels of a \
robot manipulation dataset.

The image(s) are the robot's camera view(s) at one moment in time (step {step} of \
{total}). Every instruction below was written about this exact scene. Resolve \
every object reference against what you can actually see: two instructions that \
name the same physical object with different words are talking about the same \
object, and two that name different physical objects are not — even when the \
wording is similar.

{definition}

Partition ALL of the numbered instructions below into behaviour clusters. Two \
instructions go in the same cluster if and only if they are DUPLICATES by the \
definition above. An instruction that duplicates nothing forms a cluster of one.

Instructions:
{numbered_instructions}

Output a single JSON object and nothing else — no prose, no markdown fence:

{{"clusters": [[1, 4], [2], [3, 5]]}}

Rules for the output:
- Use the instruction NUMBERS shown above, never the text.
- Every number from 1 to {count} must appear EXACTLY ONCE across all clusters.
- Do not invent numbers outside 1 to {count}."""


class ClusterResponseError(ValueError):
    """The judge's partition did not describe the list we sent. Always fail on this."""


@dataclass
class Clustering:
    """A partition of one instruction list, as indices into that list.

    Each cluster is sorted ascending and the clusters are ordered by their first
    member, so the representative of a cluster is its earliest-proposed member
    and the output is stable across runs.
    """

    clusters: list[list[int]] = field(default_factory=list)
    raw_response: str = ""

    @property
    def representatives(self) -> list[int]:
        return [cluster[0] for cluster in self.clusters]

    @property
    def duplicate_of(self) -> dict[int, int]:
        """Map each non-representative index to the index it was folded into."""
        return {
            member: cluster[0]
            for cluster in self.clusters
            for member in cluster[1:]
        }

    @property
    def num_duplicates(self) -> int:
        return sum(len(cluster) - 1 for cluster in self.clusters)


def build_cluster_prompt(
    instructions: list[str],
    step: int,
    total: int,
    definition: str = DEFAULT_DEFINITION,
    template: str = UNIQUENESS_CLUSTER_PROMPT,
) -> str:
    """Render the clustering prompt for a numbered instruction list."""
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(instructions, start=1))
    return template.format(
        step=step,
        total=total,
        definition=resolve_definition(definition)[1],
        numbered_instructions=numbered,
        count=len(instructions),
    )


def build_uniqueness_judge(
    provider: str = "gemini",
    model: str | None = None,
    temperature_zero: bool = True,
) -> VLM:
    """Build the clustering judge, deterministic by default.

    Sampling is pinned to 0 where the provider supports it, so re-running the
    same instructions gives the same partition. Without this, comparing two
    definitions measures sampling noise as much as the prompt.
    """
    kwargs: dict = {}
    if temperature_zero:
        key = provider.lower()
        if key in {"openai", "qwen"}:
            kwargs = {"temperature": 0}
        elif key == "gemini":
            from google.genai import types

            kwargs = {"config": types.GenerateContentConfig(temperature=0.0)}
    return build_vlm(provider, model=model, **kwargs)


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_CLOSERS = {"[": "]", "{": "}"}


def close_brackets(text: str) -> str:
    """Insert closing brackets the judge left out. Syntax only.

    Observed on a real 60-instruction call: the model emitted
    `{"clusters": [[1], ..., [60]}` — the closing `]` of the outer array is
    missing, so the response is one character from valid while the partition it
    describes is completely unambiguous. `finish_reason` was STOP, so this is a
    formatting slip, not truncation, and it gets more likely as the list grows.

    This only ever *adds* `]` or `}`. It never changes, drops or invents a
    number, and the partition is still validated in full afterwards — a repaired
    response that does not cover 1..n exactly once still fails loudly. Repairing
    syntax is safe; repairing a partition would not be.
    """
    out: list[str] = []
    stack: list[str] = []
    in_string = escape = False
    for char in text:
        if in_string:
            out.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _CLOSERS:
            stack.append(char)
        elif char in _CLOSERS.values():
            # Close any inner brackets the judge skipped over, then this one.
            while stack and _CLOSERS[stack[-1]] != char:
                out.append(_CLOSERS[stack.pop()])
            if stack:
                stack.pop()
        out.append(char)
    return "".join(out) + "".join(_CLOSERS[bracket] for bracket in reversed(stack))


def parse_clusters(text: str, count: int) -> list[list[int]]:
    """Parse a judge response into 0-based index clusters, validating the partition.

    Raises `ClusterResponseError` unless the numbers returned are exactly a
    partition of 1..count. Hallucinated numbers, numbers claimed by two clusters
    and dropped numbers are all hard failures: each one means the answer does not
    describe the list we sent, and a quietly repaired partition is a wrong answer
    that nothing downstream can detect.
    """
    stripped = _JSON_FENCE.sub("", text.strip())
    match = _JSON_OBJECT.search(stripped)
    candidate = match.group() if match else stripped[stripped.find("{"):] if "{" in stripped else ""
    if not candidate:
        raise ClusterResponseError(f"no JSON object in judge response: {text[:200]!r}")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            payload = json.loads(close_brackets(candidate))
        except json.JSONDecodeError as exc:
            raise ClusterResponseError(
                f"judge response was not valid JSON ({exc}): {text[:200]!r}"
            ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("clusters"), list):
        raise ClusterResponseError(f"judge response has no 'clusters' list: {text[:200]!r}")

    clusters: list[list[int]] = []
    seen: set[int] = set()
    out_of_range: list[int] = []
    overlapping: list[int] = []

    for position, raw_cluster in enumerate(payload["clusters"]):
        if not isinstance(raw_cluster, list) or not raw_cluster:
            raise ClusterResponseError(f"cluster {position} is not a non-empty list: {raw_cluster!r}")
        members: list[int] = []
        for raw_id in raw_cluster:
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise ClusterResponseError(f"cluster {position} has a non-integer id: {raw_id!r}")
            if not 1 <= raw_id <= count:
                out_of_range.append(raw_id)
            elif raw_id in seen:
                overlapping.append(raw_id)
            else:
                seen.add(raw_id)
                members.append(raw_id - 1)
        if members:
            clusters.append(sorted(members))

    problems = []
    if out_of_range:
        problems.append(f"hallucinated ids {sorted(set(out_of_range))} (valid range 1..{count})")
    if overlapping:
        problems.append(f"ids in more than one cluster {sorted(set(overlapping))}")
    missing = [i for i in range(1, count + 1) if i not in seen]
    if missing:
        problems.append(f"missing ids {missing}")
    if problems:
        raise ClusterResponseError("judge returned an invalid partition: " + "; ".join(problems))

    return sorted(clusters, key=lambda cluster: cluster[0])


def cluster_instructions(
    judge: VLM,
    instructions: list[str],
    images: list[bytes],
    step: int,
    total: int,
    definition: str = DEFAULT_DEFINITION,
    template: str = UNIQUENESS_CLUSTER_PROMPT,
) -> Clustering:
    """One judge call: partition `instructions` into behaviour clusters.

    Fewer than two instructions needs no call and costs nothing.
    """
    if len(instructions) < 2:
        return Clustering(clusters=[[i] for i in range(len(instructions))])

    prompt = build_cluster_prompt(
        instructions, step=step, total=total, definition=definition, template=template
    )
    raw_response = judge.generate(prompt, images)
    return Clustering(
        clusters=parse_clusters(raw_response, len(instructions)),
        raw_response=raw_response,
    )


def load_step_images(image_paths: dict[str, str]) -> list[bytes]:
    """Read back frames from paths (camera order), skipping unreadable files."""
    images: list[bytes] = []
    for path in image_paths.values():
        try:
            images.append(Path(path).read_bytes())
        except OSError:
            continue
    return images


def apply_clustering(
    instructions: list[str],
    clustering: Clustering,
) -> tuple[list[str], dict[str, str]]:
    """Return (representatives, dropped->representative) for a clustered list."""
    reps = [instructions[i] for i in clustering.representatives]
    dups = {
        instructions[member]: instructions[representative]
        for member, representative in clustering.duplicate_of.items()
    }
    return reps, dups


def cluster_by_step(
    instructions_by_step: dict[int, list[str]],
    image_paths_by_step: dict[int, dict[str, str]],
    *,
    judge: VLM,
    trajectory_length: int,
    definition: str = DEFAULT_DEFINITION,
    label: str = "uniqueness",
) -> dict[int, Clustering]:
    """Cluster each step's instruction list; one judge call per step with ≥2 items.

    Does not mutate `instructions_by_step`. Callers that want only
    representatives should use `apply_clustering` on each list.
    """
    clustering_by_step: dict[int, Clustering] = {}
    for step, instructions in instructions_by_step.items():
        if len(instructions) < 2:
            clustering_by_step[step] = Clustering(
                clusters=[[i] for i in range(len(instructions))]
            )
            continue
        images = load_step_images(image_paths_by_step.get(step, {}))
        if not images:
            print(
                f"[{label}] step {step}: no frames on disk, judging from text alone "
                "(set save_images: true to ground object references)"
            )
        clustering = cluster_instructions(
            judge=judge,
            instructions=instructions,
            images=images,
            step=step,
            total=trajectory_length,
            definition=definition,
        )
        clustering_by_step[step] = clustering
        print(
            f"[{label}] step {step}: {len(instructions)} -> "
            f"{len(clustering.clusters)} instructions "
            f"({clustering.num_duplicates} folded into a representative)"
        )
    return clustering_by_step
