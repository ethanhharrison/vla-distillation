"""Behavioural clustering: prompt construction, strict validation, representatives."""

from __future__ import annotations

import pytest

from pipeline.language_instruction.uniqueness import (
    DEFAULT_DEFINITION,
    DUPLICATE_DEFINITIONS,
    Clustering,
    ClusterResponseError,
    build_cluster_prompt,
    cluster_instructions,
    parse_clusters,
)
from pipeline.language_instruction.vlm import VLM

INSTRUCTIONS = [
    "Grab the clock and turn it so the face is downward",
    "Pick up the banana",
    "Place the clock face-down on the counter",
]


class ScriptedVLM(VLM):
    """Offline VLM returning a canned response and recording what it was asked."""

    def __init__(self, response: str):
        super().__init__("scripted")
        self.response = response
        self.prompts: list[str] = []
        self.images_seen: list[list[bytes]] = []

    def generate(self, prompt: str, images: list[bytes]) -> str:
        self.prompts.append(prompt)
        self.images_seen.append(list(images))
        self.usage.add(input_tokens=1000, output_tokens=50)
        return self.response


# --- prompt ---------------------------------------------------------------- #


def test_prompt_numbers_the_instructions_and_carries_the_definition():
    prompt = build_cluster_prompt(INSTRUCTIONS, step=0, total=439)

    assert "1. Grab the clock and turn it so the face is downward" in prompt
    assert "3. Place the clock face-down on the counter" in prompt
    assert DUPLICATE_DEFINITIONS[DEFAULT_DEFINITION] in prompt
    assert "1 to 3" in prompt          # the count the judge is held to
    assert "step 0 of 439" in prompt


@pytest.mark.parametrize("name", list(DUPLICATE_DEFINITIONS))
def test_every_registered_definition_can_be_selected(name):
    """A definition nobody can select is dead weight; a typo'd one must not pass."""
    assert DUPLICATE_DEFINITIONS[name] in build_cluster_prompt(
        INSTRUCTIONS, step=0, total=439, definition=name
    )


def test_an_unknown_definition_fails_rather_than_being_sent_as_the_criterion():
    with pytest.raises(ValueError, match="Unknown definition"):
        build_cluster_prompt(INSTRUCTIONS, step=0, total=439, definition="same_action")


def test_a_literal_definition_is_accepted_for_one_off_experiments():
    literal = "Two instructions are DUPLICATES when they rhyme."
    assert literal in build_cluster_prompt(INSTRUCTIONS, step=0, total=439, definition=literal)


# --- validation ------------------------------------------------------------ #


def test_parses_a_valid_partition_to_zero_based_indices():
    assert parse_clusters('{"clusters": [[1, 3], [2]]}', count=3) == [[0, 2], [1]]


def test_clusters_come_back_in_representative_order():
    assert parse_clusters('{"clusters": [[3], [1, 2]]}', count=3) == [[0, 1], [2]]


def test_tolerates_a_markdown_fence_and_surrounding_prose():
    assert parse_clusters('Sure:\n```json\n{"clusters": [[1],[2],[3]]}\n```', count=3) == [[0], [1], [2]]


def test_recovers_a_response_missing_its_closing_bracket():
    """Real failure: the judge emitted `...[3]}`, dropping the outer array's `]`."""
    assert parse_clusters('{"clusters": [[1], [2], [3]}', count=3) == [[0], [1], [2]]


def test_bracket_repair_only_adds_closers_and_leaves_valid_json_alone():
    from pipeline.language_instruction.uniqueness import close_brackets

    assert close_brackets('{"clusters": [[1], [2]]}') == '{"clusters": [[1], [2]]}'
    assert close_brackets('{"clusters": [[1], [2]}') == '{"clusters": [[1], [2]]}'
    assert close_brackets('{"clusters": [[1], [2]') == '{"clusters": [[1], [2]]}'
    # A bracket inside a string is text, not structure.
    assert close_brackets('{"note": "a ] b"}') == '{"note": "a ] b"}'


def test_a_repaired_response_is_still_validated_in_full():
    """Fixing syntax must not smuggle through a partition that misses an index."""
    with pytest.raises(ClusterResponseError, match="missing ids"):
        parse_clusters('{"clusters": [[1], [2]}', count=3)


@pytest.mark.parametrize(
    "response, message",
    [
        ('{"clusters": [[1, 2], [3, 9]]}', "hallucinated ids"),
        ('{"clusters": [[1, 2], [2, 3]]}', "more than one cluster"),
        ('{"clusters": [[1, 2]]}', "missing ids"),
        ('{"clusters": [[], [1, 2, 3]]}', "non-empty list"),
        ('{"clusters": [["one", 2, 3]]}', "non-integer id"),
        ('{"groups": [[1, 2, 3]]}', "no 'clusters' list"),
        ('{"clusters": [[1, 2] [3]]}', "not valid JSON"),  # missing comma: not a bracket slip
        ("instructions 1 and 2 are the same", "no JSON object"),
    ],
)
def test_a_partition_that_does_not_describe_our_list_fails_loudly(response, message):
    """Never silently repair: a patched partition is a wrong answer we can't detect."""
    with pytest.raises(ClusterResponseError, match=message):
        parse_clusters(response, count=3)


# --- representatives ------------------------------------------------------- #


def test_representative_is_the_earliest_member_and_duplicates_point_at_it():
    clustering = Clustering(clusters=[[0, 2], [1]])

    assert clustering.representatives == [0, 1]
    assert clustering.duplicate_of == {2: 0}
    assert clustering.num_duplicates == 1


def test_all_singletons_means_nothing_was_dropped():
    clustering = Clustering(clusters=[[0], [1], [2]])

    assert clustering.duplicate_of == {}
    assert clustering.num_duplicates == 0


# --- the call -------------------------------------------------------------- #


def test_cluster_instructions_makes_one_call_and_passes_the_frames():
    judge = ScriptedVLM('{"clusters": [[1, 3], [2]]}')
    frames = [b"jpeg-shoulder-1", b"jpeg-shoulder-2", b"jpeg-wrist"]

    clustering = cluster_instructions(judge, INSTRUCTIONS, frames, step=0, total=439)

    assert clustering.clusters == [[0, 2], [1]]
    assert clustering.duplicate_of == {2: 0}
    assert len(judge.prompts) == 1
    assert judge.images_seen == [frames]
    assert clustering.raw_response == '{"clusters": [[1, 3], [2]]}'


@pytest.mark.parametrize("instructions", [[], ["only one"]])
def test_fewer_than_two_instructions_costs_nothing(instructions):
    judge = ScriptedVLM("should never be called")

    clustering = cluster_instructions(judge, instructions, [], step=0, total=439)

    assert judge.prompts == []
    assert clustering.representatives == list(range(len(instructions)))
    assert clustering.num_duplicates == 0


def test_an_invalid_partition_propagates_out_of_the_call():
    judge = ScriptedVLM('{"clusters": [[1, 2]]}')

    with pytest.raises(ClusterResponseError, match="missing ids"):
        cluster_instructions(judge, INSTRUCTIONS, [], step=0, total=439)
