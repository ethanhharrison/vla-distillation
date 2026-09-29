"""Correctness rubric: prompt construction and pass/fail verdict parsing."""

from __future__ import annotations

import json

from pipeline.dense_description.rubric import (
    RUBRIC,
    build_judge_prompt,
    judge_description,
    parse_judge_response,
)
from pipeline.language_instruction.vlm import VLM


def _response(overrides: dict[str, bool] | None = None, drop: str | None = None) -> str:
    criteria = {c.id: {"pass": True, "reason": "ok"} for c in RUBRIC}
    for cid, passed in (overrides or {}).items():
        criteria[cid] = {"pass": passed, "reason": "because"}
    if drop:
        del criteria[drop]
    return json.dumps({"ground_truth": "gt", "objects": [], "criteria": criteria})


def test_all_criteria_pass_means_pass():
    verdict = parse_judge_response(_response())
    assert verdict["pass"] is True
    assert verdict["failed_criteria"] == []


def test_any_single_failed_criterion_fails_the_description():
    verdict = parse_judge_response(_response({"object_naming": False}))
    assert verdict["pass"] is False
    assert verdict["failed_criteria"] == ["object_naming"]


def test_criterion_the_judge_omits_counts_as_failure():
    verdict = parse_judge_response(_response(drop="terminal_gripper_position"))
    assert verdict["pass"] is False
    assert verdict["failed_criteria"] == ["terminal_gripper_position"]


def test_overall_verdict_ignores_judges_own_pass_field():
    data = json.loads(_response({"action_fidelity": False}))
    data["pass"] = True
    assert parse_judge_response(json.dumps(data))["pass"] is False


def test_parses_json_wrapped_in_prose_or_fences():
    verdict = parse_judge_response("Here you go:\n```json\n" + _response() + "\n```")
    assert verdict["pass"] is True


def test_prompt_lists_every_criterion_and_image_label():
    prompt = build_judge_prompt("Shared: ...", "wipe the stove", ["START frame, camera a", "END frame, camera a"])
    for c in RUBRIC:
        assert c.id in prompt
    assert "1. START frame, camera a" in prompt and "2. END frame, camera a" in prompt
    assert "wipe the stove" in prompt


class _FakeJudge(VLM):
    def __init__(self, reply: str):
        super().__init__("fake-judge")
        self.reply = reply
        self.turns = None

    def generate(self, prompt, images):
        raise NotImplementedError

    def generate_chat(self, turns):
        self.turns = turns
        return self.reply


def test_judge_description_sends_all_evidence_images_in_order():
    judge = _FakeJudge(_response({"terminal_gripper_state": False}))
    evidence = [("START frame", b"s"), ("MID frame", b"m"), ("END frame", b"e")]
    verdict = judge_description(judge, "Shared: ...", "instr", evidence)
    assert judge.turns[0].images == [b"s", b"m", b"e"]
    assert verdict["pass"] is False and verdict["judge_model"] == "fake-judge"
