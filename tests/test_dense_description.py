"""Dense description clip splitting, prompting, and output round-trip."""

from __future__ import annotations

from pathlib import Path

from pipeline.dense_description.generate import (
    ClipDescription,
    DenseConfig,
    DenseResult,
    clip_boundaries,
    load_config,
    ordered_clip_images,
    resolve_language_instruction,
    write_txt,
)
from pipeline.dense_description.prompts import build_dense_prompt, parse_description
from pipeline.language_instruction.trajectory import Trajectory
from pipeline.language_instruction.vlm import VLM


def test_clip_boundaries_non_overlapping_two_second_clips():
    # 15 fps * 2s = 30 frames per clip; inclusive end index is start+29.
    assert clip_boundaries(90, 30) == [(0, 29), (30, 59), (60, 89)]


def test_clip_boundaries_keeps_partial_tail_with_at_least_two_frames():
    assert clip_boundaries(35, 30) == [(0, 29), (30, 34)]
    assert clip_boundaries(32, 30) == [(0, 29), (30, 31)]


def test_clip_boundaries_drops_singleton_tail():
    # start=30, end=min(59, 30)=30 → end <= start → stop without appending
    assert clip_boundaries(31, 30) == [(0, 29)]
    assert clip_boundaries(1, 30) == []
    assert clip_boundaries(0, 30) == []


def test_clip_boundaries_exact_multiple():
    assert clip_boundaries(60, 30) == [(0, 29), (30, 59)]


def test_resolve_language_instruction_prefers_override_then_metadata():
    meta = {
        "language_instruction1": "Cover the green object",
        "language_instruction2": "Pick up the towel",
    }
    assert resolve_language_instruction(meta) == "Cover the green object"
    assert (
        resolve_language_instruction(meta, "do the thing") == "do the thing"
    )
    try:
        resolve_language_instruction({})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_build_dense_prompt_mentions_start_end_and_language():
    prompt = build_dense_prompt(
        language_instruction="pick up the banana",
        cameras=("shoulder_image_1", "wrist_image"),
        start_step=0,
        end_step=29,
        total=100,
        clip_seconds=2,
        fps=15,
    )
    assert "pick up the banana" in prompt
    assert "shoulder_image_1, wrist_image" in prompt
    assert "steps 0–29" in prompt or "steps 0-29" in prompt
    assert "START" in prompt and "END" in prompt
    assert "For EACH camera separately" in prompt
    assert "at least one full sentence" in prompt
    assert "numeric" not in prompt.lower()
    assert "5 cm" not in prompt


def test_parse_description_strips_wrappers():
    assert parse_description('  "The arm lifts the cup."  ') == "The arm lifts the cup."
    assert (
        parse_description("Here is the description:\nThe arm lifts the cup.")
        == "The arm lifts the cup."
    )


def test_ordered_clip_images_are_start_then_end():
    traj = Trajectory(
        record_path="x.tfrecord",
        length=5,
        images={
            "a": [b"a0", b"a1", b"a2", b"a3", b"a4"],
            "b": [b"b0", b"b1", b"b2", b"b3", b"b4"],
        },
    )
    assert ordered_clip_images(traj, 0, 2, ("a", "b")) == [
        b"a0",
        b"b0",
        b"a2",
        b"b2",
    ]


def test_write_txt_records_clip_blocks(tmp_path):
    config = DenseConfig(
        record_path=Path("rec.tfrecord"),
        provider="dummy",
        clip_seconds=2,
        fps=15,
    )
    result = DenseResult(
        config=config,
        trajectory_length=60,
        metadata={"language_instruction1": "Cover the object"},
        clips=[
            ClipDescription(
                clip_index=0,
                start_step=0,
                end_step=29,
                language_instruction="Cover the object",
                description="The towel slides over the green object.",
                start_image_paths={"wrist_image": "/tmp/start.jpeg"},
                end_image_paths={"wrist_image": "/tmp/end.jpeg"},
            )
        ],
    )

    class DummyVLM(VLM):
        def generate(self, prompt: str, images: list[bytes]) -> str:
            return ""

    path = write_txt(result, DummyVLM("dummy"), tmp_path / "dense.txt")
    text = path.read_text()
    assert "clip_frames: 30" in text
    assert "language_instruction: Cover the object" in text
    assert "[clip 0] steps 0-29" in text
    assert "description: The towel slides over the green object." in text
    assert "(image) start/wrist_image: /tmp/start.jpeg" in text
    assert "(image) end/wrist_image: /tmp/end.jpeg" in text


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "\n".join(
            [
                "record: datasets/droid/success/success-00285.tfrecord",
                "provider: dummy",
                "clip_seconds: 2",
                "fps: 15",
                "max_clips: 2",
                "save_images: true",
            ]
        )
    )
    cfg = load_config(path)
    assert cfg.provider == "dummy"
    assert cfg.clip_frames == 30
    assert cfg.max_clips == 2
    assert cfg.save_images is True
    assert cfg.record_path.name == "success-00285.tfrecord"


class ScriptedDenseVLM(VLM):
    def __init__(self):
        super().__init__("dense-scripted")
        self.calls = 0
        self.image_counts: list[int] = []

    def generate(self, prompt: str, images: list[bytes]) -> str:
        self.calls += 1
        self.image_counts.append(len(images))
        return "The arm moves the towel slightly to the right."


def test_generate_dense_descriptions_over_clips(tmp_path, monkeypatch):
    from pipeline.dense_description import generate as generate_mod

    traj = Trajectory(
        record_path=str(tmp_path / "rec.tfrecord"),
        length=65,
        images={
            "shoulder_image_1": [f"s{i}".encode() for i in range(65)],
            "wrist_image": [f"w{i}".encode() for i in range(65)],
        },
        metadata={"language_instruction1": "Cover the green object"},
    )

    def fake_load(record_path, cameras, index=0):
        return traj

    monkeypatch.setattr(generate_mod, "load_trajectory", fake_load)
    vlm = ScriptedDenseVLM()
    config = DenseConfig(
        record_path=Path(traj.record_path),
        provider="dummy",
        cameras=("shoulder_image_1", "wrist_image"),
        clip_seconds=2,
        fps=15,
        max_clips=2,
        save_images=True,
        image_dir=tmp_path / "frames",
    )
    result = generate_mod.generate_dense_descriptions(config, vlm=vlm)

    assert vlm.calls == 2
    assert vlm.image_counts == [4, 4]  # 2 cams * start/end
    assert len(result.clips) == 2
    assert result.clips[0].start_step == 0
    assert result.clips[0].end_step == 29
    assert result.clips[1].start_step == 30
    assert result.clips[1].end_step == 59
    assert "towel" in result.clips[0].description
    assert result.clips[0].start_image_paths
    assert (tmp_path / "frames").is_dir()


def test_visualizer_parses_dense_run_and_renders_html(tmp_path):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))
    from summarize_dense_descriptions import parse_run, render_html

    # Minimal JPEG so the embed path succeeds.
    jpeg = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xd9"
    )
    start = tmp_path / "start.jpeg"
    end = tmp_path / "end.jpeg"
    start.write_bytes(jpeg)
    end.write_bytes(jpeg)

    run_path = tmp_path / "dense.txt"
    run_path.write_text(
        "\n".join(
            [
                "record: rec.tfrecord",
                "provider: dummy",
                "model: dummy",
                "clip_seconds: 2.0",
                "fps: 15.0",
                "clip_frames: 30",
                "trajectory_length: 60",
                "num_clips: 1",
                "language_instruction: Cover the object",
                "estimated_cost_total_usd: unknown",
                "estimated_cost_per_clip_usd: unknown",
                "=" * 60,
                "[clip 0] steps 0-29",
                "  language: Cover the object",
                "  description: shoulder_image_1:",
                "The towel slides ~5 cm over the green object.",
                "",
                "wrist_image:",
                "The gripper closes around the towel edge.",
                f"  (image) start/wrist_image: {start}",
                f"  (image) end/wrist_image: {end}",
                "",
            ]
        )
    )
    run = parse_run(run_path)
    assert run["info"]["language_instruction"] == "Cover the object"
    assert len(run["clips"]) == 1
    assert run["clips"][0]["start_step"] == 0
    assert run["clips"][0]["end_step"] == 29
    desc = run["clips"][0]["description"]
    assert "towel slides ~5 cm" in desc
    assert "wrist_image:" in desc
    assert "gripper closes" in desc
    assert "wrist_image" in run["clips"][0]["start_images"]

    html = render_html(run, run_path)
    assert "Dense descriptions" in html
    assert "towel slides ~5 cm" in html
    assert "shoulder_image_1" in html
    assert "cam-section" in html
    assert "Clip 0" in html
    assert "data:image/jpeg;base64," in html
