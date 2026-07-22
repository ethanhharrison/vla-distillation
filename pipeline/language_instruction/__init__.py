"""Language-instruction generation pipeline for VLA distillation.

Given a DROID tfrecord, sample the trajectory at a configurable step interval
and prompt a swappable VLM to propose language instructions the robot could
accomplish starting at each sampled step.
"""

from .filter import ScoredInstruction, build_judge, score_instructions
from .generate import (
    GenerationConfig,
    GenerationResult,
    StepInstructions,
    generate_instructions,
    write_txt,
)
from .trajectory import Trajectory, load_trajectory, load_trajectories
from .vlm import VLM, available_providers, build_vlm, register_vlm

__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "ScoredInstruction",
    "StepInstructions",
    "build_judge",
    "generate_instructions",
    "score_instructions",
    "write_txt",
    "Trajectory",
    "load_trajectory",
    "load_trajectories",
    "VLM",
    "build_vlm",
    "register_vlm",
    "available_providers",
]
