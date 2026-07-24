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
    build_run_cost,
    generate_instructions,
    write_txt,
)
from .pricing import CostEstimate, RunCost, Usage, estimate_cost
from .trajectory import Trajectory, load_trajectory, load_trajectories
from .vlm import VLM, available_providers, build_vlm, register_vlm

__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "ScoredInstruction",
    "StepInstructions",
    "build_judge",
    "build_run_cost",
    "generate_instructions",
    "score_instructions",
    "write_txt",
    "CostEstimate",
    "RunCost",
    "Usage",
    "estimate_cost",
    "Trajectory",
    "load_trajectory",
    "load_trajectories",
    "VLM",
    "build_vlm",
    "register_vlm",
    "available_providers",
]
