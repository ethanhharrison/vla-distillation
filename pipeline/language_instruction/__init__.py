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
from .pipeline import (
    CallResult,
    PipelineConfig,
    PipelineResult,
    load_pipeline_config,
    run_pipeline,
)
from .pricing import CostEstimate, RunCost, Usage, estimate_cost
from .trajectory import Trajectory, load_trajectories, load_trajectory
from .vlm import VLM, available_providers, build_vlm, register_vlm

__all__ = [
    "VLM",
    "CallResult",
    "CostEstimate",
    "GenerationConfig",
    "GenerationResult",
    "PipelineConfig",
    "PipelineResult",
    "RunCost",
    "ScoredInstruction",
    "StepInstructions",
    "Trajectory",
    "Usage",
    "available_providers",
    "build_judge",
    "build_run_cost",
    "build_vlm",
    "estimate_cost",
    "generate_instructions",
    "load_pipeline_config",
    "load_trajectories",
    "load_trajectory",
    "register_vlm",
    "run_pipeline",
    "score_instructions",
    "write_txt",
]
