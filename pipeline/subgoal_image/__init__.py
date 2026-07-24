"""Stage B — instruction-conditioned subgoal-image generation.

Parallels `pipeline.language_instruction` (Stage A): where that stage turns a
frame into candidate *language instructions*, this stage turns a frame +
instruction into a *subgoal image* (the scene a few moments into executing the
instruction). It is independently runnable:

    python -m pipeline.subgoal_image.generate --examples <dir> [...]

Public API mirrors the language stage (`SubgoalConfig`, `generate_subgoals`).
Backends are swappable via a registry mirroring `language_instruction.vlm`.
"""

from __future__ import annotations

from .backends import (
    ImageEditBackend,
    SubgoalRequest,
    SubgoalResult,
    available_image_backends,
    build_image_backend,
    is_paid_backend,
    register_image_backend,
)
from .generate import SubgoalConfig, SubgoalRun, generate_subgoals
from .prompts import DEFAULT_TEMPLATE, TEMPLATES, build_prompt, resolve_template, template_id

__all__ = [
    "SubgoalConfig",
    "SubgoalRun",
    "generate_subgoals",
    "ImageEditBackend",
    "SubgoalRequest",
    "SubgoalResult",
    "register_image_backend",
    "build_image_backend",
    "available_image_backends",
    "is_paid_backend",
    "TEMPLATES",
    "DEFAULT_TEMPLATE",
    "build_prompt",
    "resolve_template",
    "template_id",
]
