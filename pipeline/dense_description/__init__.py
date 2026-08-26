"""Dense change-of-state description dataset generation for VLA distillation."""

from .prompts import DENSE_DESCRIPTION_PROMPT, build_dense_prompt, parse_description

__all__ = [
    "DENSE_DESCRIPTION_PROMPT",
    "build_dense_prompt",
    "parse_description",
]
