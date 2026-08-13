"""Post-merge LLM augmentation: paraphrase, noisy, and combined variants.

Runs after merge (and after uniqueness clustering, when enabled) to grow the
kept instruction list with:

- per-seed paraphrases and deliberately imperfect rewrites
- a total budget of multi-task instructions that combine 2+ pool entries
"""

from __future__ import annotations

import re

from .prompts import (
    AUGMENT_PROMPT,
    COMBINE_PROMPT,
    build_augment_prompt,
    build_combine_prompt,
    parse_instructions,
)
from .vlm import VLM, build_vlm

AUGMENT_KEYS = frozenset(
    {
        "enabled",
        "provider",
        "model",
        "paraphrases_per_instruction",
        "noisy_per_instruction",
        "combined_instructions",
    }
)

DEFAULT_PARAPHRASES_PER_INSTRUCTION = 1
DEFAULT_NOISY_PER_INSTRUCTION = 1
DEFAULT_COMBINED_INSTRUCTIONS = 0

SEED_HEADER = re.compile(r"^---\s*seed\s+(\d+)\s*---\s*$", re.IGNORECASE)


def resolve_augment_counts(settings: dict) -> tuple[int, int, int]:
    """Return (paraphrases_per_instruction, noisy_per_instruction, combined)."""
    paraphrases = int(
        settings.get(
            "paraphrases_per_instruction", DEFAULT_PARAPHRASES_PER_INSTRUCTION
        )
    )
    noisy = int(
        settings.get("noisy_per_instruction", DEFAULT_NOISY_PER_INSTRUCTION)
    )
    combined = int(
        settings.get("combined_instructions", DEFAULT_COMBINED_INSTRUCTIONS)
    )
    if paraphrases < 0:
        raise ValueError(
            f"augment.paraphrases_per_instruction must be >= 0, got {paraphrases}"
        )
    if noisy < 0:
        raise ValueError(
            f"augment.noisy_per_instruction must be >= 0, got {noisy}"
        )
    if combined < 0:
        raise ValueError(
            f"augment.combined_instructions must be >= 0, got {combined}"
        )
    if paraphrases + noisy + combined < 1:
        raise ValueError(
            "augment requires paraphrases_per_instruction + "
            "noisy_per_instruction + combined_instructions >= 1"
        )
    return paraphrases, noisy, combined


def build_augmenter(provider: str, model: str | None = None) -> VLM:
    """Build the text-only VLM used for post-merge augmentation."""
    return build_vlm(provider, model=model)


def _dedupe_variants(variants: list[str], *, blocked: set[str]) -> list[str]:
    """Drop empties, seed echoes, and within-list duplicates (order-preserving)."""
    seen = set(blocked)
    unique: list[str] = []
    for text in variants:
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def parse_augment_groups(
    text: str,
    seeds: list[str],
    *,
    expected_per_seed: int,
) -> dict[str, list[str]]:
    """Parse an augment response into seed -> variants.

    Prefers `--- seed N ---` headers. If none appear, falls back to chunking
    the flat instruction list into groups of `expected_per_seed`.
    """
    groups: dict[str, list[str]] = {seed: [] for seed in seeds}
    if not seeds:
        return groups

    lines = text.splitlines()
    header_indexes = [
        (i, int(match.group(1)) - 1)
        for i, line in enumerate(lines)
        if (match := SEED_HEADER.match(line.strip()))
    ]

    if header_indexes:
        for pos, (start, seed_idx) in enumerate(header_indexes):
            if not (0 <= seed_idx < len(seeds)):
                continue
            end = (
                header_indexes[pos + 1][0]
                if pos + 1 < len(header_indexes)
                else len(lines)
            )
            chunk = "\n".join(lines[start + 1 : end])
            groups[seeds[seed_idx]].extend(parse_instructions(chunk))
    else:
        # Flat fallback: assign consecutive blocks of the expected size.
        flat = parse_instructions(text)
        for seed_idx, seed in enumerate(seeds):
            start = seed_idx * expected_per_seed
            groups[seed].extend(flat[start : start + expected_per_seed])

    blocked = set(seeds)
    for seed in seeds:
        groups[seed] = _dedupe_variants(groups[seed], blocked=blocked)
        blocked.update(groups[seed])
    return groups


def expand_instructions(
    vlm: VLM,
    instructions: list[str],
    *,
    paraphrases_per_instruction: int = DEFAULT_PARAPHRASES_PER_INSTRUCTION,
    noisy_per_instruction: int = DEFAULT_NOISY_PER_INSTRUCTION,
    template: str = AUGMENT_PROMPT,
) -> dict[str, list[str]]:
    """Ask the LLM for new variants of each instruction.

    Returns seed -> new variants (originals are never included). Empty input
    or a zero paraphrase+noisy budget yields empty groups (no API call).
    """
    groups = {seed: [] for seed in instructions}
    if not instructions:
        return groups
    if paraphrases_per_instruction + noisy_per_instruction < 1:
        return groups

    prompt = build_augment_prompt(
        instructions,
        paraphrases_per_instruction=paraphrases_per_instruction,
        noisy_per_instruction=noisy_per_instruction,
        template=template,
    )
    # Text-only: paraphrasing does not need scene frames.
    raw = vlm.generate(prompt, images=[])
    return parse_augment_groups(
        raw,
        instructions,
        expected_per_seed=paraphrases_per_instruction + noisy_per_instruction,
    )


def combine_instructions(
    vlm: VLM,
    pool: list[str],
    *,
    num_combined: int,
    template: str = COMBINE_PROMPT,
) -> list[str]:
    """Ask the LLM for multi-task instructions that fuse 2+ pool entries.

    Returns only new combined lines (pool members are never echoed back).
    Empty pool or num_combined < 1 yields [] (no API call).
    """
    if not pool or num_combined < 1:
        return []
    if len(pool) < 2:
        raise ValueError(
            "combine_instructions needs at least 2 pool entries to fuse, "
            f"got {len(pool)}"
        )

    prompt = build_combine_prompt(pool, num_combined=num_combined, template=template)
    raw = vlm.generate(prompt, images=[])
    return _dedupe_variants(parse_instructions(raw), blocked=set(pool))[:num_combined]


def flatten_augment_groups(groups: dict[str, list[str]]) -> list[str]:
    """Flatten seed -> variants in seed order (for counts / final lists)."""
    flat: list[str] = []
    for variants in groups.values():
        flat.extend(variants)
    return flat
