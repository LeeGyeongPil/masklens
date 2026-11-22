"""Unsloth-style masking from instruction/response marker strings (``train_on_responses_only``)."""
from __future__ import annotations


def find_all(ids: list[int], pattern: list[int]) -> list[int]:
    """Start positions of every occurrence of ``pattern`` in ``ids``."""
    if not pattern:
        return []
    n = len(pattern)
    return [i for i in range(len(ids) - n + 1) if ids[i:i + n] == pattern]


def parts_mask(ids: list[int], instruction_ids: list[int], response_ids: list[int]) -> list[int]:
    """Train from the end of each response marker to the start of the next instruction marker.

    Like Unsloth: everything is masked (-100) unless a response marker is found; text after the
    last response marker is trained up to the end of the sequence.
    """
    mask = [0] * len(ids)
    instruction_starts = find_all(ids, instruction_ids)
    for start in find_all(ids, response_ids):
        begin = start + len(response_ids)
        stop = next((s for s in instruction_starts if s >= begin), len(ids))
        for i in range(begin, stop):
            mask[i] = 1
    return mask
