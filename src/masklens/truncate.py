"""Own truncation, so results do not depend on how a given transformers version truncates."""
from __future__ import annotations

TRUNCATION_MODES = ("keep_start", "keep_end")


def kept_range(n_tokens: int, max_length: int | None, mode: str = "keep_start") -> tuple[int, int]:
    """Return the ``[start, end)`` token range that survives truncation to ``max_length``."""
    if mode not in TRUNCATION_MODES:
        raise ValueError(f"truncation must be one of {TRUNCATION_MODES}, got {mode!r}")
    if max_length is None or n_tokens <= max_length:
        return 0, n_tokens
    if max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if mode == "keep_start":
        return 0, max_length
    return n_tokens - max_length, n_tokens


def lost_tokens(indices: list[int], kept: tuple[int, int]) -> int:
    """How many of the token ``indices`` fall outside the kept range."""
    start, end = kept
    return sum(1 for i in indices if not start <= i < end)
