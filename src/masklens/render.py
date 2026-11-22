"""Render a conversation, tokenize it, and compute the actual and the expected loss mask."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .loader import InputError
from .parts import parts_mask
from .truncate import kept_range

GENERATION_TAG = re.compile(r"\{%-?\s*generation\s*-?%\}")
MODES = ("generation", "parts")


@dataclass(frozen=True)
class Config:
    """What to check: mask method, template override, truncation and parts strings."""

    mode: str = "generation"
    template: str | None = None  # template text overriding the tokenizer's own
    max_length: int | None = None
    truncation: str = "keep_start"
    instruction_part: str | None = None
    response_part: str | None = None


@dataclass(frozen=True)
class Span:
    """Character span of one message's ``content`` inside the rendered text."""

    message: int
    role: str
    start: int
    end: int
    found: bool

    @property
    def empty(self) -> bool:
        return self.end <= self.start


@dataclass
class Analysis:
    """Everything the rules need to know about one rendered example."""

    text: str
    ids: list[int]
    offsets: list[tuple[int, int]]
    mask_full: list[int]  # mask over all tokens, before truncation
    kept: tuple[int, int]  # token range surviving truncation
    mask_kept: list[int]  # mask over the kept tokens
    spans: list[Span]
    special_ids: frozenset[int]
    original_text: str | None = None  # render with the tokenizer's own template (if overridden)
    parts_found: bool = True
    messages: list[dict[str, str]] = field(default_factory=list)

    def tokens_in(self, span: Span) -> list[int]:
        """Indices (in the full sequence) of tokens overlapping the span, by character overlap."""
        if not span.found or span.empty:
            return []
        return [i for i, (s, e) in enumerate(self.offsets) if e > s and s < span.end and e > span.start]


def effective_template(tokenizer: Any, override: str | None) -> str:
    """The template text that will be used, or an :class:`InputError` if there is none."""
    if override is not None:
        return override
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        template = template.get("default") or next(iter(template.values()), None)
    if not template:
        raise InputError("err.no_template")
    return template


def has_generation_tag(template: str) -> bool:
    return GENERATION_TAG.search(template) is not None


def render_text(tokenizer: Any, messages: list[dict[str, str]], template: str | None, index: int = 0) -> str:
    """Render without tokenizing, wrapping template errors."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, chat_template=template)
    except Exception as exc:  # noqa: BLE001 - jinja errors vary; all are input problems here
        raise InputError("err.render", index=index, reason=str(exc).splitlines()[0][:200]) from exc


def find_spans(text: str, messages: list[dict[str, str]]) -> list[Span]:
    """Locate each message's content in the rendered text, in order."""
    spans: list[Span] = []
    cursor = 0
    for i, message in enumerate(messages):
        content = message["content"]
        if not content:
            spans.append(Span(i, message["role"], cursor, cursor, True))
            continue
        pos = text.find(content, cursor)
        if pos < 0:
            spans.append(Span(i, message["role"], -1, -1, False))
            continue
        spans.append(Span(i, message["role"], pos, pos + len(content), True))
        cursor = pos + len(content)
    return spans


def _part_ids(tokenizer: Any, part: str, which: str) -> list[int]:
    ids = list(tokenizer(part, add_special_tokens=False)["input_ids"])
    if not ids:
        raise InputError("err.parts_empty", which=which)
    return ids


def analyze(tokenizer: Any, messages: list[dict[str, str]], config: Config, index: int = 0) -> Analysis:
    """Render one example and compute its mask the way a trainer would."""
    logging.getLogger("transformers").setLevel(logging.ERROR)
    effective_template(tokenizer, config.template)  # fail early when there is no template
    text = render_text(tokenizer, messages, config.template, index)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = [(int(s), int(e)) for s, e in encoded["offset_mapping"]]

    parts_found = True
    if config.mode == "generation":
        try:
            enc = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=True,
                return_assistant_tokens_mask=True, chat_template=config.template,
            )
        except Exception as exc:  # noqa: BLE001
            raise InputError("err.render", index=index, reason=str(exc).splitlines()[0][:200]) from exc
        if list(enc["input_ids"]) != ids:
            raise InputError("err.tokenization", index=index)
        mask_full = [int(m) for m in enc["assistant_masks"]]
        kept = kept_range(len(ids), config.max_length, config.truncation)
        mask_kept = mask_full[kept[0]:kept[1]]
    else:
        instruction_ids = _part_ids(tokenizer, config.instruction_part or "", "--instruction-part")
        response_ids = _part_ids(tokenizer, config.response_part or "", "--response-part")
        mask_full = parts_mask(ids, instruction_ids, response_ids)
        parts_found = _contains(ids, response_ids)
        kept = kept_range(len(ids), config.max_length, config.truncation)
        # Unsloth masks the already truncated sequence, so mask the kept ids on their own.
        mask_kept = parts_mask(ids[kept[0]:kept[1]], instruction_ids, response_ids)

    original_text = None
    if config.template is not None and tokenizer.chat_template:
        original_text = render_text(tokenizer, messages, None, index)
    return Analysis(
        text=text, ids=ids, offsets=offsets, mask_full=mask_full, kept=kept, mask_kept=mask_kept,
        spans=find_spans(text, messages),
        special_ids=frozenset(int(i) for i in tokenizer.all_special_ids),
        original_text=original_text, parts_found=parts_found, messages=messages,
    )


def _contains(ids: list[int], pattern: list[int]) -> bool:
    n = len(pattern)
    return any(ids[i:i + n] == pattern for i in range(len(ids) - n + 1))


def expected_mask(analysis: Analysis, include_end: bool = True) -> list[int]:
    """Independent expectation: assistant ``content`` characters, plus the end-of-turn token.

    Computed from character positions of each assistant message's content, not from any
    ``{% generation %}`` block. With ``include_end`` the special token right after the content
    and any whitespace-only tokens after it (usually a newline) are expected as well.
    """
    n = len(analysis.ids)
    mask = [0] * n
    for span in analysis.spans:
        if span.role != "assistant":
            continue
        indices = analysis.tokens_in(span)
        for i in indices:
            mask[i] = 1
        if include_end and indices:
            j = indices[-1] + 1
            if j < n and analysis.ids[j] in analysis.special_ids:
                mask[j] = 1
                j += 1
                while j < n and _is_blank(analysis, j):
                    mask[j] = 1
                    j += 1
    return mask


def _is_blank(analysis: Analysis, i: int) -> bool:
    s, e = analysis.offsets[i]
    return e > s and analysis.text[s:e].strip() == ""
