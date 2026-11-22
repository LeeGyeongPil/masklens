"""The mask rules. Rule IDs and severities are stable strings (not translated)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .render import Analysis, Config, effective_template, has_generation_tag
from .truncate import lost_tokens

ERROR = "error"
WARN = "warn"

RULES: dict[str, str] = {
    "no_generation_tag": ERROR,
    "empty_mask": ERROR,
    "prompt_leak": ERROR,
    "render_mismatch": ERROR,
    "parts_not_found": ERROR,
    "response_truncated": WARN,
    "untrained_turn": WARN,
    "eos_not_trained": WARN,
    "content_not_found": WARN,
}


@dataclass
class Finding:
    """One problem. ``index`` is the 0-based example number, or ``None`` for dataset-level rules."""

    index: int | None
    rule: str
    severity: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


def _finding(index: int | None, rule: str, detail: str, **data: Any) -> Finding:
    return Finding(index, rule, RULES[rule], detail, data)


def dataset_findings(tokenizer: Any, config: Config) -> list[Finding]:
    """Rules that look at the setup rather than at one example."""
    if config.mode == "generation" and not has_generation_tag(effective_template(tokenizer, config.template)):
        return [_finding(None, "no_generation_tag", "the template used has no generation block")]
    return []


def example_findings(index: int, a: Analysis, config: Config) -> list[Finding]:
    """Apply every per-example rule to one analysis."""
    out: list[Finding] = []
    trained_full = sum(a.mask_full)
    trained_kept = sum(a.mask_kept)
    n_kept = a.kept[1] - a.kept[0]
    assistant_spans = [s for s in a.spans if s.role == "assistant"]

    missing = [s.message for s in a.spans if not s.found]
    if missing:
        out.append(_finding(index, "content_not_found",
                            f"message(s) {missing} not found verbatim in the rendered text",
                            messages=missing))

    if a.original_text is not None and a.original_text != a.text:
        pos = next((i for i, (x, y) in enumerate(zip(a.original_text, a.text)) if x != y),
                   min(len(a.original_text), len(a.text)))
        out.append(_finding(
            index, "render_mismatch", f"first difference at character {pos}", position=pos,
            original=a.original_text[max(0, pos - 20):pos + 20], template=a.text[max(0, pos - 20):pos + 20],
        ))

    if config.mode == "parts" and assistant_spans and not a.parts_found:
        out.append(_finding(index, "parts_not_found",
                            f"response string not found in {len(a.ids)} tokens", tokens=len(a.ids)))

    if trained_kept == 0:
        if assistant_spans:
            out.append(_finding(index, "empty_mask", f"0 of {n_kept} kept tokens are trained", tokens=n_kept))
        else:
            out.append(_finding(index, "empty_mask", "no assistant message in this example",
                                tokens=n_kept, no_assistant=True))

    leaks = []
    for span in a.spans:
        if span.role == "assistant":
            continue
        hits = [i for i in a.tokens_in(span) if a.mask_full[i]]
        if hits:
            leaks.append((span, hits))
    if leaks:
        total = sum(len(hits) for _, hits in leaks)
        first = leaks[0][0]
        out.append(_finding(index, "prompt_leak",
                            f"{total} prompt token(s) trained, first in message {first.message} ({first.role})",
                            count=total, message=first.message, role=first.role))

    if trained_full:
        lost = sum(lost_tokens(a.tokens_in(s), a.kept) for s in assistant_spans)
        if lost:
            out.append(_finding(index, "response_truncated", f"{lost} answer token(s) cut off", lost=lost))

        if len(assistant_spans) > 1:
            untrained = []
            for n, span in enumerate(assistant_spans, start=1):
                idx = a.tokens_in(span)
                if idx and not any(a.mask_full[i] for i in idx):
                    untrained.append(n)
            if untrained:
                out.append(_finding(index, "untrained_turn",
                                    f"assistant turn(s) {untrained} untrained", turns=untrained))

        no_end = []
        for n, span in enumerate(assistant_spans, start=1):
            idx = a.tokens_in(span)
            if not idx:
                continue
            j = idx[-1] + 1
            if (j < len(a.ids) and a.ids[j] in a.special_ids and a.kept[0] <= j < a.kept[1]
                    and not a.mask_full[j]):
                no_end.append(n)
        if no_end:
            out.append(_finding(index, "eos_not_trained",
                                f"end token not trained after assistant turn(s) {no_end}", turns=no_end))
    return out
