"""Reports: summary table, JSON report and the highlighted ``show`` output."""
from __future__ import annotations

import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from . import __version__
from .i18n import t
from .loader import Example
from .render import Analysis, Config, analyze
from .rules import ERROR, RULES, WARN, Finding, dataset_findings, example_findings

SHOW_EXAMPLES = 10  # example numbers listed per rule in the table


@dataclass
class Report:
    """Result of checking a dataset."""

    tokenizer: str
    template_label: str
    config: Config
    examples_checked: int
    findings: list[Finding] = field(default_factory=list)
    line_of: dict[int, int] = field(default_factory=dict)

    @property
    def errors(self) -> int:
        return sum(1 for f in self.findings if f.severity == ERROR)

    @property
    def warnings(self) -> int:
        return sum(1 for f in self.findings if f.severity == WARN)

    def by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for f in self.findings:
            counts[f.rule] += 1
        return {rule: counts[rule] for rule in RULES if rule in counts}

    def exit_code(self, strict: bool = False) -> int:
        if self.errors or (strict and self.warnings):
            return 1
        return 0


def check(tokenizer: Any, examples: list[Example], config: Config, tokenizer_name: str,
          template_label: str) -> Report:
    """Run every rule over every example."""
    report = Report(tokenizer_name, template_label, config, len(examples))
    report.findings.extend(dataset_findings(tokenizer, config))
    for ex in examples:
        analysis = analyze(tokenizer, ex.messages, config, ex.index)
        report.findings.extend(example_findings(ex.index, analysis, config))
        report.line_of[ex.index] = ex.line
    return report


def _versions() -> dict[str, str]:
    import tokenizers
    import transformers

    return {"masklens": __version__, "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__}


def to_json(report: Report, strict: bool = False) -> dict[str, Any]:
    """Machine-readable report. Keys are never translated."""
    cfg = report.config
    return {
        "versions": _versions(),
        "tokenizer": report.tokenizer,
        "template": report.template_label,
        "mode": cfg.mode,
        "max_length": cfg.max_length,
        "truncation": cfg.truncation,
        "examples": report.examples_checked,
        "summary": {
            "errors": report.errors,
            "warnings": report.warnings,
            "by_rule": report.by_rule(),
            "exit_code": report.exit_code(strict),
        },
        "findings": [
            {"index": f.index, "line": report.line_of.get(f.index) if f.index is not None else None,
             "rule": f.rule, "severity": f.severity, "detail": f.detail, **({"data": f.data} if f.data else {})}
            for f in report.findings
        ],
    }


def _width(text: str) -> int:
    """Display width: Korean characters take two terminal columns."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * (width - _width(text))


def _detail(lang: str, f: Finding) -> str:
    params = dict(f.data)
    if f.rule == "prompt_leak":
        params.setdefault("role", "?")
    if "turns" in params:
        params["turns"] = ", ".join(str(x) for x in params["turns"])
    if "messages" in params:
        params["messages"] = ", ".join(str(x) for x in params["messages"])
    if params.pop("no_assistant", False):
        return t(lang, f"detail.{f.rule}_no_assistant", **params)
    return t(lang, f"detail.{f.rule}", **params)


def render_table(report: Report, lang: str, strict: bool = False) -> str:
    """Human-readable summary: header, rule table, fixes and the verdict."""
    cfg = report.config
    template = t(lang, "table.template_original") if report.template_label == "original" else report.template_label
    lines = [
        t(lang, "table.title"),
        f"  {t(lang, 'table.tokenizer')}: {report.tokenizer}",
        f"  {t(lang, 'table.template')}: {template}",
        f"  {t(lang, 'table.mode')}: {cfg.mode}",
        f"  {t(lang, 'table.max_length')}: "
        f"{cfg.max_length if cfg.max_length else t(lang, 'table.none')} ({cfg.truncation})",
        f"  {t(lang, 'table.examples')}: {report.examples_checked}",
        "",
    ]
    counts = report.by_rule()
    if not counts:
        lines.append(t(lang, "table.clean"))
    else:
        rows = []
        for rule, count in counts.items():
            idx = sorted({f.index for f in report.findings if f.rule == rule and f.index is not None})
            shown = ",".join(str(i) for i in idx[:SHOW_EXAMPLES])
            if len(idx) > SHOW_EXAMPLES:
                shown += " " + t(lang, "table.more", n=len(idx) - SHOW_EXAMPLES)
            rows.append((rule, RULES[rule], str(count), shown or t(lang, "table.dataset")))
        header = (t(lang, "table.col_rule"), t(lang, "table.col_severity"),
                  t(lang, "table.col_count"), t(lang, "table.col_examples"))
        widths = [max(_width(header[c]), *(_width(r[c]) for r in rows)) for c in range(3)]
        lines.append("  ".join(_pad(header[c], widths[c]) for c in range(3)) + "  " + header[3])
        for r in rows:
            lines.append("  ".join(_pad(r[c], widths[c]) for c in range(3)) + "  " + r[3])
        lines.append("")
        lines.append(t(lang, "table.fix") + ":")
        for rule in counts:
            first = next(f for f in report.findings if f.rule == rule)
            lines.append(f"  {rule}: {t(lang, 'rule.' + rule)}")
            lines.append(f"    -> {t(lang, 'hint.' + rule)}")
            if first.index is None:
                lines.append("    " + _detail(lang, first))
            else:
                lines.append("    " + t(lang, "table.detail", index=first.index, detail=_detail(lang, first)))
    lines.append("")
    if report.errors:
        lines.append(t(lang, "table.result_fail", errors=report.errors, warnings=report.warnings))
    elif report.warnings and strict:
        lines.append(t(lang, "table.result_strict", warnings=report.warnings))
    elif report.warnings:
        lines.append(t(lang, "table.result_warn", warnings=report.warnings))
    else:
        lines.append(t(lang, "table.result_pass"))
    return "\n".join(lines)


def _escape(text: str) -> str:
    """Make newlines visible while keeping the line structure."""
    return text.replace("\n", "\\n\n")


def render_show(analysis: Analysis, index: int, config: Config, findings: list[Finding], lang: str,
                color: bool) -> str:
    """One example with the trained tokens highlighted (ANSI color, or ``[[...]]`` without color)."""
    start, end = analysis.kept
    # Merge consecutive tokens with the same mask value into one run of characters.
    runs: list[tuple[int, int, int]] = []  # (char_start, char_end, trained)
    cursor = 0
    for offset, flag in zip(analysis.offsets[start:end], analysis.mask_kept):
        s, e = max(offset[0], cursor), offset[1]
        if e <= s:
            continue
        if runs and runs[-1][2] == flag and s == runs[-1][1]:
            runs[-1] = (runs[-1][0], e, flag)
        else:
            runs.append((s, e, flag))
        cursor = e
    body = []
    for s, e, flag in runs:
        piece = _escape(analysis.text[s:e])
        if not flag:
            body.append(piece)
        elif color:
            body.append(f"\x1b[1;42;30m{piece}\x1b[0m")
        else:
            body.append(f"[[{piece}]]")
    trained = sum(analysis.mask_kept)
    n_kept = end - start
    lines = [t(lang, "show.title", index=index), ""]
    lines.append("".join(body))
    lines.append("")
    lines.append(t(lang, "show.trained", n=trained, total=n_kept))
    if (start, end) != (0, len(analysis.ids)):
        lines.append(t(lang, "show.truncated", kept=n_kept, full=len(analysis.ids),
                       truncation=config.truncation))
    lines.append(t(lang, "show.legend_color" if color else "show.legend_plain"))
    if not trained:
        lines.append(t(lang, "show.nothing"))
    if findings:
        lines.append("")
        lines.append(t(lang, "show.findings") + ":")
        for f in findings:
            lines.append(f"  {f.rule} ({f.severity}): {_detail(lang, f)}")
    return "\n".join(lines)
