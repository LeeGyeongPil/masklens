"""Input loading: messages JSONL, tokenizer folder, template file.

Problems raise :class:`InputError` carrying a message key and parameters; the CLI turns
that into a localized message and exit code 2.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class InputError(Exception):
    """Bad user input. ``key`` is a ``err.*`` entry of the message table."""

    def __init__(self, key: str, **params: Any) -> None:
        super().__init__(key)
        self.key = key
        self.params = params


@dataclass(frozen=True)
class Example:
    """One dataset example: its 0-based position, its JSONL line number and its messages."""

    index: int
    line: int
    messages: list[dict[str, str]] = field(default_factory=list)


def _check_messages(path: str, line: int, field_name: str, value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise InputError("err.shape_not_list", path=path, line=line, field=field_name)
    for n, message in enumerate(value, start=1):
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
        ):
            raise InputError("err.shape_bad_message", path=path, line=line, n=n)
    return [{"role": m["role"], "content": m["content"]} for m in value]


def load_examples(path: str | Path, field_name: str = "messages", limit: int | None = None) -> list[Example]:
    """Read a messages JSONL file. Blank lines are skipped; errors carry the line number."""
    p = Path(path)
    if not p.is_file():
        raise InputError("err.file_not_found", path=str(p))
    examples: list[Example] = []
    line_no = 0
    try:
        with p.open("rb") as handle:  # decode per line so an encoding error has an exact line number
            for raw_bytes in handle:
                line_no += 1
                raw = raw_bytes.decode("utf-8-sig" if line_no == 1 else "utf-8")  # first line may hold a BOM
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise InputError("err.json_line", path=str(p), line=line_no, reason=exc.msg) from exc
                if not isinstance(obj, dict):
                    raise InputError("err.shape_not_object", path=str(p), line=line_no)
                if field_name not in obj:
                    raise InputError("err.shape_missing_field", path=str(p), line=line_no, field=field_name)
                messages = _check_messages(str(p), line_no, field_name, obj[field_name])
                examples.append(Example(index=len(examples), line=line_no, messages=messages))
                if limit is not None and len(examples) >= limit:
                    break
    except UnicodeDecodeError as exc:
        raise InputError("err.encoding", path=str(p), line=line_no) from exc
    if not examples:
        raise InputError("err.empty_data", path=str(p))
    return examples


def load_template(path: str | Path) -> str:
    """Read a Jinja chat template file (one trailing newline is dropped, like Jinja does)."""
    p = Path(path)
    if not p.is_file():
        raise InputError("err.file_not_found", path=str(p))
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError("err.template_read", path=str(p), reason=str(exc)) from exc
    return text[:-1] if text.endswith("\n") else text


def load_tokenizer(name_or_path: str) -> Any:
    """Load a tokenizer with ``AutoTokenizer`` (local folder or Hub ID)."""
    logging.getLogger("transformers").setLevel(logging.ERROR)
    from transformers import AutoTokenizer  # imported lazily so --help stays fast

    try:
        return AutoTokenizer.from_pretrained(name_or_path)
    except Exception as exc:  # noqa: BLE001 - any loader failure is a user-facing input problem
        raise InputError("err.tokenizer", path=name_or_path, reason=str(exc).splitlines()[0][:200]) from exc
