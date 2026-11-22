"""Optional: run against real tokenizers you downloaded (skipped unless MASKLENS_REAL_TOK_DIR is set).

MASKLENS_REAL_TOK_DIR points to a folder whose sub-folders are tokenizer folders
(for example ``qwen25/`` and ``smollm3/``).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from conftest import CHAT

from masklens.loader import load_examples, load_tokenizer
from masklens.render import Config, has_generation_tag
from masklens.report import check

ROOT = os.environ.get("MASKLENS_REAL_TOK_DIR")
FOLDERS = sorted(p for p in Path(ROOT).iterdir() if (p / "tokenizer_config.json").exists()) if ROOT else []


@pytest.mark.skipif(not FOLDERS, reason="set MASKLENS_REAL_TOK_DIR to run against real tokenizers")
@pytest.mark.parametrize("folder", FOLDERS, ids=lambda p: p.name)
def test_real_tokenizer_runs(folder):
    tok = load_tokenizer(str(folder))
    report = check(tok, load_examples(CHAT), Config(), str(folder), "original")
    template = tok.chat_template if isinstance(tok.chat_template, str) else tok.chat_template["default"]
    assert ("no_generation_tag" in report.by_rule()) == (not has_generation_tag(template))
