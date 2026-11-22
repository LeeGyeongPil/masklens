"""Shared fixtures: everything runs offline against the tiny tokenizer shipped in the repo."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import pytest  # noqa: E402

from masklens.loader import load_examples, load_template, load_tokenizer  # noqa: E402
from masklens.render import Analysis, Config, analyze  # noqa: E402
from masklens.rules import Finding, dataset_findings, example_findings  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = ROOT / "examples" / "tiny-chatml"
TRAINING = ROOT / "examples" / "tiny-chatml" / "training.jinja"
CHAT = ROOT / "examples" / "chat.jsonl"
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def tok():
    return load_tokenizer(str(TOKENIZER_DIR))


@pytest.fixture(scope="session")
def examples():
    return load_examples(CHAT)


@pytest.fixture(scope="session")
def training_template() -> str:
    return load_template(TRAINING)


def fixture_template(name: str) -> str:
    return load_template(FIXTURES / f"{name}.jinja")


def run_example(tok, messages, config: Config, index: int = 0) -> tuple[Analysis, list[Finding]]:
    analysis = analyze(tok, messages, config, index)
    return analysis, dataset_findings(tok, config) + example_findings(index, analysis, config)


def rules_of(findings: list[Finding]) -> set[str]:
    return {f.rule for f in findings}
