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


@pytest.fixture(autouse=True)
def _stable_environment(monkeypatch):
    """Make every test independent of the shell or CI runner it runs in.

    로컬 셸과 CI 러너의 차이(터미널 폭, 색 강제, 언어 변수, 실제 토크나이저 경로)가
    출력 문자열 비교를 흔들지 않도록 테스트마다 같은 환경으로 맞춘다.
    """
    monkeypatch.setenv("COLUMNS", "100")  # argparse 줄바꿈 폭 고정
    monkeypatch.setenv("NO_COLOR", "1")  # Python 3.14+ argparse 색과 masklens 색 끄기
    monkeypatch.setenv("PYTHON_COLORS", "0")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    for name in ("MASKLENS_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(name, raising=False)  # 언어는 각 테스트가 --lang·환경변수로 직접 정한다
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


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
