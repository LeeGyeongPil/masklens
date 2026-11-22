"""AC-8: English and Korean screens."""
from __future__ import annotations

import pytest
from conftest import CHAT, TOKENIZER_DIR

from masklens.cli import main
from masklens.i18n import MESSAGES, resolve_lang, t
from masklens.rules import RULES


def test_key_parity():
    assert set(MESSAGES["en"]) == set(MESSAGES["ko"])
    for lang in ("en", "ko"):
        for rule in RULES:
            for prefix in ("rule.", "hint.", "detail."):
                assert prefix + rule in MESSAGES[lang]
        assert all(text.strip() for text in MESSAGES[lang].values())


def test_lang_precedence():
    env = {"MASKLENS_LANG": "en", "LC_ALL": "ko_KR.UTF-8", "LANG": "ko_KR.UTF-8"}
    assert resolve_lang("ko", env) == "ko"  # --lang beats everything
    assert resolve_lang(None, env) == "en"  # MASKLENS_LANG beats locale variables
    assert resolve_lang(None, {"LC_ALL": "ko_KR.UTF-8", "LANG": "en_US.UTF-8"}) == "ko"
    assert resolve_lang(None, {"LC_MESSAGES": "en_US.UTF-8", "LANG": "ko_KR.UTF-8"}) == "en"
    assert resolve_lang(None, {"LANG": "ko_KR.UTF-8"}) == "ko"
    assert resolve_lang(None, {"LANG": "de_DE.UTF-8"}) == "en"
    assert resolve_lang(None, {}) == "en"
    assert resolve_lang("fr", {"MASKLENS_LANG": "ko"}) == "ko"  # unknown --lang is ignored


@pytest.mark.parametrize(("lang", "needle", "other"), [("en", "Preflight check", "사전 점검"),
                                                        ("ko", "사전 점검", "Preflight check")])
def test_help_in_both_languages(capsys, lang, needle, other):
    with pytest.raises(SystemExit) as exc:
        main(["--lang", lang, "--help"])
    assert exc.value.code == 0
    # argparse 는 터미널 폭(COLUMNS)에 따라 줄을 바꾸므로, 공백을 하나로 합친 뒤 비교한다
    out = " ".join(capsys.readouterr().out.split())
    assert needle in out and other not in out
    with pytest.raises(SystemExit):
        main(["check", "--lang", lang, "--help"])
    sub = " ".join(capsys.readouterr().out.split())
    assert "{% generation %}" in sub  # the % sign survives argparse help formatting


@pytest.mark.parametrize("columns", ["40", "60", "200"])
def test_help_survives_any_terminal_width(capsys, monkeypatch, columns):
    # CI 러너·작은 터미널처럼 폭이 달라도 도움말 문구 검사가 깨지지 않아야 한다
    monkeypatch.setenv("COLUMNS", columns)
    with pytest.raises(SystemExit):
        main(["check", "--lang", "en", "--help"])
    assert "{% generation %}" in " ".join(capsys.readouterr().out.split())


@pytest.mark.parametrize(("lang", "needle"), [("en", "RULE"), ("ko", "규칙")])
def test_summary_table_in_both_languages(capsys, lang, needle):
    code = main(["check", str(TOKENIZER_DIR), str(CHAT), "--lang", lang])
    out = capsys.readouterr().out
    assert code == 1 and needle in out
    assert "empty_mask" in out  # rule IDs are never translated


def test_env_var_switches_language(capsys, monkeypatch):
    monkeypatch.setenv("MASKLENS_LANG", "ko")
    main(["check", str(TOKENIZER_DIR), str(CHAT)])
    assert "결과: 실패" in capsys.readouterr().out
    main(["check", str(TOKENIZER_DIR), str(CHAT), "--lang", "en"])
    assert "Result: FAIL" in capsys.readouterr().out


def test_t_falls_back_to_english():
    assert t("xx", "table.clean") == MESSAGES["en"]["table.clean"]
