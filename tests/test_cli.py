"""AC-1, AC-2, AC-7: command behavior, JSON report, exit codes."""
from __future__ import annotations

import json

from conftest import CHAT, FIXTURES, TOKENIZER_DIR, TRAINING

from masklens.cli import main

BASE = [str(TOKENIZER_DIR), str(CHAT)]


def test_show_marks_trained_tokens(capsys):
    code = main(["show", *BASE, "--index", "1", "--template", str(TRAINING), "--no-color", "--lang", "en"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[[The answer is 4.<|im_end|>" in out
    assert "trained 22/82 tokens" in out


def test_show_original_template_has_no_highlight(capsys):
    code = main(["show", *BASE, "--no-color", "--lang", "en"])
    out = capsys.readouterr().out
    assert code == 0
    body = out.split("trained 0/")[0]
    assert "[[" not in body and "trained 0/" in out
    assert "no_generation_tag" in out


def test_check_original_template_flags_no_generation_tag(capsys):
    code = main(["check", *BASE, "--lang", "en", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["summary"]["by_rule"] == {"no_generation_tag": 1, "empty_mask": 5}
    assert [f["index"] for f in report["findings"] if f["rule"] == "empty_mask"] == [0, 1, 2, 3, 4]
    dataset = [f for f in report["findings"] if f["rule"] == "no_generation_tag"]
    assert dataset[0]["index"] is None


def test_json_report_fields(capsys):
    main(["check", *BASE, "--template", str(TRAINING), "--max-length", "48", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["versions"]["transformers"].startswith("5.")
    assert report["mode"] == "generation"
    assert report["max_length"] == 48
    assert report["truncation"] == "keep_start"
    assert report["template"] == str(TRAINING)
    assert report["summary"]["by_rule"] == {"empty_mask": 2, "response_truncated": 3}
    for finding in report["findings"]:
        assert {"index", "rule", "severity"} <= finding.keys()


def test_exit_codes(capsys):
    # clean -> 0
    assert main(["check", *BASE, "--template", str(TRAINING)]) == 0
    # error -> 1
    assert main(["check", *BASE]) == 1
    # warnings only: 0, but 1 with --strict
    no_eos = str(FIXTURES / "no_eos.jinja")
    assert main(["check", *BASE, "--template", no_eos]) == 0
    assert main(["check", *BASE, "--template", no_eos, "--strict"]) == 1
    capsys.readouterr()
    # broken JSONL -> 2, message names the line
    code = main(["check", str(TOKENIZER_DIR), str(FIXTURES / "bad.jsonl"), "--lang", "en"])
    err = capsys.readouterr().err
    assert code == 2
    assert "bad.jsonl:3" in err


def test_input_errors_exit_2(capsys, tmp_path):
    assert main(["check", *BASE[:1], str(tmp_path / "missing.jsonl"), "--lang", "en"]) == 2
    assert "File not found" in capsys.readouterr().err
    assert main(["check", str(tmp_path / "no-tokenizer"), str(CHAT), "--lang", "en"]) == 2
    assert "Could not load the tokenizer" in capsys.readouterr().err
    assert main(["show", *BASE, "--index", "99", "--lang", "en"]) == 2
    assert "out of range" in capsys.readouterr().err
    assert main(["check", *BASE, "--mode", "parts", "--lang", "en"]) == 2
    assert "--instruction-part" in capsys.readouterr().err
    assert main(["check", *BASE, "--max-length", "0", "--lang", "en"]) == 2
    assert main(["check", *BASE, "--template", str(tmp_path / "x.jinja")]) == 2
    broken = tmp_path / "broken.jinja"
    broken.write_text("{% for m in messages %}{{ m.content ", encoding="utf-8")
    assert main(["check", *BASE, "--template", str(broken), "--lang", "en"]) == 2
    assert "Rendering example 0 failed" in capsys.readouterr().err


def test_bad_message_shape_names_line(capsys, tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"messages": [{"role": "user", "content": 5}]}\n', encoding="utf-8")
    assert main(["check", str(TOKENIZER_DIR), str(path), "--lang", "en"]) == 2
    assert "d.jsonl:1" in capsys.readouterr().err
    path.write_text('{"chat": []}\n', encoding="utf-8")
    assert main(["check", str(TOKENIZER_DIR), str(path), "--lang", "en"]) == 2
    assert "missing field" in capsys.readouterr().err
    assert main(["check", str(TOKENIZER_DIR), str(path), "--field", "chat", "--lang", "en"]) == 2
    assert "non-empty list" in capsys.readouterr().err


def test_limit_checks_only_first_examples(capsys):
    main(["check", *BASE, "--limit", "2", "--json"])
    assert json.loads(capsys.readouterr().out)["examples"] == 2


def test_non_utf8_file_exits_2_with_line(capsys, tmp_path):
    good = '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}\n'
    bad = tmp_path / "latin.jsonl"
    bad.write_bytes(good.encode() + b'{"messages": "\xff\xfe"}\n')
    assert main(["check", str(TOKENIZER_DIR), str(bad), "--lang", "en"]) == 2
    err = capsys.readouterr().err
    assert "latin.jsonl:2" in err and "UTF-8" in err and "Traceback" not in err


def test_bom_file_is_accepted(tmp_path):
    src = CHAT.read_text(encoding="utf-8")
    bom = tmp_path / "bom.jsonl"
    bom.write_bytes(b"\xef\xbb\xbf" + src.encode("utf-8"))
    assert main(["check", str(TOKENIZER_DIR), str(bom), "--template", str(TRAINING), "--lang", "en"]) == 0


def test_example_without_assistant_explained(capsys, tmp_path):
    f = tmp_path / "noans.jsonl"
    f.write_text('{"messages":[{"role":"user","content":"hi"}]}\n', encoding="utf-8")
    main(["check", str(TOKENIZER_DIR), str(f), "--template", str(TRAINING), "--lang", "en"])
    assert "no assistant message" in capsys.readouterr().out
