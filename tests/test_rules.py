"""AC-4 and AC-6: template mistakes are detected; the right template is clean."""
from __future__ import annotations

from conftest import fixture_template, rules_of, run_example

from masklens.render import Config, expected_mask

UNTRAINED_FIRST_TURN = (
    "{%- for message in messages %}{%- if message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if loop.index > 3 %}{%- generation %}{{- message.content + '<|im_end|>' + '\\n' }}{%- endgeneration %}"
    "{%- else %}{{- message.content + '<|im_end|>' + '\\n' }}{%- endif %}"
    "{%- else %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endif %}{%- endfor %}"
)
UPPERCASING = (
    "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + (message.content | upper)"
    " + '<|im_end|>' + '\\n' }}{%- endfor %}"
)


def test_leaky_template_reports_prompt_leak(tok, examples):
    cfg = Config(template=fixture_template("leaky"))
    for ex in examples:
        _, findings = run_example(tok, ex.messages, cfg, ex.index)
        leak = [f for f in findings if f.rule == "prompt_leak"]
        assert len(leak) == 1 and leak[0].severity == "error"
        assert leak[0].data["count"] > 0


def test_no_eos_template_warns(tok, examples):
    cfg = Config(template=fixture_template("no_eos"))
    for ex in examples:
        _, findings = run_example(tok, ex.messages, cfg, ex.index)
        assert rules_of(findings) == {"eos_not_trained"}
        assert findings[0].severity == "warn"
    # In a multi-turn example the warning lists both turns.
    _, findings = run_example(tok, examples[1].messages, cfg, 1)
    assert findings[0].data["turns"] == [1, 2]


def test_training_template_clean(tok, examples, training_template):
    cfg = Config(template=training_template)
    for ex in examples:
        analysis, findings = run_example(tok, ex.messages, cfg, ex.index)
        assert findings == []
        # The mask transformers computed from {% generation %} equals the one derived
        # independently from the assistant content character positions.
        assert analysis.mask_full == expected_mask(analysis)
        assert sum(analysis.mask_full) > 0


def test_expected_mask_without_end_token_differs(tok, examples, training_template):
    analysis, _ = run_example(tok, examples[0].messages, Config(template=training_template))
    assert sum(expected_mask(analysis, include_end=False)) < sum(expected_mask(analysis))


def test_render_mismatch_reports_offset(tok, examples, training_template):
    drift = Config(template=fixture_template("drift"))
    for ex in examples:
        analysis, findings = run_example(tok, ex.messages, drift, ex.index)
        mismatch = [f for f in findings if f.rule == "render_mismatch"]
        assert len(mismatch) == 1
        position = mismatch[0].data["position"]
        assert analysis.original_text is not None
        assert analysis.original_text[:position] == analysis.text[:position]
        assert analysis.original_text[position] != analysis.text[position]
        assert analysis.text[position] == " "  # the stray space the template adds
    good = Config(template=training_template)
    for ex in examples:
        assert "render_mismatch" not in rules_of(run_example(tok, ex.messages, good, ex.index)[1])


def test_untrained_turn_lists_turn_number(tok, examples):
    _, findings = run_example(tok, examples[1].messages, Config(template=UNTRAINED_FIRST_TURN), 1)
    turn = [f for f in findings if f.rule == "untrained_turn"]
    assert len(turn) == 1 and turn[0].data["turns"] == [1]


def test_content_not_found_is_a_warning(tok, examples):
    _, findings = run_example(tok, examples[0].messages, Config(template=UPPERCASING))
    missing = [f for f in findings if f.rule == "content_not_found"]
    assert len(missing) == 1 and missing[0].severity == "warn"
    assert missing[0].data["messages"] == [0, 1]
