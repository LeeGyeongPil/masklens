"""AC-3: own truncation, partial/full answer loss, keep_end."""
from __future__ import annotations

import pytest
from conftest import rules_of, run_example

from masklens.render import Config
from masklens.truncate import kept_range, lost_tokens


def test_kept_range_basics():
    assert kept_range(10, None) == (0, 10)
    assert kept_range(10, 10) == (0, 10)
    assert kept_range(10, 4, "keep_start") == (0, 4)
    assert kept_range(10, 4, "keep_end") == (6, 10)
    assert lost_tokens([0, 5, 6, 9], (6, 10)) == 2


def test_kept_range_rejects_bad_input():
    with pytest.raises(ValueError, match="truncation"):
        kept_range(10, 4, "middle")
    with pytest.raises(ValueError, match="positive"):
        kept_range(10, 0)


def _long_prompt_example(examples):
    return examples[3].messages  # long user prompt, short answer


def test_keep_start_partial_and_full_loss(tok, examples, training_template):
    messages = _long_prompt_example(examples)
    base = Config(template=training_template)
    full, _ = run_example(tok, messages, base)
    trained = [i for i, m in enumerate(full.mask_full) if m]
    first_answer = trained[0]
    assert len(trained) > 4

    # Cut in the middle of the answer: some answer tokens survive, some are lost.
    cfg = Config(template=training_template, max_length=first_answer + 3)
    analysis, findings = run_example(tok, messages, cfg)
    assert sum(analysis.mask_kept) == 3
    assert "empty_mask" not in rules_of(findings)
    lost = next(f for f in findings if f.rule == "response_truncated").data["lost"]
    # The answer content has fewer tokens than the trained span (end token + newline are trained too).
    answer_tokens = len(analysis.tokens_in(analysis.spans[1]))
    assert lost == answer_tokens - 3
    assert lost > 0

    # Cut right before the answer: nothing is trained any more.
    cfg = Config(template=training_template, max_length=first_answer)
    analysis, findings = run_example(tok, messages, cfg)
    assert sum(analysis.mask_kept) == 0
    assert {"empty_mask", "response_truncated"} <= rules_of(findings)
    assert len(analysis.mask_kept) == first_answer


def test_keep_end_keeps_last_answer(tok, examples, training_template):
    messages = _long_prompt_example(examples)
    full, _ = run_example(tok, messages, Config(template=training_template))
    first_answer = next(i for i, m in enumerate(full.mask_full) if m)
    max_length = len(full.ids) - 10
    assert max_length > len(full.ids) - first_answer  # the answer fits entirely

    start_cfg = Config(template=training_template, max_length=max_length, truncation="keep_start")
    assert {"empty_mask", "response_truncated"} & rules_of(run_example(tok, messages, start_cfg)[1])

    end_cfg = Config(template=training_template, max_length=max_length, truncation="keep_end")
    analysis, findings = run_example(tok, messages, end_cfg)
    assert findings == []
    assert analysis.kept == (10, len(full.ids))
    assert sum(analysis.mask_kept) == sum(full.mask_full)
