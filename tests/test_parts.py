"""AC-5: Unsloth-style marker strings."""
from __future__ import annotations

from conftest import rules_of, run_example

from masklens.parts import find_all, parts_mask
from masklens.render import Config

INSTRUCTION = "<|im_start|>user\n"
RESPONSE = "<|im_start|>assistant\n"


def test_parts_mask_on_plain_ids():
    ids = [1, 2, 3, 9, 8, 7, 1, 2, 3, 9, 6]
    assert find_all(ids, [1, 2, 3]) == [0, 6]
    mask = parts_mask(ids, instruction_ids=[1, 2, 3], response_ids=[9, 8])
    # Trained from after [9, 8] (index 5) until the next instruction marker at index 6.
    assert mask == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]


def test_parts_mask_without_response_marker_is_all_masked():
    assert parts_mask([1, 2, 3, 4], [1], [7, 7]) == [0, 0, 0, 0]


def test_parts_matches_generation_mask(tok, examples, training_template):
    generation = Config(template=training_template)
    parts = Config(mode="parts", instruction_part=INSTRUCTION, response_part=RESPONSE)
    for ex in examples:
        gen_analysis, _ = run_example(tok, ex.messages, generation, ex.index)
        parts_analysis, findings = run_example(tok, ex.messages, parts, ex.index)
        assert parts_analysis.mask_full == gen_analysis.mask_full
        assert not [f for f in findings if f.severity == "error"], findings


def test_wrong_response_part_not_found(tok, examples):
    parts = Config(mode="parts", instruction_part=INSTRUCTION, response_part="<|im_start|>assistent\n")
    for ex in examples:
        analysis, findings = run_example(tok, ex.messages, parts, ex.index)
        assert {"parts_not_found", "empty_mask"} <= rules_of(findings)
        assert sum(analysis.mask_full) == 0


def test_wrong_response_part_exit_code(examples):
    from conftest import CHAT, TOKENIZER_DIR

    from masklens.cli import main

    code = main(["check", str(TOKENIZER_DIR), str(CHAT), "--mode", "parts", "--instruction-part",
                 "<|im_start|>user\\n", "--response-part", "<|im_start|>assistent\\n"])
    assert code == 1
