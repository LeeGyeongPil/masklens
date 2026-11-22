"""Rebuild examples/tiny-chatml deterministically (a tiny practice tokenizer, not a real model's).

Usage: uv run python scripts/make_tiny_tokenizer.py
"""
from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

OUT = Path(__file__).resolve().parent.parent / "examples" / "tiny-chatml"

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

# ChatML-style template WITHOUT any {% generation %} block (the "silent failure" case).
ORIGINAL_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

# Same rendered text, but the assistant answer, its end token and the newline are trainable.
TRAINING_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- generation %}{{- message.content + '<|im_end|>' + '\\n' }}{%- endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

CORPUS = [
    "You are a helpful assistant.",
    "What is the capital of France?",
    "The capital of France is Paris.",
    "Write a short poem about the sea.",
    "The sea is wide and the waves are blue, the wind is soft and the sky is new.",
    "How do I reverse a list in Python?",
    "Use the reverse method or slicing, for example items[::-1].",
    "Explain what a loss mask is.",
    "A loss mask tells the trainer which tokens count toward the loss.",
    "Thank you, that was very helpful!",
    "You are welcome. Let me know if you have more questions.",
    "What is 2+2? The answer is 4. And 3+3 is 6.",
    "Summarize the following text in one sentence.",
    "Please translate this sentence into Spanish.",
    "Hello there, how are you today? I am fine, thanks.",
]


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS * 3, trainer)
    tok.save(str(OUT / "tokenizer.json"))
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": None,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": ["<|im_start|>"],
        "clean_up_tokenization_spaces": False,
        "model_max_length": 2048,
        "chat_template": ORIGINAL_TEMPLATE,
    }
    (OUT / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "training.jinja").write_text(TRAINING_TEMPLATE + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
