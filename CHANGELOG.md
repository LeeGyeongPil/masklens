# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-11-23

### Fixed
- CI workflow: pinned the Python matrix and the uv setup step.

## [0.1.0] - 2026-11-23

### Added
- `masklens show`: one example with the trained tokens highlighted (`[[...]]` or ANSI color).
- `masklens check`: dataset-wide rules (`no_generation_tag`, `empty_mask`, `prompt_leak`, `render_mismatch`, `parts_not_found`, `response_truncated`, `untrained_turn`, `eos_not_trained`, plus the informational `content_not_found`), summary table, `--json` report and exit codes.
- Two mask methods: `--mode generation` and `--mode parts` (Unsloth-style marker strings).
- Own truncation with `--max-length` and `--truncation keep_start|keep_end`.
- English and Korean screens (`--lang`, `MASKLENS_LANG`, locale variables).
- Practice tokenizer, training template and sample data under `examples/`.
