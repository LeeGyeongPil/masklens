"""Command line entry point: ``masklens show`` and ``masklens check``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from . import __version__
from .i18n import LANGS, resolve_lang, t
from .loader import InputError, load_examples, load_template, load_tokenizer
from .render import MODES, Config, analyze
from .report import check, render_show, render_table, to_json
from .rules import dataset_findings, example_findings
from .truncate import TRUNCATION_MODES


def _h(lang: str, key: str) -> str:
    """Help text for argparse (``%`` must be doubled there)."""
    return t(lang, key).replace("%", "%%")


def _unescape(value: str) -> str:
    """Turn the two-character sequences ``\\n`` and ``\\t`` typed in a shell into real characters."""
    return value.replace("\\n", "\n").replace("\\t", "\t")


def _common(p: argparse.ArgumentParser, lang: str) -> None:
    p.add_argument("tokenizer", help=_h(lang, "help.tokenizer"))
    p.add_argument("data", help=_h(lang, "help.data"))
    p.add_argument("--field", default="messages", help=_h(lang, "help.field"))
    p.add_argument("--template", metavar="FILE", help=_h(lang, "help.template"))
    p.add_argument("--mode", choices=MODES, default="generation", help=_h(lang, "help.mode"))
    p.add_argument("--instruction-part", metavar="TEXT", help=_h(lang, "help.instruction_part"))
    p.add_argument("--response-part", metavar="TEXT", help=_h(lang, "help.response_part"))
    p.add_argument("--max-length", type=int, metavar="N", help=_h(lang, "help.max_length"))
    p.add_argument("--truncation", choices=TRUNCATION_MODES, default="keep_start",
                   help=_h(lang, "help.truncation"))
    p.add_argument("--lang", choices=LANGS, help=_h(lang, "help.lang"))
    p.add_argument("-h", "--help", action="help", help=_h(lang, "help.help"))


def build_parser(lang: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="masklens", description=_h(lang, "help.description"), epilog=_h(lang, "help.epilog"),
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help=_h(lang, "help.help"))
    parser.add_argument("--version", action="version", version=f"masklens {__version__}",
                        help=_h(lang, "help.version"))
    parser.add_argument("--lang", choices=LANGS, help=_h(lang, "help.lang"))
    sub = parser.add_subparsers(dest="command", required=True, metavar="{show,check}")

    show = sub.add_parser("show", help=_h(lang, "help.show"), description=_h(lang, "help.show"),
                          add_help=False)
    _common(show, lang)
    show.add_argument("--index", type=int, default=0, metavar="I", help=_h(lang, "help.index"))
    show.add_argument("--no-color", action="store_true", help=_h(lang, "help.no_color"))

    chk = sub.add_parser("check", help=_h(lang, "help.check"), description=_h(lang, "help.check"),
                         add_help=False)
    _common(chk, lang)
    chk.add_argument("--limit", type=int, metavar="N", help=_h(lang, "help.limit"))
    chk.add_argument("--json", action="store_true", help=_h(lang, "help.json"))
    chk.add_argument("--strict", action="store_true", help=_h(lang, "help.strict"))
    return parser


def _peek_lang(argv: Sequence[str]) -> str | None:
    """Find ``--lang`` before the real parser exists, so ``--help`` can be localized."""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


def _config(args: argparse.Namespace, lang: str) -> tuple[Config, str]:
    if args.max_length is not None and args.max_length < 1:
        raise InputError("err.bad_max_length")
    template = load_template(args.template) if args.template else None
    if args.mode == "parts" and not (args.instruction_part and args.response_part):
        raise InputError("err.parts_required")
    config = Config(
        mode=args.mode, template=template, max_length=args.max_length, truncation=args.truncation,
        instruction_part=_unescape(args.instruction_part) if args.instruction_part else None,
        response_part=_unescape(args.response_part) if args.response_part else None,
    )
    return config, (args.template if args.template else "original")


def _use_color(no_color: bool) -> bool:
    return not no_color and sys.stdout.isatty() and "NO_COLOR" not in os.environ


def run(args: argparse.Namespace, lang: str) -> int:
    config, label = _config(args, lang)
    if args.command == "check" and args.limit is not None and args.limit < 1:
        raise InputError("err.bad_limit")
    tokenizer = load_tokenizer(args.tokenizer)
    if args.command == "show":
        examples = load_examples(args.data, args.field)
        if not 0 <= args.index < len(examples):
            raise InputError("err.index_range", index=args.index, count=len(examples))
        ex = examples[args.index]
        analysis = analyze(tokenizer, ex.messages, config, ex.index)
        findings = dataset_findings(tokenizer, config) + example_findings(ex.index, analysis, config)
        print(render_show(analysis, ex.index, config, findings, lang, _use_color(args.no_color)))
        return 0

    examples = load_examples(args.data, args.field, args.limit)
    report = check(tokenizer, examples, config, args.tokenizer, label)
    if args.json:
        print(json.dumps(to_json(report, args.strict), indent=2, ensure_ascii=False))
    else:
        print(render_table(report, lang, args.strict))
    return report.exit_code(args.strict)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # silence import-time notices
    lang = resolve_lang(_peek_lang(argv))
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser(lang).parse_args(argv)
    lang = resolve_lang(args.lang)
    try:
        return run(args, lang)
    except InputError as exc:
        print(t(lang, exc.key, **exc.params), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
