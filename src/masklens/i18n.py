"""Message tables (English / Korean) and language resolution.

All user-facing text lives here. Rule IDs and JSON keys are never translated.

Language precedence: ``--lang`` > ``MASKLENS_LANG`` > ``LC_ALL`` > ``LC_MESSAGES`` > ``LANG``.
Anything that does not start with ``ko`` falls back to English.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

LANGS = ("en", "ko")

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        # --- help ---
        "help.description": "Preflight check for SFT loss masks: see which tokens your chat template, "
        "tokenizer and labels will really train on, before you spend GPU time.",
        "help.epilog": "Exit codes: 0 clean, 1 problems found, 2 input error.",
        "help.show": "Show one example with the trained tokens highlighted",
        "help.check": "Check every example against the mask rules",
        "help.tokenizer": "Tokenizer folder (or Hub ID) passed to AutoTokenizer",
        "help.data": "JSONL file with one {\"messages\": [...]} object per line",
        "help.field": "JSON key that holds the message list (default: messages)",
        "help.template": "Jinja chat template file to use instead of the tokenizer's own",
        "help.mode": "Mask method: generation ({% generation %} blocks) or parts "
        "(Unsloth-style instruction/response strings)",
        "help.instruction_part": "parts mode: string that starts a user turn, e.g. \"<|im_start|>user\\n\"",
        "help.response_part": "parts mode: string that starts an assistant turn, e.g. \"<|im_start|>assistant\\n\"",
        "help.max_length": "Truncate to this many tokens before judging the mask (like the trainer would)",
        "help.truncation": "Which side to keep when truncating: keep_start or keep_end",
        "help.lang": "Display language: en or ko",
        "help.index": "0-based example number to show (default: 0)",
        "help.no_color": "Mark trained tokens with [[...]] instead of ANSI colors",
        "help.limit": "Only check the first N examples",
        "help.json": "Print a machine-readable JSON report instead of the table",
        "help.strict": "Treat warnings as failures (exit code 1)",
        "help.version": "Show the version and exit",
        "help.help": "Show this help message and exit",
        # --- check table ---
        "table.title": "masklens check",
        "table.tokenizer": "tokenizer",
        "table.template": "template",
        "table.template_original": "tokenizer's own",
        "table.mode": "mode",
        "table.max_length": "max_length",
        "table.none": "none",
        "table.examples": "examples checked",
        "table.col_rule": "RULE",
        "table.col_severity": "SEVERITY",
        "table.col_count": "COUNT",
        "table.col_examples": "EXAMPLES",
        "table.dataset": "(dataset)",
        "table.more": "+{n} more",
        "table.clean": "No findings.",
        "table.fix": "How to fix",
        "table.result_pass": "Result: PASS",
        "table.result_fail": "Result: FAIL ({errors} error(s), {warnings} warning(s))",
        "table.result_warn": "Result: PASS with {warnings} warning(s)",
        "table.result_strict": "Result: FAIL in strict mode ({warnings} warning(s))",
        "table.detail": "e.g. example {index}: {detail}",
        # --- show ---
        "show.title": "masklens show - example {index}",
        "show.trained": "trained {n}/{total} tokens",
        "show.truncated": "truncated to {kept} of {full} tokens ({truncation})",
        "show.legend_plain": "Trained tokens are wrapped in [[ ]].",
        "show.legend_color": "Trained tokens are highlighted.",
        "show.nothing": "No token is trained in this example.",
        "show.findings": "Findings for this example",
        # --- rule descriptions ---
        "rule.no_generation_tag": "Template has no {% generation %} block",
        "rule.empty_mask": "No trained token left (all labels would be -100)",
        "rule.prompt_leak": "Prompt text (system/user/tool) is trained",
        "rule.render_mismatch": "--template renders differently from the original template",
        "rule.parts_not_found": "Response string not found in the rendered tokens",
        "rule.response_truncated": "Truncation cut into the assistant answer",
        "rule.untrained_turn": "An assistant turn has no trained token",
        "rule.eos_not_trained": "End-of-turn token after the answer is not trained",
        "rule.content_not_found": "Message content not found verbatim in the rendered text",
        # --- fixes ---
        "hint.no_generation_tag": "Pass a training template with --template (wrap the assistant "
        "answer in {% generation %}...{% endgeneration %}) or use --mode parts.",
        "hint.empty_mask": "Raise --max-length, try --truncation keep_end, or fix the template markers.",
        "hint.prompt_leak": "Move {% generation %} so it covers only the assistant content "
        "(plus its end token), not system/user text.",
        "hint.render_mismatch": "Make the training template render the exact same text as the original; "
        "only add {% generation %} markers.",
        "hint.parts_not_found": "Copy the response string from a rendered example (see `masklens show`); "
        "check spaces and newlines.",
        "hint.response_truncated": "Raise --max-length, shorten the prompts, or drop these examples.",
        "hint.untrained_turn": "Check that every assistant branch of the template is inside {% generation %}.",
        "hint.eos_not_trained": "Put the end token (e.g. <|im_end|>) inside the generation block "
        "so the model learns to stop.",
        "hint.content_not_found": "The template rewrites the content (trim, think removal); "
        "masks for that message could not be cross-checked.",
        # --- finding details ---
        "detail.no_generation_tag": "the template used has no generation block",
        "detail.empty_mask_no_assistant": "no assistant message in this example, so nothing can be trained; remove the example or add an answer",
        "detail.empty_mask": "0 of {tokens} kept tokens are trained",
        "detail.prompt_leak": "{count} prompt token(s) trained, first in message {message} ({role})",
        "detail.render_mismatch": "first difference at character {position}",
        "detail.parts_not_found": "response string not found in {tokens} tokens",
        "detail.response_truncated": "{lost} answer token(s) cut off",
        "detail.untrained_turn": "assistant turn(s) {turns} untrained",
        "detail.eos_not_trained": "end token not trained after assistant turn(s) {turns}",
        "detail.content_not_found": "message(s) {messages} not found in the rendered text",
        # --- errors ---
        "err.file_not_found": "File not found: {path}. Check the path.",
        "err.encoding": "{path}:{line}: not UTF-8 text. Save the file as UTF-8 (JSONL must be UTF-8).",
        "err.json_line": "{path}:{line}: invalid JSON ({reason}). Fix or remove that line.",
                "err.shape_not_object": "{path}:{line}: each line must be a JSON object",
        "err.shape_missing_field": "{path}:{line}: missing field \"{field}\" (use --field to change the key)",
        "err.shape_not_list": "{path}:{line}: \"{field}\" must be a non-empty list of messages",
        "err.shape_bad_message": "{path}:{line}: message {n} must be an object with string \"role\" and string \"content\"",
        "err.empty_data": "{path} has no examples.",
        "err.tokenizer": "Could not load the tokenizer from {path}: {reason}",
        "err.no_template": "The tokenizer has no chat template. Pass one with --template file.jinja.",
        "err.render": "Rendering example {index} failed: {reason}",
        "err.tokenization": "Example {index}: rendered text does not tokenize to the same ids as "
        "apply_chat_template. This tokenizer is not supported.",
        "err.parts_required": "--mode parts needs both --instruction-part and --response-part.",
        "err.parts_empty": "{which} produced no tokens. Give a non-empty string.",
        "err.bad_max_length": "--max-length must be a positive integer.",
        "err.index_range": "--index {index} is out of range (the file has {count} example(s)).",
        "err.bad_limit": "--limit must be a positive integer.",
        "err.template_read": "Could not read the template file {path}: {reason}",
    },
    "ko": {
        # --- help ---
        "help.description": "SFT 손실 마스크 사전 점검: 채팅 템플릿·토크나이저·레이블이 실제로 "
        "어떤 토큰을 학습하는지 GPU 를 쓰기 전에 확인합니다.",
        "help.epilog": "종료 코드: 0 문제 없음, 1 문제 발견, 2 입력 오류.",
        "help.show": "예시 하나에서 학습되는 토큰을 강조해 보여 줍니다",
        "help.check": "모든 예시를 마스크 규칙으로 점검합니다",
        "help.tokenizer": "AutoTokenizer 에 넘길 토크나이저 폴더(또는 Hub ID)",
        "help.data": "한 줄에 {\"messages\": [...]} 하나인 JSONL 파일",
        "help.field": "메시지 목록이 들어 있는 JSON 키(기본: messages)",
        "help.template": "토크나이저 기본 템플릿 대신 쓸 Jinja 채팅 템플릿 파일",
        "help.mode": "마스크 방식: generation({% generation %} 블록) 또는 "
        "parts(Unsloth 식 instruction/response 구분 문자열)",
        "help.instruction_part": "parts 모드: 사용자 턴이 시작되는 문자열, 예: \"<|im_start|>user\\n\"",
        "help.response_part": "parts 모드: 어시스턴트 턴이 시작되는 문자열, 예: \"<|im_start|>assistant\\n\"",
        "help.max_length": "마스크를 판정하기 전에 이 토큰 수로 자릅니다(학습기와 같게)",
        "help.truncation": "자를 때 남길 쪽: keep_start 또는 keep_end",
        "help.lang": "표시 언어: en 또는 ko",
        "help.index": "보여 줄 예시 번호(0부터, 기본 0)",
        "help.no_color": "ANSI 색 대신 학습 토큰을 [[...]] 로 표시합니다",
        "help.limit": "앞의 N개 예시만 점검합니다",
        "help.json": "표 대신 기계가 읽는 JSON 보고서를 출력합니다",
        "help.strict": "경고도 실패로 봅니다(종료 코드 1)",
        "help.version": "버전을 표시하고 끝냅니다",
        "help.help": "이 도움말을 표시하고 끝냅니다",
        # --- check table ---
        "table.title": "masklens check",
        "table.tokenizer": "토크나이저",
        "table.template": "템플릿",
        "table.template_original": "토크나이저 기본값",
        "table.mode": "방식",
        "table.max_length": "max_length",
        "table.none": "없음",
        "table.examples": "점검한 예시",
        "table.col_rule": "규칙",
        "table.col_severity": "심각도",
        "table.col_count": "개수",
        "table.col_examples": "예시 번호",
        "table.dataset": "(데이터셋)",
        "table.more": "외 {n}개",
        "table.clean": "발견된 문제가 없습니다.",
        "table.fix": "고치는 방법",
        "table.result_pass": "결과: 통과",
        "table.result_fail": "결과: 실패 (오류 {errors}건, 경고 {warnings}건)",
        "table.result_warn": "결과: 통과 (경고 {warnings}건)",
        "table.result_strict": "결과: 엄격 모드에서 실패 (경고 {warnings}건)",
        "table.detail": "예: 예시 {index}: {detail}",
        # --- show ---
        "show.title": "masklens show - 예시 {index}",
        "show.trained": "학습 토큰 {n}/{total}개",
        "show.truncated": "{full}개 중 {kept}개로 절단됨 ({truncation})",
        "show.legend_plain": "학습되는 토큰은 [[ ]] 로 감쌉니다.",
        "show.legend_color": "학습되는 토큰은 색으로 강조합니다.",
        "show.nothing": "이 예시에서는 학습되는 토큰이 하나도 없습니다.",
        "show.findings": "이 예시의 점검 결과",
        # --- rule descriptions ---
        "rule.no_generation_tag": "템플릿에 {% generation %} 블록이 없음",
        "rule.empty_mask": "학습 토큰이 하나도 남지 않음(레이블이 전부 -100)",
        "rule.prompt_leak": "프롬프트(system/user/tool) 글자가 학습됨",
        "rule.render_mismatch": "--template 의 렌더 결과가 원본 템플릿과 다름",
        "rule.parts_not_found": "렌더된 토큰에서 response 구분 문자열을 못 찾음",
        "rule.response_truncated": "절단이 어시스턴트 답변을 잘라 냄",
        "rule.untrained_turn": "학습 토큰이 없는 어시스턴트 턴이 있음",
        "rule.eos_not_trained": "답변 뒤의 턴 종료 토큰이 학습되지 않음",
        "rule.content_not_found": "메시지 내용이 렌더 문자열에서 그대로 발견되지 않음",
        # --- fixes ---
        "hint.no_generation_tag": "--template 로 학습용 템플릿을 넘기거나(어시스턴트 답변을 "
        "{% generation %}...{% endgeneration %} 로 감싸기) --mode parts 를 쓰세요.",
        "hint.empty_mask": "--max-length 를 늘리거나 --truncation keep_end 를 써 보고, 템플릿 표시를 고치세요.",
        "hint.prompt_leak": "{% generation %} 이 어시스턴트 내용(과 종료 토큰)만 덮도록 옮기세요. "
        "system/user 글자는 빼야 합니다.",
        "hint.render_mismatch": "학습용 템플릿이 원본과 똑같은 글자를 렌더하게 하세요. "
        "{% generation %} 표시만 추가해야 합니다.",
        "hint.parts_not_found": "`masklens show` 로 렌더된 예시를 보고 response 문자열을 그대로 복사하세요. "
        "공백과 줄바꿈도 확인하세요.",
        "hint.response_truncated": "--max-length 를 늘리거나 프롬프트를 줄이거나 해당 예시를 빼세요.",
        "hint.untrained_turn": "템플릿의 모든 어시스턴트 분기가 {% generation %} 안에 있는지 확인하세요.",
        "hint.eos_not_trained": "종료 토큰(예: <|im_end|>)을 generation 블록 안에 넣어 "
        "모델이 멈추는 법을 배우게 하세요.",
        "hint.content_not_found": "템플릿이 내용을 바꿔 출력합니다(공백 제거, think 제거 등). "
        "해당 메시지의 마스크는 교차 확인하지 못했습니다.",
        # --- finding details ---
        "detail.no_generation_tag": "사용한 템플릿에 generation 블록이 없음",
        "detail.empty_mask_no_assistant": "이 예시에는 어시스턴트 메시지가 없어 학습할 토큰이 없습니다. 예시를 빼거나 답변을 추가하세요",
        "detail.empty_mask": "남은 {tokens}개 토큰 중 학습 토큰 0개",
        "detail.prompt_leak": "프롬프트 토큰 {count}개가 학습됨, 처음은 메시지 {message}({role})",
        "detail.render_mismatch": "{position}번째 글자에서 처음 달라짐",
        "detail.parts_not_found": "{tokens}개 토큰에서 response 문자열을 못 찾음",
        "detail.response_truncated": "답변 토큰 {lost}개가 잘림",
        "detail.untrained_turn": "어시스턴트 턴 {turns} 학습 안 됨",
        "detail.eos_not_trained": "어시스턴트 턴 {turns} 뒤 종료 토큰이 학습 안 됨",
        "detail.content_not_found": "메시지 {messages} 을(를) 렌더 문자열에서 못 찾음",
        # --- errors ---
        "err.file_not_found": "파일을 찾을 수 없습니다: {path}. 경로를 확인하세요.",
        "err.encoding": "{path}:{line}: UTF-8 텍스트가 아닙니다. 파일을 UTF-8 로 다시 저장하세요(JSONL 은 UTF-8 이어야 합니다).",
        "err.json_line": "{path}:{line}: JSON 이 올바르지 않습니다({reason}). 그 줄을 고치거나 지우세요.",
                "err.shape_not_object": "{path}:{line}: 각 줄은 JSON 객체여야 합니다",
        "err.shape_missing_field": "{path}:{line}: \"{field}\" 필드가 없습니다(--field 로 키를 바꿀 수 있습니다)",
        "err.shape_not_list": "{path}:{line}: \"{field}\" 는 비어 있지 않은 메시지 목록이어야 합니다",
        "err.shape_bad_message": "{path}:{line}: 메시지 {n} 은 문자열 \"role\" 과 문자열 \"content\" 를 가진 객체여야 합니다",
        "err.empty_data": "{path} 에 예시가 없습니다.",
        "err.tokenizer": "{path} 에서 토크나이저를 불러오지 못했습니다: {reason}",
        "err.no_template": "토크나이저에 채팅 템플릿이 없습니다. --template file.jinja 로 넘겨 주세요.",
        "err.render": "예시 {index} 렌더링에 실패했습니다: {reason}",
        "err.tokenization": "예시 {index}: 렌더된 글자를 토큰화한 결과가 apply_chat_template 의 id 와 다릅니다. "
        "지원하지 않는 토크나이저입니다.",
        "err.parts_required": "--mode parts 에는 --instruction-part 와 --response-part 가 모두 필요합니다.",
        "err.parts_empty": "{which} 가 토큰을 만들지 못했습니다. 비어 있지 않은 문자열을 주세요.",
        "err.bad_max_length": "--max-length 는 양의 정수여야 합니다.",
        "err.index_range": "--index {index} 가 범위를 벗어났습니다(파일에 예시 {count}개).",
        "err.bad_limit": "--limit 는 양의 정수여야 합니다.",
        "err.template_read": "템플릿 파일 {path} 을(를) 읽지 못했습니다: {reason}",
    },
}


def resolve_lang(cli_lang: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """Pick the display language: ``--lang`` > ``MASKLENS_LANG`` > ``LC_ALL`` > ``LC_MESSAGES`` > ``LANG``."""
    env = os.environ if env is None else env
    if cli_lang in LANGS:
        return cli_lang  # type: ignore[return-value]
    explicit = env.get("MASKLENS_LANG", "").strip().lower()
    if explicit in LANGS:
        return explicit
    for key in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = env.get(key, "").strip()
        if value:  # the first variable that is set decides, like POSIX locale lookup
            return "ko" if value.lower().startswith("ko") else "en"
    return "en"


def t(lang: str, key: str, **params: object) -> str:
    """Look up ``key`` in ``lang`` (falls back to English) and fill ``params``."""
    table = MESSAGES.get(lang, MESSAGES["en"])
    template = table.get(key) or MESSAGES["en"][key]
    return template.format(**params) if params else template
