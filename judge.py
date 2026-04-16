from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any

from env_utils import load_project_env
from telemetry import extract_text_from_response
from telemetry import invoke_anthropic_with_retry

try:
    from anthropic import Anthropic
except ImportError as exc:  # pragma: no cover - depends on local environment.
    Anthropic = Any  # type: ignore[assignment]
    ANTHROPIC_IMPORT_ERROR = exc
else:
    ANTHROPIC_IMPORT_ERROR = None

load_project_env(__file__)

LOGGER = logging.getLogger(__name__)
ALLOWED_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}
ALLOWED_LABEL_ACCURACY = {"pass", "fail"}
REQUIRED_PIPELINE_OUTPUT_KEYS = ("risk_level", "explanation", "suggested_redline")
REQUIRED_JUDGE_KEYS = ("label_accuracy", "explanation_quality", "redline_usefulness")
DEFAULT_MODEL = os.environ["JUDGE_MODEL"]
DEFAULT_MAX_TOKENS = int(os.environ["JUDGE_MAX_TOKENS"])
API_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
MAX_LOG_OUTPUT_CHARS = 600

BASE_SYSTEM_PROMPT = """
You are a strict evaluator for a legal clause risk review system.
Return JSON only with no markdown, commentary, or surrounding text.
The JSON object must contain exactly these keys:
- label_accuracy: "pass" or "fail"
- explanation_quality: integer 1-5
- redline_usefulness: integer 1-5
""".strip()

FORMAT_FIX_SYSTEM_PROMPT = """
You repair malformed judge outputs into strict JSON.
Return JSON only with exactly these keys:
- label_accuracy
- explanation_quality
- redline_usefulness
Use label_accuracy values "pass" or "fail" only.
Use integer scores from 1 to 5 only.
Do not add markdown or extra text.
""".strip()


class JudgeError(RuntimeError):
    """Base exception for judge scoring failures."""


class JudgeValidationError(JudgeError):
    """Raised when the judge output does not match the required schema."""


class ProviderRequestError(JudgeError):
    """Raised when the provider request fails after retry attempts."""


def score_output(
    clause_text: str,
    pipeline_output: dict[str, Any] | str,
    ground_truth_label: str,
    model: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: str | None = None,
    client: Anthropic | None = None,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    clause = clause_text.strip()
    if not clause:
        raise ValueError("clause_text must be a non-empty string.")

    normalized_ground_truth = _normalize_ground_truth_label(ground_truth_label)
    normalized_pipeline_output = _normalize_pipeline_output(pipeline_output)

    if client is None and ANTHROPIC_IMPORT_ERROR is not None:
        raise JudgeError(
            "The anthropic package is not installed. Install dependencies from requirements.txt before running the judge."
        ) from ANTHROPIC_IMPORT_ERROR

    selected_model = model or DEFAULT_MODEL
    anthropic_client = client or Anthropic(api_key=api_key or _read_api_key())
    judge_prompt = build_judge_prompt(clause, normalized_pipeline_output, normalized_ground_truth)

    usage_totals = {"input_tokens": 0, "output_tokens": 0}
    attempt_summaries: list[dict[str, Any]] = []
    judge_invocation = invoke_anthropic_with_retry(
        client=anthropic_client,
        model=selected_model,
        max_tokens=max_tokens,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=judge_prompt,
        operation="judge",
        error_cls=ProviderRequestError,
        prompt_version=prompt_version,
    )
    raw_output = extract_text_from_response(judge_invocation["response"])
    if not raw_output:
        raise JudgeValidationError("Model response did not contain any text content.")
    _accumulate_usage(usage_totals, judge_invocation["usage"])
    attempt_summaries.append(build_call_summary(call_index=1, call_type="judge", invocation=judge_invocation))

    try:
        scores = parse_and_validate_judge_output(raw_output)
    except JudgeValidationError as exc:
        LOGGER.warning(
            "Judge returned invalid JSON.\nReason: %s\nRaw output:\n%s\nRetrying with JSON repair prompt.",
            exc,
            format_model_output_for_log(raw_output),
        )
        format_fix_prompt = build_format_fix_prompt(judge_prompt, raw_output, str(exc))
        repair_invocation = invoke_anthropic_with_retry(
            client=anthropic_client,
            model=selected_model,
            max_tokens=max_tokens,
            system_prompt=FORMAT_FIX_SYSTEM_PROMPT,
            user_prompt=format_fix_prompt,
            operation="judge_format_fix",
            error_cls=ProviderRequestError,
            prompt_version=prompt_version,
        )
        repaired_output = extract_text_from_response(repair_invocation["response"])
        if not repaired_output:
            raise JudgeValidationError("Model response did not contain any text content.")
        _accumulate_usage(usage_totals, repair_invocation["usage"])
        attempt_summaries.append(build_call_summary(call_index=2, call_type="format_fix", invocation=repair_invocation))

        try:
            scores = parse_and_validate_judge_output(repaired_output)
        except JudgeValidationError as retry_exc:
            LOGGER.error(
                "Judge output was still invalid after the repair attempt.\nInitial error: %s\nRetry error: %s\nRetry output:\n%s",
                exc,
                retry_exc,
                format_model_output_for_log(repaired_output),
            )
            raise JudgeValidationError(
                "Judge output could not be converted into the required JSON object after the repair attempt."
            ) from retry_exc

    total_tokens = usage_totals["input_tokens"] + usage_totals["output_tokens"]
    total_latency_ms = round(sum(call_summary["latency_ms"] for call_summary in attempt_summaries), 2)

    return {
        "scores": scores,
        "metadata": {
            "model": selected_model,
            "prompt_version": prompt_version,
            "latency_ms": total_latency_ms,
            "tokens": {
                "input_tokens": usage_totals["input_tokens"],
                "output_tokens": usage_totals["output_tokens"],
                "total_tokens": total_tokens,
            },
            "attempt_count": len(attempt_summaries),
            "attempts": attempt_summaries,
            "ground_truth_label": normalized_ground_truth,
        },
    }


def build_judge_prompt(
    clause_text: str,
    pipeline_output: dict[str, str],
    ground_truth_label: str,
) -> str:
    serialized_pipeline_output = json.dumps(pipeline_output, indent=2)
    return (
        "Evaluate the pipeline output for the contract clause below.\n"
        "Apply the rubric strictly and return JSON only.\n\n"
        "Rubric anchors:\n"
        '- label_accuracy: "pass" only if the pipeline risk_level exactly matches the ground truth label. '
        '"fail" for any mismatch.\n'
        "- explanation_quality:\n"
        "  1 = incorrect, generic, or unsupported by the clause text.\n"
        "  3 = partly correct and somewhat clause-specific, but misses important nuance or business impact.\n"
        "  5 = accurate, clause-specific, concise, and explains the practical business risk clearly.\n"
        "- redline_usefulness:\n"
        "  1 = unusable, missing, or does not mitigate the clause risk.\n"
        "  3 = somewhat useful but incomplete, overbroad, or not contract-ready.\n"
        "  5 = concrete, contract-ready, and directly mitigates the identified risk while preserving business intent.\n\n"
        f"Original clause:\n{clause_text}\n\n"
        f"Ground truth risk label:\n{ground_truth_label}\n\n"
        "Pipeline output JSON:\n"
        f"{serialized_pipeline_output}\n\n"
        "Return exactly this JSON shape:\n"
        '{"label_accuracy":"pass","explanation_quality":4,"redline_usefulness":4}'
    )


def build_format_fix_prompt(original_prompt: str, raw_output: str, validation_error: str) -> str:
    return (
        "The earlier judge response was not valid for the required schema.\n"
        "Repair it into strict JSON using only the allowed keys: "
        "label_accuracy, explanation_quality, redline_usefulness.\n\n"
        f"Validation error:\n{validation_error}\n\n"
        f"Original judge instructions:\n{original_prompt}\n\n"
        "Invalid model output to repair:\n"
        f"{raw_output}"
    )


def parse_and_validate_judge_output(raw_output: str) -> dict[str, Any]:
    json_candidate = normalize_json_candidate(raw_output)
    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError as exc:
        raise JudgeValidationError(f"Output is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise JudgeValidationError("Output JSON must be an object.")

    missing_keys = [key for key in REQUIRED_JUDGE_KEYS if key not in parsed]
    if missing_keys:
        raise JudgeValidationError(f"Missing required keys: {', '.join(missing_keys)}")

    extra_keys = [key for key in parsed if key not in REQUIRED_JUDGE_KEYS]
    if extra_keys:
        raise JudgeValidationError(f"Unexpected keys present: {', '.join(extra_keys)}")

    label_accuracy = parsed["label_accuracy"]
    if not isinstance(label_accuracy, str):
        raise JudgeValidationError("Field 'label_accuracy' must be a string.")

    normalized_label_accuracy = label_accuracy.strip().lower()
    if normalized_label_accuracy not in ALLOWED_LABEL_ACCURACY:
        raise JudgeValidationError("Field 'label_accuracy' must be 'pass' or 'fail'.")

    explanation_quality = _validate_score_field(parsed["explanation_quality"], "explanation_quality")
    redline_usefulness = _validate_score_field(parsed["redline_usefulness"], "redline_usefulness")

    return {
        "label_accuracy": normalized_label_accuracy,
        "explanation_quality": explanation_quality,
        "redline_usefulness": redline_usefulness,
    }


def normalize_json_candidate(raw_output: str) -> str:
    """Strip common markdown fencing before JSON validation."""
    stripped_output = raw_output.strip()
    if not stripped_output.startswith("```"):
        return stripped_output

    lines = stripped_output.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return stripped_output

    opening_fence = lines[0].strip()
    if opening_fence not in {"```", "```json"}:
        return stripped_output

    return "\n".join(lines[1:-1]).strip()


def format_model_output_for_log(raw_output: str, max_chars: int = MAX_LOG_OUTPUT_CHARS) -> str:
    """Format raw model output into an indented, truncated log block."""
    compact_output = raw_output.strip() or "<empty>"
    if len(compact_output) > max_chars:
        compact_output = f"{compact_output[:max_chars].rstrip()}\n... <truncated>"
    return textwrap.indent(compact_output, "  ")


def _normalize_ground_truth_label(label: str) -> str:
    normalized_label = label.strip().upper()
    if normalized_label not in ALLOWED_RISK_LEVELS:
        raise ValueError(
            f"ground_truth_label must be one of {sorted(ALLOWED_RISK_LEVELS)}, got '{label}'."
        )
    return normalized_label


def _normalize_pipeline_output(pipeline_output: dict[str, Any] | str) -> dict[str, str]:
    parsed_output: Any = pipeline_output
    if isinstance(pipeline_output, str):
        try:
            parsed_output = json.loads(pipeline_output)
        except json.JSONDecodeError as exc:
            raise ValueError(f"pipeline_output string must be valid JSON: {exc}") from exc

    if not isinstance(parsed_output, dict):
        raise ValueError("pipeline_output must be a dict or JSON object string.")

    if "analysis" in parsed_output and isinstance(parsed_output["analysis"], dict):
        candidate_output = parsed_output["analysis"]
    else:
        candidate_output = parsed_output

    missing_keys = [key for key in REQUIRED_PIPELINE_OUTPUT_KEYS if key not in candidate_output]
    if missing_keys:
        raise ValueError(
            "pipeline_output must include risk_level, explanation, and suggested_redline."
        )

    normalized_output: dict[str, str] = {}
    for key in REQUIRED_PIPELINE_OUTPUT_KEYS:
        value = candidate_output[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"pipeline_output field '{key}' must be a non-empty string.")
        normalized_output[key] = value.strip()

    normalized_risk_level = normalized_output["risk_level"].upper()
    if normalized_risk_level not in ALLOWED_RISK_LEVELS:
        raise ValueError(
            f"pipeline_output risk_level must be one of {sorted(ALLOWED_RISK_LEVELS)}."
        )

    normalized_output["risk_level"] = normalized_risk_level
    return normalized_output


def _validate_score_field(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JudgeValidationError(f"Field '{field_name}' must be an integer from 1 to 5.")
    if value < 1 or value > 5:
        raise JudgeValidationError(f"Field '{field_name}' must be between 1 and 5.")
    return value


def _read_api_key() -> str:
    for env_var in API_KEY_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return value

    raise JudgeError(
        "Missing Anthropic API key. Set ANTHROPIC_API_KEY or CLAUDE_API_KEY before running the judge."
    )


def _accumulate_usage(usage_totals: dict[str, int], usage: dict[str, int]) -> None:
    usage_totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
    usage_totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)


def build_call_summary(call_index: int, call_type: str, invocation: dict[str, Any]) -> dict[str, Any]:
    """Normalize shared telemetry into the judge metadata shape."""
    return {
        "attempt": call_index,
        "type": call_type,
        "model": invocation["provider_attempts"][-1]["model"],
        "prompt_version": invocation["provider_attempts"][-1].get("prompt_version"),
        "latency_ms": invocation["latency_ms"],
        "tokens": invocation["usage"],
        "provider_attempt_count": invocation["provider_attempt_count"],
        "provider_attempts": invocation["provider_attempts"],
        "stop_reason": getattr(invocation["response"], "stop_reason", None),
    }
