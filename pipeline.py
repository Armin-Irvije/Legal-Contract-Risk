from __future__ import annotations

import hashlib
import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any

from env_utils import load_project_env
from telemetry import extract_text_from_response
from telemetry import invoke_anthropic_with_retry

try:
    from anthropic import Anthropic
except ImportError as exc:  # pragma: no cover - depends on local environment.
    Anthropic = Any  # type: ignore[assignment]
    # ignore so module can still be imported even if anthropic is not installed
    ANTHROPIC_IMPORT_ERROR = exc
else:
    ANTHROPIC_IMPORT_ERROR = None

load_project_env(__file__)

LOGGER = logging.getLogger(__name__)

ALLOWED_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}
REQUIRED_OUTPUT_KEYS = ("risk_level", "explanation", "suggested_redline")
DEFAULT_MODEL = os.environ["PIPELINE_MODEL"]
DEFAULT_MAX_TOKENS = int(os.environ["PIPELINE_MAX_TOKENS"])
PROMPT_PLACEHOLDER = "{{CLAUSE_TEXT}}"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
API_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
MAX_LOG_OUTPUT_CHARS = 600

BASE_SYSTEM_PROMPT = """
You analyze a single contract clause for legal risk.
Return JSON only with no markdown, commentary, or surrounding text.
The JSON object must contain exactly these keys:
- risk_level: one of LOW, MEDIUM, HIGH
- explanation: 2-4 plain-English sentences
- suggested_redline: revised clause language that mitigates the risk
""".strip()

FORMAT_FIX_SYSTEM_PROMPT = """
You repair malformed model outputs into strict JSON.
Return JSON only with exactly these keys:
- risk_level
- explanation
- suggested_redline
Use risk_level values LOW, MEDIUM, or HIGH only.
Do not add markdown or extra text.
""".strip()


class PipelineError(RuntimeError):
    """Base exception for the clause analysis pipeline."""


class PromptTemplateError(PipelineError):
    """Raised when a prompt template cannot be loaded or rendered."""


class ResponseValidationError(PipelineError):
    """Raised when the model output is not valid JSON matching the schema."""


class ProviderRequestError(PipelineError):
    """Raised when the provider request fails after retry attempts."""


def analyze_clause(clause_text: str, prompt: str = "v1", model: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS, api_key: str | None = None, client: Anthropic | None = None) -> dict[str, Any]:
    clause = clause_text.strip()
    if not clause:
        raise ValueError("clause_text must be a non-empty string.")
    if client is None and ANTHROPIC_IMPORT_ERROR is not None:
        raise PipelineError(
            "The anthropic package is not installed. Install dependencies from requirements.txt before running the pipeline."
        ) from ANTHROPIC_IMPORT_ERROR

    prompt_path = resolve_prompt_path(prompt)
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_template:
        raise PromptTemplateError(f"Prompt template is empty: {prompt_path}")
    prompt_hash = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()

    selected_model = model or DEFAULT_MODEL
    anthropic_client = client or Anthropic(api_key=api_key or _read_api_key())
    rendered_prompt = render_prompt(prompt_template, clause)

    usage_totals = {"input_tokens": 0, "output_tokens": 0}
    attempt_summaries: list[dict[str, Any]] = []
    analysis_invocation = invoke_anthropic_with_retry(
        client=anthropic_client,
        model=selected_model,
        max_tokens=max_tokens,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=rendered_prompt,
        operation="analysis",
        error_cls=ProviderRequestError,
        prompt_version=prompt_path.stem,
    )
    raw_output = extract_text_from_response(analysis_invocation["response"])
    if not raw_output:
        raise ResponseValidationError("Model response did not contain any text content.")
    _accumulate_usage(usage_totals, analysis_invocation["usage"])
    attempt_summaries.append(build_call_summary(call_index=1, call_type="analysis", invocation=analysis_invocation))

    try:
        analysis = parse_and_validate_output(raw_output)
    except ResponseValidationError as exc:
        LOGGER.warning(
            "Prompt '%s' returned invalid analysis JSON.\nReason: %s\nRaw output:\n%s\nRetrying with JSON repair prompt.",
            prompt_path.name,
            exc,
            format_model_output_for_log(raw_output),
        )

        format_fix_prompt = build_format_fix_prompt(clause, raw_output, str(exc))
        repair_invocation = invoke_anthropic_with_retry(
            client=anthropic_client,
            model=selected_model,
            max_tokens=max_tokens,
            system_prompt=FORMAT_FIX_SYSTEM_PROMPT,
            user_prompt=format_fix_prompt,
            operation="format_fix",
            error_cls=ProviderRequestError,
            prompt_version=prompt_path.stem,
        )
        repaired_output = extract_text_from_response(repair_invocation["response"])
        if not repaired_output:
            raise ResponseValidationError("Model response did not contain any text content.")
        _accumulate_usage(usage_totals, repair_invocation["usage"])
        attempt_summaries.append(build_call_summary(call_index=2, call_type="format_fix", invocation=repair_invocation))

        try:
            analysis = parse_and_validate_output(repaired_output)
        except ResponseValidationError as retry_exc:
            LOGGER.error(
                "Prompt '%s' still returned invalid analysis JSON after the repair attempt.\nInitial error: %s\nRetry error: %s\nRetry output:\n%s",
                prompt_path.name,
                exc,
                retry_exc,
                format_model_output_for_log(repaired_output),
            )
            raise ResponseValidationError(
                "Model output could not be converted into the required JSON object after the repair attempt."
            ) from retry_exc

    total_tokens = usage_totals["input_tokens"] + usage_totals["output_tokens"]
    total_latency_ms = round(sum(call_summary["latency_ms"] for call_summary in attempt_summaries), 2)

    return {
        "analysis": analysis,
        "metadata": {
            "model": selected_model,
            "prompt": prompt_path.stem,
            "prompt_path": str(prompt_path),
            "prompt_hash": prompt_hash,
            "latency_ms": total_latency_ms,
            "tokens": {
                "input_tokens": usage_totals["input_tokens"],
                "output_tokens": usage_totals["output_tokens"],
                "total_tokens": total_tokens,
            },
            "attempt_count": len(attempt_summaries),
            "attempts": attempt_summaries,
        },
    }


def resolve_prompt_path(prompt: str) -> Path:
    prompt_value = prompt.strip()
    if not prompt_value:
        raise PromptTemplateError("Prompt selection cannot be empty.")

    candidate = Path(prompt_value)
    if candidate.is_file():
        return candidate.resolve()

    if not candidate.suffix:
        named_prompt = PROMPTS_DIR / f"{prompt_value}.txt"
        if named_prompt.is_file():
            return named_prompt.resolve()

    if candidate.suffix and candidate.is_file():
        return candidate.resolve()

    raise PromptTemplateError(
        f"Prompt template '{prompt}' was not found. Expected a named prompt in {PROMPTS_DIR} or a file path."
    )


def render_prompt(prompt_template: str, clause_text: str) -> str:
    if PROMPT_PLACEHOLDER in prompt_template:
        return prompt_template.replace(PROMPT_PLACEHOLDER, clause_text)

    return f"{prompt_template.rstrip()}\n\nClause to analyze:\n{clause_text}"


def build_format_fix_prompt(clause_text: str, raw_output: str, validation_error: str) -> str:
    return (
        "The earlier clause analysis response was not valid for the required schema.\n"
        "Repair it into strict JSON using only the allowed keys: risk_level, explanation, suggested_redline.\n\n"
        f"Validation error:\n{validation_error}\n\n"
        f"Original clause:\n{clause_text}\n\n"
        "Invalid model output to repair:\n"
        f"{raw_output}"
    )


def parse_and_validate_output(raw_output: str) -> dict[str, str]:
    parsed: Any
    json_candidate = normalize_json_candidate(raw_output)
    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError(f"Output is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ResponseValidationError("Output JSON must be an object.")

    missing_keys = [key for key in REQUIRED_OUTPUT_KEYS if key not in parsed]
    if missing_keys:
        raise ResponseValidationError(f"Missing required keys: {', '.join(missing_keys)}")

    extra_keys = [key for key in parsed if key not in REQUIRED_OUTPUT_KEYS]
    if extra_keys:
        raise ResponseValidationError(f"Unexpected keys present: {', '.join(extra_keys)}")

    normalized_output = {}
    for key in REQUIRED_OUTPUT_KEYS:
        value = parsed[key]
        if not isinstance(value, str) or not value.strip():
            raise ResponseValidationError(f"Field '{key}' must be a non-empty string.")
        normalized_output[key] = value.strip()

    normalized_risk = normalized_output["risk_level"].upper()
    if normalized_risk not in ALLOWED_RISK_LEVELS:
        raise ResponseValidationError(
            f"risk_level must be one of {sorted(ALLOWED_RISK_LEVELS)}, got '{normalized_output['risk_level']}'."
        )

    normalized_output["risk_level"] = normalized_risk
    return normalized_output


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


def _read_api_key() -> str:
    for env_var in API_KEY_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return value

    raise PipelineError(
        "Missing Anthropic API key. Set ANTHROPIC_API_KEY or CLAUDE_API_KEY before running the pipeline."
    )


def _accumulate_usage(usage_totals: dict[str, int], usage: dict[str, int]) -> None:
    usage_totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
    usage_totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)


def build_call_summary(call_index: int, call_type: str, invocation: dict[str, Any]) -> dict[str, Any]:
    """Normalize shared telemetry into the pipeline metadata shape."""
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
