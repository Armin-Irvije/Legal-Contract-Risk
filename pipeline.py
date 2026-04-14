from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from env_utils import load_project_env

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

    selected_model = model or DEFAULT_MODEL
    anthropic_client = client or Anthropic(api_key=api_key or _read_api_key())
    rendered_prompt = render_prompt(prompt_template, clause)

    usage_totals = {"input_tokens": 0, "output_tokens": 0}
    attempt_summaries: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    raw_output, response, latency_ms = _invoke_anthropic(client=anthropic_client, model=selected_model, max_tokens=max_tokens, system_prompt=BASE_SYSTEM_PROMPT, user_prompt=rendered_prompt)
    _accumulate_usage(usage_totals, response)
    attempt_summaries.append(
        {
            "attempt": 1,
            "type": "analysis",
            "latency_ms": latency_ms,
            "stop_reason": getattr(response, "stop_reason", None),
        }
    )

    try:
        analysis = parse_and_validate_output(raw_output)
    except ResponseValidationError as exc:
        LOGGER.warning(
            "Initial clause analysis output failed validation for prompt '%s': %s. Raw output: %r",
            prompt_path.name,
            exc,
            raw_output,
        )

        format_fix_prompt = build_format_fix_prompt(clause, raw_output, str(exc))
        repaired_output, repaired_response, repaired_latency_ms = _invoke_anthropic(client=anthropic_client, model=selected_model, max_tokens=max_tokens, system_prompt=FORMAT_FIX_SYSTEM_PROMPT, user_prompt=format_fix_prompt)
        _accumulate_usage(usage_totals, repaired_response)
        attempt_summaries.append(
            {
                "attempt": 2,
                "type": "format_fix",
                "latency_ms": repaired_latency_ms,
                "stop_reason": getattr(repaired_response, "stop_reason", None),
            }
        )

        try:
            analysis = parse_and_validate_output(repaired_output)
        except ResponseValidationError as retry_exc:
            LOGGER.error(
                "Format-fix retry also failed for prompt '%s': %s. Raw retry output: %r",
                prompt_path.name,
                retry_exc,
                repaired_output,
            )
            raise ResponseValidationError(
                "Model output could not be coerced into the required JSON schema."
            ) from retry_exc

    total_latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    total_tokens = usage_totals["input_tokens"] + usage_totals["output_tokens"]

    return {
        "analysis": analysis,
        "metadata": {
            "model": selected_model,
            "prompt": prompt_path.stem,
            "prompt_path": str(prompt_path),
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
    try:
        parsed = json.loads(raw_output)
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


def _invoke_anthropic(client: Anthropic, model: str, max_tokens: int, system_prompt: str, user_prompt: str) -> tuple[str, Any, float]:
    started_at = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return _extract_text(response), response, latency_ms


def _extract_text(response: Any) -> str:
    text_parts = []
    for block in getattr(response, "content", []):
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", ""))

    combined = "".join(text_parts).strip()
    if not combined:
        raise ResponseValidationError("Model response did not contain any text content.")
    return combined


def _read_api_key() -> str:
    for env_var in API_KEY_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return value

    raise PipelineError(
        "Missing Anthropic API key. Set ANTHROPIC_API_KEY or CLAUDE_API_KEY before running the pipeline."
    )


def _accumulate_usage(usage_totals: dict[str, int], response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    usage_totals["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
    usage_totals["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
