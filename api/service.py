"""Clause analysis service for the HTTP API using OpenRouter."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from cost import estimate_model_cost_usd
from cost import load_pricing_config
from env_utils import load_project_env
from pipeline import BASE_SYSTEM_PROMPT
from pipeline import FORMAT_FIX_SYSTEM_PROMPT
from pipeline import PromptTemplateError
from pipeline import ProviderRequestError
from pipeline import ResponseValidationError
from pipeline import build_format_fix_prompt
from pipeline import format_model_output_for_log
from pipeline import parse_and_validate_output
from pipeline import render_prompt
from pipeline import resolve_prompt_path

from api.providers import OpenRouterError
from api.providers import create_openrouter_client
from api.providers import invoke_openrouter_with_retry

# Load repo-root .env (same file the CLI uses).
load_project_env(str(Path(__file__).resolve().parent.parent / "env_utils.py"))

LOGGER = logging.getLogger(__name__)

DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
PRICING_PATH = Path(__file__).resolve().parent.parent / "pricing.json"


def analyze_clause_openrouter(
    clause_text: str,
    prompt: str = "v1",
    model: str | None = None,
    max_tokens: int | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Analyze a clause via OpenRouter using shared prompt and validation helpers."""
    clause = clause_text.strip()
    if not clause:
        raise ValueError("clause_text must be a non-empty string.")

    prompt_path = resolve_prompt_path(prompt)
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_template:
        raise PromptTemplateError(f"Prompt template is empty: {prompt_path}")

    prompt_hash = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    selected_model = model or os.getenv("CLAUSEGUARD_MODEL") or DEFAULT_OPENROUTER_MODEL
    token_limit = max_tokens if max_tokens is not None else int(os.getenv("PIPELINE_MAX_TOKENS", "900"))
    openrouter_client = client or create_openrouter_client()
    rendered_prompt = render_prompt(prompt_template, clause)

    usage_totals = {"input_tokens": 0, "output_tokens": 0}
    attempt_summaries: list[dict[str, Any]] = []

    try:
        analysis_invocation = invoke_openrouter_with_retry(
            client=openrouter_client,
            model=selected_model,
            max_tokens=token_limit,
            system_prompt=BASE_SYSTEM_PROMPT,
            user_prompt=rendered_prompt,
            operation="analysis",
            prompt_version=prompt_path.stem,
        )
    except OpenRouterError as exc:
        raise ProviderRequestError(str(exc)) from exc

    raw_output = analysis_invocation["text"]
    if not raw_output:
        raise ResponseValidationError("Model response did not contain any text content.")
    _accumulate_usage(usage_totals, analysis_invocation["usage"])
    attempt_summaries.append(_build_call_summary(1, "analysis", analysis_invocation, selected_model))

    try:
        analysis = parse_and_validate_output(raw_output)
    except ResponseValidationError as exc:
        LOGGER.warning(
            "OpenRouter prompt '%s' returned invalid analysis JSON.\nReason: %s\nRaw output:\n%s\nRetrying with JSON repair.",
            prompt_path.name,
            exc,
            format_model_output_for_log(raw_output),
        )
        format_fix_prompt = build_format_fix_prompt(clause, raw_output, str(exc))
        try:
            repair_invocation = invoke_openrouter_with_retry(
                client=openrouter_client,
                model=selected_model,
                max_tokens=token_limit,
                system_prompt=FORMAT_FIX_SYSTEM_PROMPT,
                user_prompt=format_fix_prompt,
                operation="format_fix",
                prompt_version=prompt_path.stem,
            )
        except OpenRouterError as repair_exc:
            raise ProviderRequestError(str(repair_exc)) from repair_exc

        repaired_output = repair_invocation["text"]
        if not repaired_output:
            raise ResponseValidationError("Model response did not contain any text content.")
        _accumulate_usage(usage_totals, repair_invocation["usage"])
        attempt_summaries.append(_build_call_summary(2, "format_fix", repair_invocation, selected_model))

        try:
            analysis = parse_and_validate_output(repaired_output)
        except ResponseValidationError as retry_exc:
            raise ResponseValidationError(
                "Model output could not be converted into the required JSON object after the repair attempt."
            ) from retry_exc

    total_tokens = usage_totals["input_tokens"] + usage_totals["output_tokens"]
    total_latency_ms = round(sum(item["latency_ms"] for item in attempt_summaries), 2)
    pricing_catalog = load_pricing_config(PRICING_PATH)
    cost_info = estimate_model_cost_usd(
        selected_model,
        usage_totals["input_tokens"],
        usage_totals["output_tokens"],
        pricing_catalog,
    )

    return {
        "analysis": analysis,
        "metadata": {
            "provider": "openrouter",
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
            "estimated_cost_usd": cost_info.get("estimated_cost_usd"),
            "pricing_found": cost_info.get("pricing_found", False),
            "attempt_count": len(attempt_summaries),
            "attempts": attempt_summaries,
        },
    }


def _accumulate_usage(usage_totals: dict[str, int], usage: dict[str, int]) -> None:
    usage_totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
    usage_totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)


def _build_call_summary(call_index: int, call_type: str, invocation: dict[str, Any], model: str) -> dict[str, Any]:
    """Normalize OpenRouter telemetry into the API metadata shape."""
    return {
        "attempt": call_index,
        "type": call_type,
        "model": model,
        "prompt_version": invocation["provider_attempts"][-1].get("prompt_version"),
        "latency_ms": invocation["latency_ms"],
        "tokens": invocation["usage"],
        "provider_attempt_count": invocation["provider_attempt_count"],
        "provider_attempts": invocation["provider_attempts"],
        "stop_reason": invocation["provider_attempts"][-1].get("stop_reason"),
    }
