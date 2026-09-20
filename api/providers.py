"""OpenRouter chat-completions client with retry and usage telemetry."""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

LOGGER = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_SECONDS = 1.0
DEFAULT_MAX_DELAY_SECONDS = 8.0

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - depends on local environment.
    OpenAI = Any  # type: ignore[assignment,misc]
    OPENAI_IMPORT_ERROR = exc
else:
    OPENAI_IMPORT_ERROR = None


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter requests fail after retries."""


def create_openrouter_client(api_key: str | None = None) -> Any:
    """Build an OpenAI-compatible client pointed at OpenRouter."""
    if OPENAI_IMPORT_ERROR is not None:
        raise OpenRouterError(
            "The openai package is not installed. Install dependencies from requirements.txt."
        ) from OPENAI_IMPORT_ERROR

    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError("Missing OPENROUTER_API_KEY. Set it before calling the ClauseGuard API.")

    return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)


def invoke_openrouter_with_retry(
    *,
    client: Any,
    model: str,
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
    operation: str,
    prompt_version: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS,
) -> dict[str, Any]:
    """Call OpenRouter chat completions with shared telemetry and transient retries."""
    provider_attempts: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    last_error: Exception | None = None

    for provider_attempt_number in range(1, max_retries + 2):
        attempt_started_at = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            attempt_latency_ms = round((time.perf_counter() - attempt_started_at) * 1000, 2)
            usage = extract_usage(response)
            provider_attempts.append(
                {
                    "attempt": provider_attempt_number,
                    "operation": operation,
                    "status": "success",
                    "model": model,
                    "prompt_version": prompt_version,
                    "latency_ms": attempt_latency_ms,
                    "stop_reason": getattr(response.choices[0], "finish_reason", None) if response.choices else None,
                    "tokens": usage,
                }
            )
            return {
                "response": response,
                "text": extract_text(response),
                "usage": usage,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "provider_attempt_count": len(provider_attempts),
                "provider_attempts": provider_attempts,
            }
        except Exception as exc:  # pragma: no cover - depends on remote API behavior.
            last_error = exc
            attempt_latency_ms = round((time.perf_counter() - attempt_started_at) * 1000, 2)
            status_code = extract_status_code(exc)
            attempt_entry = {
                "attempt": provider_attempt_number,
                "operation": operation,
                "status": "error",
                "model": model,
                "prompt_version": prompt_version,
                "latency_ms": attempt_latency_ms,
                "status_code": status_code,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

            if is_retryable(exc) and provider_attempt_number <= max_retries:
                delay_seconds = min(
                    base_delay_seconds * (2 ** (provider_attempt_number - 1))
                    + random.uniform(0.0, min(base_delay_seconds, 1.0)),
                    max_delay_seconds,
                )
                attempt_entry["retry_delay_ms"] = round(delay_seconds * 1000, 2)
                provider_attempts.append(attempt_entry)
                LOGGER.warning(
                    "Retrying OpenRouter %s for model '%s' after %s (attempt %s/%s).",
                    operation,
                    model,
                    status_code or type(exc).__name__,
                    provider_attempt_number,
                    max_retries + 1,
                )
                time.sleep(delay_seconds)
                continue

            provider_attempts.append(attempt_entry)
            break

    raise OpenRouterError(
        f"{operation} request failed after {len(provider_attempts)} provider attempt(s): {last_error}"
    ) from last_error


def extract_text(response: Any) -> str:
    """Extract assistant text from an OpenAI-compatible chat completion."""
    if not getattr(response, "choices", None):
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", None) or ""
    return content.strip() if isinstance(content, str) else ""


def extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage from an OpenAI-compatible response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def extract_status_code(error: Exception) -> int | None:
    """Extract HTTP status when the OpenAI SDK surfaces one."""
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def is_retryable(error: Exception) -> bool:
    """Return whether an OpenRouter error looks safe to retry."""
    status_code = extract_status_code(error)
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    return type(error).__name__ in {"RateLimitError", "APIConnectionError", "InternalServerError"}
