from __future__ import annotations

import logging
import random
import time
from typing import Any

LOGGER = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 529}
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_SECONDS = 1.0
DEFAULT_MAX_DELAY_SECONDS = 8.0


def extract_text_from_response(response: Any) -> str:
    """Extract concatenated text blocks from an Anthropic response."""
    text_parts: list[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            text_parts.append(getattr(block, "text", ""))
    return "".join(text_parts).strip()


def extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage counts from an Anthropic response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def invoke_anthropic_with_retry(
    *,
    client: Any,
    model: str,
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
    operation: str,
    error_cls: type[Exception],
    prompt_version: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS,
) -> dict[str, Any]:
    """Call Anthropic with shared telemetry and retry transient provider failures."""
    provider_attempts: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    last_error: Exception | None = None

    for provider_attempt_number in range(1, max_retries + 2):
        attempt_started_at = time.perf_counter()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
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
                    "stop_reason": getattr(response, "stop_reason", None),
                    "tokens": usage,
                }
            )
            total_latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            return {
                "response": response,
                "usage": usage,
                "latency_ms": total_latency_ms,
                "provider_attempt_count": len(provider_attempts),
                "provider_attempts": provider_attempts,
            }
        except Exception as exc:  # pragma: no cover - depends on remote API behavior.
            last_error = exc
            attempt_latency_ms = round((time.perf_counter() - attempt_started_at) * 1000, 2)
            status_code = extract_status_code(exc)
            retryable = is_retryable_api_error(exc)
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

            if retryable and provider_attempt_number <= max_retries:
                delay_seconds = compute_backoff_delay(
                    retry_index=provider_attempt_number,
                    base_delay_seconds=base_delay_seconds,
                    max_delay_seconds=max_delay_seconds,
                )
                attempt_entry["retry_delay_ms"] = round(delay_seconds * 1000, 2)
                provider_attempts.append(attempt_entry)
                LOGGER.warning(
                    "Retrying %s call for model '%s' after transient provider error %s (attempt %s/%s).",
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

    raise error_cls(
        f"{operation} request failed after {len(provider_attempts)} provider attempt(s): {last_error}"
    ) from last_error


def compute_backoff_delay(retry_index: int, base_delay_seconds: float, max_delay_seconds: float) -> float:
    """Compute exponential backoff with bounded jitter for retry sleeps."""
    exponential_delay = min(base_delay_seconds * (2 ** (retry_index - 1)), max_delay_seconds)
    jitter_seconds = random.uniform(0.0, min(base_delay_seconds, 1.0))
    return min(exponential_delay + jitter_seconds, max_delay_seconds)


def extract_status_code(error: Exception) -> int | None:
    """Extract the HTTP status code when the SDK surfaces one."""
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(error, "response", None)
    response_status_code = getattr(response, "status_code", None)
    if isinstance(response_status_code, int):
        return response_status_code

    return None


def is_retryable_api_error(error: Exception) -> bool:
    """Return whether an Anthropic error looks transient and safe to retry."""
    status_code = extract_status_code(error)
    if status_code in RETRYABLE_STATUS_CODES:
        return True

    return type(error).__name__ in {"RateLimitError", "OverloadedError"}
