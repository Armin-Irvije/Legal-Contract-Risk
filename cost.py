from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_pricing_config(pricing_path: Path) -> dict[str, Any]:
    """Load model pricing config from JSON or return an empty config when absent."""
    if not pricing_path.is_file():
        return {"models": {}}

    raw_config = json.loads(pricing_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("pricing.json must contain a JSON object at the top level.")

    models = raw_config.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("pricing.json must contain a 'models' object.")

    return raw_config


def estimate_cost_usd(input_tokens: int, output_tokens: int, pricing_config: dict[str, Any]) -> float:
    """Estimate request cost from token counts and a single model pricing block."""
    input_rate = float(pricing_config["input_per_million_tokens_usd"])
    output_rate = float(pricing_config["output_per_million_tokens_usd"])
    estimated_cost = ((input_tokens / 1_000_000) * input_rate) + ((output_tokens / 1_000_000) * output_rate)
    return round(estimated_cost, 6)


def estimate_model_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing_catalog: dict[str, Any],
) -> dict[str, float | bool | str | None]:
    """Estimate request cost by resolving a model against the pricing catalog."""
    model_pricing = resolve_model_pricing(model, pricing_catalog)
    if model_pricing is None:
        return {"estimated_cost_usd": 0.0, "pricing_found": False}

    return {
        "estimated_cost_usd": estimate_cost_usd(input_tokens, output_tokens, model_pricing),
        "pricing_found": True,
        "pricing_model_key": resolve_model_key(model, pricing_catalog),
    }


def resolve_model_pricing(model: str, pricing_catalog: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a model identifier to the closest matching pricing block."""
    resolved_key = resolve_model_key(model, pricing_catalog)
    if resolved_key is None:
        return None
    model_pricing = pricing_catalog.get("models", {}).get(resolved_key)
    return model_pricing if isinstance(model_pricing, dict) else None


def resolve_model_key(model: str, pricing_catalog: dict[str, Any]) -> str | None:
    """Resolve a model identifier to an exact or normalized pricing key."""
    models = pricing_catalog.get("models", {})
    if not isinstance(models, dict):
        return None

    candidate_keys = [model, normalize_model_id(model)]
    for candidate_key in candidate_keys:
        if candidate_key in models:
            return candidate_key
    return None


def normalize_model_id(model: str) -> str:
    """Normalize versioned model identifiers to a stable pricing family key."""
    return re.sub(r"-\d{8}$", "", model.strip())
