"""Request and response models for the ClauseGuard API."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class AnalyzeRequest(BaseModel):
    """Incoming clause analysis payload."""

    clause_text: str = Field(..., min_length=1, description="Contract clause to analyze.")
    prompt: str = Field(default="v1", description="Named prompt version or path.")
    model: str | None = Field(default=None, description="Optional OpenRouter model override.")


class AnalysisResult(BaseModel):
    """Validated pipeline analysis fields."""

    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    explanation: str
    suggested_redline: str


class AnalyzeResponse(BaseModel):
    """Successful analysis response with telemetry and disclaimer."""

    analysis: AnalysisResult
    metadata: dict[str, Any]
    disclaimer: str


class RefusalResponse(BaseModel):
    """Structured refusal when safety rails block a request."""

    refused: Literal[True] = True
    reason: str
    disclaimer: str


class HealthResponse(BaseModel):
    """Liveness payload for compose and free hosts."""

    status: Literal["ok"] = "ok"
    service: str = "clauseguard-api"
