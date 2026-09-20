"""FastAPI entrypoint for ClauseGuard clause analysis."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from api.safety import DISCLAIMER
from api.safety import check_clause_request
from api.schemas import AnalysisResult
from api.schemas import AnalyzeRequest
from api.schemas import AnalyzeResponse
from api.schemas import HealthResponse
from api.schemas import RefusalResponse
from api.service import analyze_clause_openrouter
from pipeline import PromptTemplateError
from pipeline import ProviderRequestError
from pipeline import ResponseValidationError

LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="ClauseGuard API",
    description="Legal clause risk analysis demo. Not legal advice.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a simple liveness payload."""
    return HealthResponse()


@app.post("/analyze", response_model=AnalyzeResponse, responses={400: {"model": RefusalResponse}})
def analyze(request: AnalyzeRequest) -> AnalyzeResponse | JSONResponse:
    """Analyze a clause via OpenRouter, or return a structured safety refusal."""
    refusal_reason = check_clause_request(request.clause_text)
    if refusal_reason:
        body = RefusalResponse(reason=refusal_reason, disclaimer=DISCLAIMER)
        return JSONResponse(status_code=400, content=body.model_dump())

    try:
        result = analyze_clause_openrouter(
            clause_text=request.clause_text,
            prompt=request.prompt,
            model=request.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PromptTemplateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ResponseValidationError as exc:
        LOGGER.exception("Model output failed validation.")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ProviderRequestError as exc:
        LOGGER.exception("OpenRouter provider request failed.")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AnalyzeResponse(
        analysis=AnalysisResult(**result["analysis"]),
        metadata=result["metadata"],
        disclaimer=DISCLAIMER,
    )
