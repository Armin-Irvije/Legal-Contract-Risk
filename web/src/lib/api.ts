/** Client helpers for the ClauseGuard analyze endpoint. */

import type { AnalyzeOutcome, AnalyzeRefusal, AnalyzeSuccess } from "./types";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

/** Resolve the API base URL from env, falling back to local Docker/uvicorn. */
export function getApiBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_URL?.trim();
  return (configured || DEFAULT_API_BASE).replace(/\/$/, "");
}

/** POST one clause to /analyze and normalize success, refusal, or transport errors. */
export async function analyzeClause(clauseText: string, prompt = "v1"): Promise<AnalyzeOutcome> {
  const url = `${getApiBaseUrl()}/analyze`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clause_text: clauseText, prompt }),
    });
  } catch {
    return {
      kind: "error",
      message: `Could not reach ClauseGuard API at ${getApiBaseUrl()}. Is the API running?`,
    };
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return { kind: "error", message: `API returned non-JSON (HTTP ${response.status}).` };
  }

  if (response.status === 400 && isRefusal(body)) {
    return { kind: "refusal", data: body };
  }

  if (!response.ok) {
    const detail = extractDetail(body);
    return { kind: "error", message: detail || `Analyze failed (HTTP ${response.status}).` };
  }

  if (!isSuccess(body)) {
    return { kind: "error", message: "API response was missing analysis fields." };
  }

  return { kind: "success", data: body };
}

function isRefusal(value: unknown): value is AnalyzeRefusal {
  return (
    typeof value === "object" &&
    value !== null &&
    "refused" in value &&
    (value as AnalyzeRefusal).refused === true &&
    typeof (value as AnalyzeRefusal).reason === "string"
  );
}

function isSuccess(value: unknown): value is AnalyzeSuccess {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as AnalyzeSuccess;
  return (
    typeof candidate.analysis === "object" &&
    candidate.analysis !== null &&
    typeof candidate.analysis.risk_level === "string" &&
    typeof candidate.analysis.explanation === "string" &&
    typeof candidate.analysis.suggested_redline === "string"
  );
}

function extractDetail(body: unknown): string | null {
  if (typeof body !== "object" || body === null) return null;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  return null;
}
