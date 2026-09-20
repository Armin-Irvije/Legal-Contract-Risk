/** Types mirroring the ClauseGuard FastAPI analyze contract. */

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

export type AnalysisResult = {
  risk_level: RiskLevel;
  explanation: string;
  suggested_redline: string;
};

export type AnalyzeMetadata = {
  provider?: string;
  model?: string;
  prompt?: string;
  prompt_hash?: string;
  latency_ms?: number;
  tokens?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
  };
  estimated_cost_usd?: number | null;
  pricing_found?: boolean;
};

export type AnalyzeSuccess = {
  analysis: AnalysisResult;
  metadata: AnalyzeMetadata;
  disclaimer: string;
};

export type AnalyzeRefusal = {
  refused: true;
  reason: string;
  disclaimer: string;
};

export type AnalyzeOutcome =
  | { kind: "success"; data: AnalyzeSuccess }
  | { kind: "refusal"; data: AnalyzeRefusal }
  | { kind: "error"; message: string };
