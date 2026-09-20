"use client";

/** Paste-clause analyzer: calls the API and shows risk, redline, and cost telemetry. */

import { useState } from "react";
import type { FormEvent } from "react";

import { analyzeClause } from "@/lib/api";
import { FIXTURES } from "@/lib/fixtures";
import type { AnalyzeOutcome, AnalyzeSuccess, RiskLevel } from "@/lib/types";

import styles from "./Analyzer.module.css";

type ViewState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; outcome: AnalyzeOutcome };

export default function Analyzer() {
  const [clauseText, setClauseText] = useState(FIXTURES[0].clause_text);
  const [view, setView] = useState<ViewState>({ status: "idle" });

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = clauseText.trim();
    if (!trimmed) {
      setView({
        status: "ready",
        outcome: { kind: "error", message: "Paste a clause before analyzing." },
      });
      return;
    }

    setView({ status: "loading" });
    const outcome = await analyzeClause(trimmed);
    setView({ status: "ready", outcome });
  }

  return (
    <div className={styles.shell}>
      <header className={styles.brand}>
        <p className={styles.mark}>ClauseGuard</p>
        <h1 className={styles.headline}>Clause risk, explained</h1>
        <p className={styles.lede}>
          Paste a synthetic contract clause. Get a risk level, plain-language explanation, suggested
          redline, plus tokens, latency, and estimated cost.
        </p>
      </header>

      <form className={styles.form} onSubmit={onSubmit}>
        <div className={styles.fixtures} role="group" aria-label="Sample clauses">
          {FIXTURES.map((fixture) => (
            <button
              key={fixture.id}
              type="button"
              className={styles.fixture}
              onClick={() => setClauseText(fixture.clause_text)}
            >
              {fixture.label}
            </button>
          ))}
        </div>

        <label className={styles.label} htmlFor="clause">
          Clause text
        </label>
        <textarea
          id="clause"
          className={styles.textarea}
          value={clauseText}
          onChange={(event) => setClauseText(event.target.value)}
          rows={8}
          spellCheck={false}
          disabled={view.status === "loading"}
        />

        <div className={styles.actions}>
          <button className={styles.submit} type="submit" disabled={view.status === "loading"}>
            {view.status === "loading" ? "Analyzing…" : "Analyze clause"}
          </button>
          <p className={styles.hint}>Uses OpenRouter via the local ClauseGuard API.</p>
        </div>
      </form>

      <section className={styles.results} aria-live="polite">
        {view.status === "idle" && (
          <p className={styles.placeholder}>Results appear here after you analyze.</p>
        )}
        {view.status === "loading" && <p className={styles.placeholder}>Calling /analyze…</p>}
        {view.status === "ready" && <OutcomeView outcome={view.outcome} />}
      </section>
    </div>
  );
}

/** Render success, refusal, or error outcomes from the API. */
function OutcomeView({ outcome }: { outcome: AnalyzeOutcome }) {
  if (outcome.kind === "error") {
    return (
      <div className={styles.bannerError} role="alert">
        <p className={styles.bannerTitle}>Request failed</p>
        <p>{outcome.message}</p>
      </div>
    );
  }

  if (outcome.kind === "refusal") {
    return (
      <div className={styles.bannerRefuse} role="alert">
        <p className={styles.bannerTitle}>Request refused</p>
        <p>{outcome.data.reason}</p>
        <p className={styles.disclaimer}>{outcome.data.disclaimer}</p>
      </div>
    );
  }

  return <SuccessView data={outcome.data} />;
}

/** Successful analysis: risk badge, explanation, redline, and telemetry. */
function SuccessView({ data }: { data: AnalyzeSuccess }) {
  const { analysis, metadata, disclaimer } = data;
  const cost =
    metadata.estimated_cost_usd == null
      ? "n/a"
      : `$${metadata.estimated_cost_usd.toFixed(6)}`;

  return (
    <div className={styles.success}>
      <div className={styles.riskRow}>
        <span className={`${styles.risk} ${riskClass(analysis.risk_level)}`}>
          {analysis.risk_level}
        </span>
        <span className={styles.metaInline}>
          {metadata.model ?? "model?"} · {formatLatency(metadata.latency_ms)} ·{" "}
          {metadata.tokens?.total_tokens ?? "?"} tokens · {cost}
        </span>
      </div>

      <h2 className={styles.sectionTitle}>Explanation</h2>
      <p className={styles.body}>{analysis.explanation}</p>

      <h2 className={styles.sectionTitle}>Suggested redline</h2>
      <pre className={styles.redline}>{analysis.suggested_redline}</pre>

      <p className={styles.disclaimer}>{disclaimer}</p>
    </div>
  );
}

function riskClass(level: RiskLevel): string {
  if (level === "LOW") return styles.riskLow;
  if (level === "MEDIUM") return styles.riskMedium;
  return styles.riskHigh;
}

function formatLatency(ms: number | undefined): string {
  if (ms == null || Number.isNaN(ms)) return "? ms";
  return `${Math.round(ms)} ms`;
}
