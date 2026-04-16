from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Sequence

from cost import estimate_model_cost_usd
from cost import load_pricing_config
from env_utils import load_env_file
from env_utils import load_project_env
from env_utils import normalize_api_key_env
from judge import DEFAULT_MAX_TOKENS as DEFAULT_JUDGE_MAX_TOKENS
from judge import DEFAULT_MODEL as DEFAULT_JUDGE_MODEL
from judge import JudgeError
from judge import score_output
from pipeline import DEFAULT_MAX_TOKENS
from pipeline import DEFAULT_MODEL
from pipeline import PipelineError
from pipeline import analyze_clause
from pipeline import resolve_prompt_path
from reporting import write_judge_report

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = load_project_env(__file__)
DEFAULT_SMOKE_SET = PROJECT_ROOT / "data" / "smoke_set.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_RUN_LOG = PROJECT_ROOT / "runs.jsonl"
DEFAULT_PRICING_FILE = PROJECT_ROOT / "pricing.json"
LOGGER = logging.getLogger(__name__)
NOISY_LOGGERS = ("httpx", "httpcore")


def build_parser() -> argparse.ArgumentParser:
    """Build the eval CLI argument parser."""
    parser = argparse.ArgumentParser(description="Legal Clause Risk Evaluator CLI")
    parser.add_argument(
        "--prompt",
        nargs="+",
        default=["v1"],
        help="One or more prompt IDs (v1/v2/v3) or prompt template paths.",
    )
    parser.add_argument(
        "--pipeline-model",
        "--model",
        dest="pipeline_model",
        default=os.getenv("PIPELINE_MODEL", DEFAULT_MODEL),
        help="Anthropic model to use for clause analysis.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        help="Anthropic model to use for output scoring.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RUN_LOG),
        help="Append evaluation run records to this JSONL file.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum output tokens for the pipeline response.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_TOKENS,
        help="Maximum output tokens for the judge response.",
    )
    parser.add_argument("--clause-text", help="Analyze a single clause directly from the command line.")
    parser.add_argument(
        "--clauses",
        default=str(DEFAULT_SMOKE_SET),
        help="Path to a clause dataset JSON file. Defaults to the synthetic smoke set.",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Optional .env file to load before running.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the readable judge report JSON will be written.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def configure_logging(log_level_name: str) -> None:
    """Configure concise CLI logging and silence noisy HTTP logs."""
    log_level = getattr(logging, log_level_name.upper())
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")

    if log_level > logging.DEBUG:
        for logger_name in NOISY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None) -> int:
    """Run single-clause analysis or one-or-more dataset evaluation runs."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_logging(args.log_level)
    load_env_file(Path(args.env_file))
    normalize_api_key_env()

    if args.clause_text and len(args.prompt) > 1:
        parser.error("--clause-text supports exactly one --prompt value.")

    if args.clause_text:
        return run_single_clause(args)

    try:
        pricing_config = load_pricing_config(DEFAULT_PRICING_FILE)
        run_results = run_prompt_evaluations(args, pricing_config)
    except (JudgeError, PipelineError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    print(build_run_success_message(run_results))

    return 0


def run_single_clause(args: argparse.Namespace) -> int:
    """Analyze one clause without invoking the judge harness."""
    try:
        result = analyze_clause(
            clause_text=args.clause_text,
            prompt=args.prompt[0],
            model=args.pipeline_model,
            max_tokens=args.max_tokens,
        )
    except (PipelineError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    print("Clause analysis completed successfully.")
    return 0


def run_prompt_evaluations(args: argparse.Namespace, pricing_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute one evaluation run per prompt and persist the run artifacts."""
    output_path = Path(args.output)
    report_dir = Path(args.output_dir)
    run_results: list[dict[str, Any]] = []

    for prompt_value in args.prompt:
        LOGGER.info("Running evaluation for prompt '%s'", prompt_value)
        run_result = evaluate_dataset(
            clauses_path=Path(args.clauses),
            prompt=prompt_value,
            pipeline_model=args.pipeline_model,
            judge_model=args.judge_model,
            pipeline_max_tokens=args.max_tokens,
            judge_max_tokens=args.judge_max_tokens,
            pricing_config=pricing_config,
        )
        report_path = write_judge_report(run_result, report_dir)
        run_result["artifacts"] = {
            "run_log_path": str(output_path.resolve()),
            "judge_report_path": str(report_path.resolve()),
        }
        append_run_log(run_result, output_path)
        LOGGER.info("Saved run log entry to %s", output_path)
        LOGGER.info("Saved judge report to %s", report_path)
        run_results.append(run_result)

    return run_results


def evaluate_dataset(
    clauses_path: Path,
    prompt: str,
    pipeline_model: str,
    judge_model: str,
    pipeline_max_tokens: int,
    judge_max_tokens: int,
    pricing_config: dict[str, Any],
) -> dict[str, Any]:
    """Run the pipeline and judge across a dataset for one prompt variant."""
    dataset = load_dataset(clauses_path)
    prompt_metadata = build_prompt_metadata(prompt)
    records: list[dict[str, Any]] = []
    total_records = len(dataset)

    for index, record in enumerate(dataset, start=1):
        clause_text = record["clause_text"]
        ground_truth_risk = record["ground_truth_risk"]
        notes = record.get("notes")
        pipeline_result: dict[str, Any] | None = None

        LOGGER.info(
            "[%s/%s][%s] Evaluating clause: %s",
            index,
            total_records,
            prompt_metadata["prompt_id"],
            summarize_clause_text(clause_text),
        )
        try:
            pipeline_result = analyze_clause(
                clause_text=clause_text,
                prompt=prompt,
                model=pipeline_model,
                max_tokens=pipeline_max_tokens,
            )
            judge_result = score_output(
                clause_text=clause_text,
                pipeline_output=pipeline_result,
                ground_truth_label=ground_truth_risk,
                model=judge_model,
                max_tokens=judge_max_tokens,
                prompt_version=prompt_metadata["prompt_id"],
            )
            record_metrics = build_record_metrics(pipeline_result, judge_result, pipeline_model, judge_model, pricing_config)
            LOGGER.info(
                "[%s/%s][%s] Completed | predicted=%s | expected=%s | judge=%s | total_latency_ms=%s | total_tokens=%s",
                index,
                total_records,
                prompt_metadata["prompt_id"],
                pipeline_result["analysis"]["risk_level"],
                ground_truth_risk,
                judge_result["scores"]["label_accuracy"],
                record_metrics["total_latency_ms"],
                record_metrics["total_tokens"],
            )
            records.append(
                {
                    "record_index": index,
                    "status": "completed",
                    "prompt_id": prompt_metadata["prompt_id"],
                    "prompt_hash": prompt_metadata["prompt_hash"],
                    "clause_text": clause_text,
                    "ground_truth_risk": ground_truth_risk,
                    "notes": notes,
                    "pipeline": pipeline_result,
                    "judge": judge_result,
                    "metrics": record_metrics,
                    "error": None,
                }
            )
        except (JudgeError, PipelineError, ValueError) as exc:
            failure_stage = "judge" if pipeline_result is not None else "pipeline"
            LOGGER.error(
                "[%s/%s][%s] Failed during %s | expected=%s | error=%s",
                index,
                total_records,
                prompt_metadata["prompt_id"],
                failure_stage,
                ground_truth_risk,
                exc,
            )
            records.append(
                build_failed_record(
                    record_index=index,
                    prompt_metadata=prompt_metadata,
                    clause_text=clause_text,
                    ground_truth_risk=ground_truth_risk,
                    notes=notes,
                    failure_stage=failure_stage,
                    error=exc,
                    pipeline_result=pipeline_result,
                    pipeline_model=pipeline_model,
                    judge_model=judge_model,
                    pricing_config=pricing_config,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive safeguard.
            failure_stage = "judge" if pipeline_result is not None else "pipeline"
            LOGGER.exception(
                "[%s/%s][%s] Unexpected failure during %s.",
                index,
                total_records,
                prompt_metadata["prompt_id"],
                failure_stage,
            )
            records.append(
                build_failed_record(
                    record_index=index,
                    prompt_metadata=prompt_metadata,
                    clause_text=clause_text,
                    ground_truth_risk=ground_truth_risk,
                    notes=notes,
                    failure_stage=failure_stage,
                    error=exc,
                    pipeline_result=pipeline_result,
                    pipeline_model=pipeline_model,
                    judge_model=judge_model,
                    pricing_config=pricing_config,
                )
            )

    aggregates = compute_aggregates(records)
    return {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(clauses_path.resolve()),
        "prompt": prompt,
        "prompt_id": prompt_metadata["prompt_id"],
        "prompt_path": prompt_metadata["prompt_path"],
        "prompt_hash": prompt_metadata["prompt_hash"],
        "pipeline_model": pipeline_model,
        "judge_model": judge_model,
        "record_count": len(records),
        "aggregates": aggregates,
        "results": records,
    }


def build_prompt_metadata(prompt: str) -> dict[str, str]:
    """Resolve a prompt input into a stable identifier and content hash."""
    prompt_path = resolve_prompt_path(prompt)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return {
        "prompt_id": prompt_path.stem,
        "prompt_path": str(prompt_path),
        "prompt_hash": prompt_hash,
    }


def build_record_metrics(
    pipeline_result: dict[str, Any],
    judge_result: dict[str, Any],
    pipeline_model: str,
    judge_model: str,
    pricing_config: dict[str, Any],
) -> dict[str, Any]:
    """Collect per-clause metrics for auditing and aggregate rollups."""
    pipeline_tokens = pipeline_result["metadata"]["tokens"]
    judge_tokens = judge_result["metadata"]["tokens"]
    pipeline_cost = estimate_model_cost_usd(
        pipeline_model,
        pipeline_tokens["input_tokens"],
        pipeline_tokens["output_tokens"],
        pricing_config,
    )
    judge_cost = estimate_model_cost_usd(
        judge_model,
        judge_tokens["input_tokens"],
        judge_tokens["output_tokens"],
        pricing_config,
    )
    pipeline_latency_ms = float(pipeline_result["metadata"]["latency_ms"])
    judge_latency_ms = float(judge_result["metadata"]["latency_ms"])
    total_latency_ms = round(pipeline_latency_ms + judge_latency_ms, 2)
    total_tokens = int(pipeline_tokens["total_tokens"]) + int(judge_tokens["total_tokens"])
    total_cost_usd = round(
        float(pipeline_cost["estimated_cost_usd"]) + float(judge_cost["estimated_cost_usd"]),
        6,
    )

    return {
        "status": "completed",
        "label_correct": judge_result["scores"]["label_accuracy"] == "pass",
        "pipeline_latency_ms": pipeline_latency_ms,
        "judge_latency_ms": judge_latency_ms,
        "total_latency_ms": total_latency_ms,
        "pipeline_tokens": int(pipeline_tokens["total_tokens"]),
        "judge_tokens": int(judge_tokens["total_tokens"]),
        "total_tokens": total_tokens,
        "pipeline_cost_usd": float(pipeline_cost["estimated_cost_usd"]),
        "judge_cost_usd": float(judge_cost["estimated_cost_usd"]),
        "total_cost_usd": total_cost_usd,
        "pipeline_pricing_found": bool(pipeline_cost["pricing_found"]),
        "judge_pricing_found": bool(judge_cost["pricing_found"]),
        "cost_estimate_complete": bool(pipeline_cost["pricing_found"]) and bool(judge_cost["pricing_found"]),
    }


def build_failed_record(
    record_index: int,
    prompt_metadata: dict[str, str],
    clause_text: str,
    ground_truth_risk: str,
    notes: str | None,
    failure_stage: str,
    error: Exception,
    pipeline_result: dict[str, Any] | None,
    pipeline_model: str,
    judge_model: str,
    pricing_config: dict[str, Any],
) -> dict[str, Any]:
    """Build a failure record without aborting the full evaluation run."""
    return {
        "record_index": record_index,
        "status": "failed",
        "prompt_id": prompt_metadata["prompt_id"],
        "prompt_hash": prompt_metadata["prompt_hash"],
        "clause_text": clause_text,
        "ground_truth_risk": ground_truth_risk,
        "notes": notes,
        "pipeline": pipeline_result,
        "judge": None,
        "metrics": build_failed_record_metrics(pipeline_result, pipeline_model, pricing_config, failure_stage),
        "error": build_error_details(error, failure_stage, judge_model),
    }


def build_failed_record_metrics(
    pipeline_result: dict[str, Any] | None,
    pipeline_model: str,
    pricing_config: dict[str, Any],
    failure_stage: str,
) -> dict[str, Any]:
    """Build partial metrics for failed clauses so totals remain auditable."""
    pipeline_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    pipeline_latency_ms = 0.0
    pipeline_cost = {"estimated_cost_usd": 0.0, "pricing_found": False}

    if pipeline_result is not None:
        pipeline_tokens = pipeline_result["metadata"]["tokens"]
        pipeline_latency_ms = float(pipeline_result["metadata"]["latency_ms"])
        pipeline_cost = estimate_model_cost_usd(
            pipeline_model,
            pipeline_tokens["input_tokens"],
            pipeline_tokens["output_tokens"],
            pricing_config,
        )

    return {
        "status": "failed",
        "label_correct": False,
        "failure_stage": failure_stage,
        "pipeline_latency_ms": pipeline_latency_ms,
        "judge_latency_ms": 0.0,
        "total_latency_ms": round(pipeline_latency_ms, 2),
        "pipeline_tokens": int(pipeline_tokens["total_tokens"]),
        "judge_tokens": 0,
        "total_tokens": int(pipeline_tokens["total_tokens"]),
        "pipeline_cost_usd": float(pipeline_cost["estimated_cost_usd"]),
        "judge_cost_usd": 0.0,
        "total_cost_usd": float(pipeline_cost["estimated_cost_usd"]),
        "pipeline_pricing_found": bool(pipeline_cost["pricing_found"]),
        "judge_pricing_found": False,
        "cost_estimate_complete": False,
    }


def build_error_details(error: Exception, failure_stage: str, judge_model: str) -> dict[str, Any]:
    """Project exception details into a structured per-clause failure record."""
    return {
        "stage": failure_stage,
        "type": type(error).__name__,
        "message": str(error),
        "judge_model": judge_model,
    }


def append_run_log(run_result: dict[str, Any], output_path: Path) -> Path:
    """Append one evaluation run to an auditable JSONL ledger."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_log_entry = build_run_log_entry(run_result)
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(run_log_entry, ensure_ascii=True))
        output_file.write("\n")
    return output_path


def build_run_log_entry(run_result: dict[str, Any]) -> dict[str, Any]:
    """Project a detailed run into a compact run-summary JSONL entry."""
    return {
        "run_timestamp": run_result["run_timestamp"],
        "dataset_path": run_result["dataset_path"],
        "prompt": run_result["prompt"],
        "prompt_id": run_result["prompt_id"],
        "prompt_path": run_result["prompt_path"],
        "prompt_hash": run_result["prompt_hash"],
        "pipeline_model": run_result["pipeline_model"],
        "judge_model": run_result["judge_model"],
        "record_count": run_result["record_count"],
        "aggregates": run_result["aggregates"],
        "artifacts": run_result.get("artifacts", {}),
    }


def summarize_clause_text(clause_text: str, max_chars: int = 100) -> str:
    """Return a single-line preview for progress logging."""
    compact_text = " ".join(clause_text.split())
    if len(compact_text) <= max_chars:
        return compact_text
    return textwrap.shorten(compact_text, width=max_chars, placeholder="...")


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """Load and validate the clause dataset JSON file."""
    if not dataset_path.is_file():
        raise ValueError(f"Dataset file was not found: {dataset_path}")

    try:
        raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset file is not valid JSON: {exc}") from exc

    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError("Dataset file must contain a non-empty JSON array.")

    validated_records: list[dict[str, Any]] = []
    for index, item in enumerate(raw_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset record {index} must be a JSON object.")

        clause_text = item.get("clause_text")
        ground_truth_risk = item.get("ground_truth_risk")
        notes = item.get("notes")

        if not isinstance(clause_text, str) or not clause_text.strip():
            raise ValueError(f"Dataset record {index} is missing a non-empty 'clause_text'.")
        if not isinstance(ground_truth_risk, str) or not ground_truth_risk.strip():
            raise ValueError(f"Dataset record {index} is missing a non-empty 'ground_truth_risk'.")
        if notes is not None and not isinstance(notes, str):
            raise ValueError(f"Dataset record {index} has a non-string 'notes' value.")

        validated_records.append(
            {
                "clause_text": clause_text.strip(),
                "ground_truth_risk": ground_truth_risk.strip().upper(),
                "notes": notes.strip() if isinstance(notes, str) else None,
            }
        )

    return validated_records


def compute_aggregates(records: list[dict[str, Any]]) -> dict[str, float | int | bool]:
    """Roll up per-clause metrics into prompt-level summary metrics."""
    if not records:
        return {
            "completed_records": 0,
            "failed_records": 0,
            "completion_rate_pct": 0.0,
            "label_accuracy_pct": 0.0,
            "avg_explanation_quality": 0.0,
            "avg_redline_usefulness": 0.0,
            "avg_latency_ms": 0.0,
            "avg_pipeline_latency_ms": 0.0,
            "avg_judge_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "total_pipeline_tokens": 0,
            "total_judge_tokens": 0,
            "total_tokens": 0,
            "total_pipeline_cost_usd": 0.0,
            "total_judge_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "estimated_cost_per_1k_clause_reviews_usd": 0.0,
            "cost_estimate_complete": False,
        }

    completed_records = [record for record in records if record.get("status") == "completed"]
    failed_records = [record for record in records if record.get("status") == "failed"]
    passed_count = sum(1 for record in completed_records if record["metrics"]["label_correct"])
    explanation_total = sum(record["judge"]["scores"]["explanation_quality"] for record in completed_records)
    redline_total = sum(record["judge"]["scores"]["redline_usefulness"] for record in completed_records)
    pipeline_latency_total = sum(record["metrics"]["pipeline_latency_ms"] for record in records)
    judge_latency_total = sum(record["metrics"]["judge_latency_ms"] for record in records)
    total_latency_ms = sum(record["metrics"]["total_latency_ms"] for record in records)
    pipeline_token_total = sum(record["metrics"]["pipeline_tokens"] for record in records)
    judge_token_total = sum(record["metrics"]["judge_tokens"] for record in records)
    total_token_total = sum(record["metrics"]["total_tokens"] for record in records)
    pipeline_cost_total = sum(record["metrics"]["pipeline_cost_usd"] for record in records)
    judge_cost_total = sum(record["metrics"]["judge_cost_usd"] for record in records)
    total_cost = sum(record["metrics"]["total_cost_usd"] for record in records)
    cost_estimate_complete = all(record["metrics"]["cost_estimate_complete"] for record in records)

    record_count = len(records)
    completed_count = len(completed_records)
    failed_count = len(failed_records)
    return {
        "completed_records": completed_count,
        "failed_records": failed_count,
        "completion_rate_pct": round((completed_count / record_count) * 100, 2),
        "label_accuracy_pct": round((passed_count / completed_count) * 100, 2) if completed_count else 0.0,
        "avg_explanation_quality": round(explanation_total / completed_count, 2) if completed_count else 0.0,
        "avg_redline_usefulness": round(redline_total / completed_count, 2) if completed_count else 0.0,
        "avg_latency_ms": round(total_latency_ms / record_count, 2),
        "avg_pipeline_latency_ms": round(pipeline_latency_total / record_count, 2),
        "avg_judge_latency_ms": round(judge_latency_total / record_count, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "total_pipeline_tokens": pipeline_token_total,
        "total_judge_tokens": judge_token_total,
        "total_tokens": total_token_total,
        "total_pipeline_cost_usd": round(pipeline_cost_total, 6),
        "total_judge_cost_usd": round(judge_cost_total, 6),
        "total_cost_usd": round(total_cost, 6),
        "estimated_cost_per_1k_clause_reviews_usd": round((total_cost / record_count) * 1000, 6),
        "cost_estimate_complete": cost_estimate_complete,
    }


def render_comparison_table(run_results: list[dict[str, Any]]) -> str:
    """Render a compact markdown comparison table across prompt runs."""
    headers = [
        "Prompt",
        "Hash",
        "Completion %",
        "Label Accuracy %",
        "Avg Explanation",
        "Avg Redline",
        "Avg Latency ms",
        "Total Tokens",
        "Total Cost USD",
        "Cost / 1k USD",
    ]
    rows = [headers]

    for run_result in run_results:
        aggregates = run_result["aggregates"]
        rows.append(
            [
                str(run_result["prompt_id"]),
                str(run_result["prompt_hash"])[:8],
                f"{aggregates['completion_rate_pct']:.2f}",
                f"{aggregates['label_accuracy_pct']:.2f}",
                f"{aggregates['avg_explanation_quality']:.2f}",
                f"{aggregates['avg_redline_usefulness']:.2f}",
                f"{aggregates['avg_latency_ms']:.2f}",
                str(aggregates["total_tokens"]),
                format_cost_for_table(aggregates["total_cost_usd"], aggregates["cost_estimate_complete"]),
                format_cost_for_table(
                    aggregates["estimated_cost_per_1k_clause_reviews_usd"],
                    aggregates["cost_estimate_complete"],
                ),
            ]
        )

    table_output = format_markdown_table(rows)
    if any(run_result["aggregates"]["failed_records"] > 0 for run_result in run_results):
        table_output = f"{table_output}\n\n* Completion below 100% means at least one clause failed after retries and was recorded in the report."
    if any(not run_result["aggregates"]["cost_estimate_complete"] for run_result in run_results):
        table_output = f"{table_output}\n\n* Cost is partial because pricing was missing for at least one model."
    return table_output


def build_run_success_message(run_results: list[dict[str, Any]]) -> str:
    """Build a concise terminal success message for completed evaluation runs."""
    if not run_results:
        return "Evaluation completed with no runs to report."

    if len(run_results) == 1:
        run_result = run_results[0]
        aggregates = run_result["aggregates"]
        return (
            f"Evaluation completed successfully for prompt '{run_result['prompt_id']}'. "
            f"Completed clauses: {aggregates['completed_records']}/{run_result['record_count']}. "
            f"Judge report: {run_result['artifacts']['judge_report_path']}. "
            f"Run log: {run_result['artifacts']['run_log_path']}."
        )

    prompt_ids = ", ".join(run_result["prompt_id"] for run_result in run_results)
    latest_run = run_results[-1]
    return (
        f"Evaluation completed successfully for prompts: {prompt_ids}. "
        f"Run summaries appended to {latest_run['artifacts']['run_log_path']}. "
        f"Judge reports were written to {Path(latest_run['artifacts']['judge_report_path']).parent}."
    )


def format_cost_for_table(total_cost_usd: float | int, cost_estimate_complete: bool) -> str:
    """Format cost totals for comparison output."""
    if not cost_estimate_complete:
        return f"{float(total_cost_usd):.6f}*"
    return f"{float(total_cost_usd):.6f}"


def format_markdown_table(rows: list[list[str]]) -> str:
    """Render rows as a simple markdown table for CLI output."""
    header = rows[0]
    divider = ["---"] * len(header)
    rendered_rows = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for row in rows[1:]:
        rendered_rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rendered_rows)


if __name__ == "__main__":
    raise SystemExit(main())
