from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any
from typing import Sequence

from env_utils import load_env_file
from env_utils import load_project_env
from env_utils import normalize_api_key_env
from judge import DEFAULT_MAX_TOKENS as DEFAULT_JUDGE_MAX_TOKENS
from judge import DEFAULT_MODEL as DEFAULT_JUDGE_MODEL
from judge import JudgeError
from judge import score_output
from pipeline import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, PipelineError, analyze_clause
from reporting import write_judge_report

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = load_project_env(__file__)
DEFAULT_SMOKE_SET = PROJECT_ROOT / "data" / "smoke_set.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
LOGGER = logging.getLogger(__name__)
NOISY_LOGGERS = ("httpx", "httpcore")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legal Clause Risk Evaluator CLI")
    parser.add_argument("--prompt", default="v1", help="Named prompt (v1/v2/v3) or path to a prompt template file.")
    parser.add_argument("--model", default=os.getenv("PIPELINE_MODEL", DEFAULT_MODEL), help="Anthropic model to use for clause analysis.")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL), help="Anthropic model to use for output scoring.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum output tokens for the pipeline response.")
    parser.add_argument("--judge-max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS, help="Maximum output tokens for the judge response.")
    parser.add_argument("--clause-text", help="Analyze a single clause directly from the command line.")
    parser.add_argument("--clauses", default=str(DEFAULT_SMOKE_SET), help="Path to a clause dataset JSON file. Defaults to the synthetic smoke set.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Optional .env file to load before running.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where the readable judge report JSON will be written.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="Logging verbosity.")
    return parser


def configure_logging(log_level_name: str) -> None:
    """Configure concise CLI logging and silence noisy HTTP logs."""
    log_level = getattr(logging, log_level_name.upper())
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")

    if log_level > logging.DEBUG:
        for logger_name in NOISY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_logging(args.log_level)
    load_env_file(Path(args.env_file))
    normalize_api_key_env()

    if args.clause_text:
        try:
            result = analyze_clause(clause_text=args.clause_text, prompt=args.prompt, model=args.model, max_tokens=args.max_tokens)
        except (PipelineError, ValueError) as exc:
            LOGGER.error("%s", exc)
            return 1

        print(json.dumps(result, indent=2))
        return 0

    try:
        results = evaluate_dataset(
            clauses_path=Path(args.clauses),
            prompt=args.prompt,
            pipeline_model=args.model,
            judge_model=args.judge_model,
            pipeline_max_tokens=args.max_tokens,
            judge_max_tokens=args.judge_max_tokens,
        )
    except (JudgeError, PipelineError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    report_path = write_judge_report(results, Path(args.output_dir))
    LOGGER.info("Saved judge report to %s", report_path)
    print(json.dumps(results, indent=2))
    return 0


def evaluate_dataset(
    clauses_path: Path,
    prompt: str,
    pipeline_model: str,
    judge_model: str,
    pipeline_max_tokens: int,
    judge_max_tokens: int,
) -> dict[str, Any]:
    dataset = load_dataset(clauses_path)
    records: list[dict[str, Any]] = []
    total_records = len(dataset)

    for index, record in enumerate(dataset, start=1):
        clause_text = record["clause_text"]
        ground_truth_risk = record["ground_truth_risk"]
        notes = record.get("notes")

        LOGGER.info("[%s/%s] Evaluating clause: %s", index, total_records, summarize_clause_text(clause_text))
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
        )
        LOGGER.info(
            "[%s/%s] Completed | predicted=%s | expected=%s | judge=%s | pipeline_attempts=%s | judge_attempts=%s",
            index,
            total_records,
            pipeline_result["analysis"]["risk_level"],
            ground_truth_risk,
            judge_result["scores"]["label_accuracy"],
            pipeline_result["metadata"]["attempt_count"],
            judge_result["metadata"]["attempt_count"],
        )

        records.append(
            {
                "clause_text": clause_text,
                "ground_truth_risk": ground_truth_risk,
                "notes": notes,
                "pipeline": pipeline_result,
                "judge": judge_result,
            }
        )

    return {
        "dataset_path": str(clauses_path.resolve()),
        "prompt": prompt,
        "pipeline_model": pipeline_model,
        "judge_model": judge_model,
        "record_count": len(records),
        "aggregates": compute_aggregates(records),
        "results": records,
    }


def summarize_clause_text(clause_text: str, max_chars: int = 100) -> str:
    """Return a single-line preview for progress logging."""
    compact_text = " ".join(clause_text.split())
    if len(compact_text) <= max_chars:
        return compact_text
    return textwrap.shorten(compact_text, width=max_chars, placeholder="...")


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
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


def compute_aggregates(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {
            "label_accuracy_pct": 0.0,
            "avg_explanation_quality": 0.0,
            "avg_redline_usefulness": 0.0,
            "avg_pipeline_latency_ms": 0.0,
            "avg_judge_latency_ms": 0.0,
            "total_pipeline_tokens": 0,
            "total_judge_tokens": 0,
        }

    passed_count = sum(1 for record in records if record["judge"]["scores"]["label_accuracy"] == "pass")
    explanation_total = sum(record["judge"]["scores"]["explanation_quality"] for record in records)
    redline_total = sum(record["judge"]["scores"]["redline_usefulness"] for record in records)
    pipeline_latency_total = sum(record["pipeline"]["metadata"]["latency_ms"] for record in records)
    judge_latency_total = sum(record["judge"]["metadata"]["latency_ms"] for record in records)
    pipeline_token_total = sum(record["pipeline"]["metadata"]["tokens"]["total_tokens"] for record in records)
    judge_token_total = sum(record["judge"]["metadata"]["tokens"]["total_tokens"] for record in records)

    record_count = len(records)
    return {
        "label_accuracy_pct": round((passed_count / record_count) * 100, 2),
        "avg_explanation_quality": round(explanation_total / record_count, 2),
        "avg_redline_usefulness": round(redline_total / record_count, 2),
        "avg_pipeline_latency_ms": round(pipeline_latency_total / record_count, 2),
        "avg_judge_latency_ms": round(judge_latency_total / record_count, 2),
        "total_pipeline_tokens": pipeline_token_total,
        "total_judge_tokens": judge_token_total,
    }


if __name__ == "__main__":
    raise SystemExit(main())
