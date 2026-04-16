from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_output_dir(output_dir: Path) -> Path:
    """Create the output directory when it does not already exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_judge_report_filename(dataset_path: str) -> str:
    """Create a stable, readable filename for a judge report."""
    dataset_name = Path(dataset_path).stem or "dataset"
    safe_dataset_name = sanitize_filename_component(dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"judge_report_{safe_dataset_name}_{timestamp}.json"


def sanitize_filename_component(value: str) -> str:
    """Normalize a filename component to ASCII-friendly characters."""
    sanitized = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    collapsed = "_".join(part for part in sanitized.split("_") if part)
    return collapsed or "report"


def build_judge_report(evaluation_results: dict[str, Any]) -> dict[str, Any]:
    """Project evaluation results into a readable judge-focused JSON report."""
    results = evaluation_results.get("results", [])
    report_records: list[dict[str, Any]] = []

    for index, record in enumerate(results, start=1):
        pipeline = record["pipeline"]
        judge = record["judge"]
        report_records.append(
            {
                "record_index": index,
                "clause_text": record["clause_text"],
                "ground_truth_risk": record["ground_truth_risk"],
                "predicted_risk": pipeline["analysis"]["risk_level"],
                "judge_scores": judge["scores"],
                "judge_summary": build_judge_summary(judge["scores"]),
                "pipeline_explanation": pipeline["analysis"]["explanation"],
                "suggested_redline": pipeline["analysis"]["suggested_redline"],
                "judge_metadata": judge["metadata"],
                "pipeline_metadata": pipeline["metadata"],
                "notes": record.get("notes"),
            }
        )

    return {
        "report_type": "judge_evaluation_report",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": evaluation_results["dataset_path"],
        "prompt": evaluation_results["prompt"],
        "pipeline_model": evaluation_results["pipeline_model"],
        "judge_model": evaluation_results["judge_model"],
        "record_count": evaluation_results["record_count"],
        "aggregates": evaluation_results["aggregates"],
        "results": report_records,
    }


def build_judge_summary(scores: dict[str, Any]) -> str:
    """Render a short human-readable summary for a judge score block."""
    label_accuracy = scores["label_accuracy"]
    explanation_quality = scores["explanation_quality"]
    redline_usefulness = scores["redline_usefulness"]
    return (
        f"Label accuracy: {label_accuracy}. "
        f"Explanation quality: {explanation_quality}/5. "
        f"Redline usefulness: {redline_usefulness}/5."
    )


def write_judge_report(evaluation_results: dict[str, Any], output_dir: Path) -> Path:
    """Write a readable judge report JSON file into the chosen output directory."""
    report = build_judge_report(evaluation_results)
    target_dir = ensure_output_dir(output_dir)
    output_path = target_dir / build_judge_report_filename(evaluation_results["dataset_path"])
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path
