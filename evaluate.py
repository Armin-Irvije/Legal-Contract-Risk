from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Sequence

from pipeline import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, PipelineError, analyze_clause

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legal Clause Risk Evaluator CLI")
    parser.add_argument("--prompt", default="v1", help="Named prompt (v1/v2/v3) or path to a prompt template file.")
    parser.add_argument("--model", default=os.getenv("PIPELINE_MODEL", DEFAULT_MODEL), help="Anthropic model to use for analysis.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum output tokens for the model response.")
    parser.add_argument("--clause-text", help="Analyze a single clause directly from the command line.")
    parser.add_argument("--clauses", help="Reserved for Phase 2 dataset evaluation.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Optional .env file to load before running.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="Logging verbosity.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")
    load_env_file(Path(args.env_file))
    normalize_api_key_env()

    if args.clause_text:
        try:
            result = analyze_clause(clause_text=args.clause_text, prompt=args.prompt, model=args.model, max_tokens=args.max_tokens)
        except (PipelineError, ValueError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 1

        print(json.dumps(result, indent=2))
        return 0

    if args.clauses:
        parser.error("Dataset evaluation is planned for Phase 2. For Phase 1, use --clause-text for the smoke runner.")

    parser.error("Provide --clause-text to analyze a single clause.")
    return 2


def load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def normalize_api_key_env() -> None:
    if os.getenv("ANTHROPIC_API_KEY"):
        return

    claude_api_key = os.getenv("CLAUDE_API_KEY")
    if claude_api_key:
        os.environ["ANTHROPIC_API_KEY"] = claude_api_key


if __name__ == "__main__":
    raise SystemExit(main())
