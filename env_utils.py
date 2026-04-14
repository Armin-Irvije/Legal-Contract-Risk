from __future__ import annotations

import os
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs into os.environ when absent."""
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_project_env(source_file: str) -> Path:
    """Load the `.env` file that lives beside the given source file."""
    env_path = Path(source_file).resolve().parent / ".env"
    load_env_file(env_path)
    return env_path


def normalize_api_key_env() -> None:
    """Mirror CLAUDE_API_KEY into ANTHROPIC_API_KEY when needed."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return

    claude_api_key = os.getenv("CLAUDE_API_KEY")
    if claude_api_key:
        os.environ["ANTHROPIC_API_KEY"] = claude_api_key
