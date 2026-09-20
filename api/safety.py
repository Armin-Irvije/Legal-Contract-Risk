"""Safety rails for ClauseGuard requests — refusals before any model call."""

from __future__ import annotations

import re

DISCLAIMER = (
    "ClauseGuard is a prototype for educational and demo purposes only. "
    "It is not a lawyer and does not provide legal advice. "
    "Do not rely on outputs for real contracts or compliance decisions."
)

MAX_CLAUSE_CHARS = 8_000

# Patterns that ask the tool to hide risk, evade liability, or give binding counsel.
_REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(hide|conceal|obscure)\b.{0,40}\b(risk|liability|obligation)s?\b", re.I),
    re.compile(r"\b(make\s+(this|it)\s+look\s+(safer|fine|ok|okay))\b", re.I),
    re.compile(r"\b(evade|circumvent|dodge)\b.{0,40}\b(liability|law|regulation)s?\b", re.I),
    re.compile(r"\b(how\s+do\s+i\s+(get\s+away\s+with|avoid\s+getting\s+caught))\b", re.I),
    re.compile(r"\b(binding\s+legal\s+advice|act\s+as\s+my\s+lawyer|represent\s+me\s+in\s+court)\b", re.I),
    re.compile(r"\b(jurisdiction[- ]specific\s+counsel|file\s+this\s+in\s+court)\b", re.I),
)


def check_clause_request(clause_text: str) -> str | None:
    """Return a refusal reason when the request should not reach the model, else None."""
    stripped = clause_text.strip()
    if not stripped:
        return "clause_text must be a non-empty string."
    if len(stripped) > MAX_CLAUSE_CHARS:
        return f"clause_text exceeds the {MAX_CLAUSE_CHARS} character limit."

    for pattern in _REFUSAL_PATTERNS:
        if pattern.search(stripped):
            return (
                "This request asks for risk concealment, liability evasion, or binding legal advice, "
                "which ClauseGuard refuses to assist with."
            )
    return None
