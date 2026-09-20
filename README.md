# Legal Clause Risk Evaluator (ClauseGuard)

Prototype for **legaltech / LLMOps**: score individual contract clauses for risk, suggest a redline, and measure prompt quality with an LLM-as-judge harness. Includes a thin **ClauseGuard HTTP API** (OpenRouter) for demos.

> **Not legal advice.** Demo and educational use only. Use synthetic clauses in public demos — never real client contracts.

## What it does

| Path | Purpose |
|------|---------|
| **CLI** (`evaluate.py`) | Run Anthropic pipeline + judge over a clause dataset; log tokens, latency, cost |
| **API** (`api/`) | `POST /analyze` via cheap OpenRouter models with safety refusals |

Shared pieces: prompt versions (`prompts/v1`–`v3`), strict JSON validation, format-fix retry, pricing in `pricing.json`.

## Setup

```bash
python -m venv myenv
# Windows
.\myenv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

| Variable | Used by | Notes |
|----------|---------|--------|
| `ANTHROPIC_API_KEY` | CLI | Required for `evaluate.py` |
| `PIPELINE_MODEL` / `JUDGE_MODEL` | CLI | Defaults in `.env.example` |
| `OPENROUTER_API_KEY` | API | Required for ClauseGuard |
| `CLAUSEGUARD_MODEL` | API | Default `openai/gpt-4o-mini` |

## CLI (eval harness)

Smoke eval (default dataset `data/smoke_set.json`):

```bash
python evaluate.py
```

Compare prompts:

```bash
python evaluate.py --prompt v1 v2
```

Single clause:

```bash
python evaluate.py --clause-text "Supplier shall indemnify Customer against third-party IP claims."
```

Custom dataset / models:

```bash
python evaluate.py --clauses data/smoke_set.json --pipeline-model claude-haiku-4-5-20251001 --judge-model claude-haiku-4-5-20251001
```

**Outputs**

- `runs.jsonl` — append-only run log (prompt hash, scores, tokens, cost)
- `output/` — per-run judge reports (JSON)

## ClauseGuard API

```bash
uvicorn api.main:app --reload
```

- `GET /health` — liveness
- `POST /analyze` — analyze one clause (OpenRouter)

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/analyze ^
  -H "Content-Type: application/json" ^
  -d "{\"clause_text\":\"Supplier shall indemnify Customer against third-party IP claims.\",\"prompt\":\"v1\"}"
```

Successful responses include `analysis` (`risk_level`, `explanation`, `suggested_redline`), `metadata` (model, latency, tokens, estimated cost), and a disclaimer.

Safety rails refuse empty/oversized input and requests that ask to hide risk, evade liability, or get binding legal advice (HTTP 400 + structured refusal). Offline checks: `api/test_safety.py`.

## Layout

```text
pipeline.py      # clause analysis (Anthropic)
judge.py         # LLM-as-judge scoring
evaluate.py      # CLI entrypoint
cost.py / pricing.json / telemetry.py
prompts/         # v1, v2, v3 templates
data/            # smoke / golden clause sets
api/             # FastAPI ClauseGuard (OpenRouter)
  main.py
  service.py
  providers.py
  safety.py
  schemas.py
```

## Safety and non-goals

- Prototype only — not a substitute for a lawyer
- MVP: no PDF/DOCX ingest, no auth, no DMS integrations, no jurisdiction-specific counsel
- Public demo data should stay synthetic

## Roadmap (short)

- Docker Compose for the API
- TypeScript/Next.js UI (`web/`)
- Zero-cost public hosting (Vercel UI + free API or tunnel)
