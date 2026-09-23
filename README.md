# Legal Clause Risk Evaluator (ClauseGuard)

Prototype for **legaltech / LLMOps**: score individual contract clauses for risk, suggest a redline, and measure prompt quality with an LLM-as-judge harness. Includes a thin **ClauseGuard HTTP API** (OpenRouter) for demos.

> **Not legal advice.** Demo and educational use only. Use synthetic clauses in public demos — never real client contracts.

## What it does

| Path | Purpose |
|------|---------|
| **CLI** (`evaluate.py`) | Run Anthropic pipeline + judge over a clause dataset; log tokens, latency, cost |
| **API** (`api/`) | `POST /analyze` via cheap OpenRouter models with safety refusals |
| **Web** (`web/`) | Next.js UI — paste clause → risk / explanation / redline + cost |

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

### Docker (API + web UI)

Requires Docker Desktop (or Engine + Compose). Put `OPENROUTER_API_KEY` in `.env` first.

```bash
docker compose up --build
```

Starts both services:

| Service | URL |
|---------|-----|
| API | `http://127.0.0.1:8000` |
| Web UI | `http://localhost:3000` |

The UI image bakes `NEXT_PUBLIC_API_URL` at build time (default `http://127.0.0.1:8000`). Override when building if needed:

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 docker compose up --build
```

```bash
curl -s http://127.0.0.1:8000/health
```

Stop with `Ctrl+C` or `docker compose down`.

## ClauseGuard web UI (local Node, optional)

For UI hot-reload without rebuilding the web image:

```bash
cd web
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. Point `NEXT_PUBLIC_API_URL` at the API (default `http://127.0.0.1:8000`). The API allows those origins via `CLAUSEGUARD_CORS_ORIGINS`. Keep the API up via Compose or `uvicorn`.

## Public hosting (Render API + Vercel UI)

The browser calls the API directly (`NEXT_PUBLIC_API_URL`). Host the FastAPI image on Render and the Next.js app on Vercel.

### Render (API)

1. Push this repo, then create a **Web Service** from it.
2. Runtime **Docker**, Dockerfile `./Dockerfile` (repo root), instance **Free**, health check path `/health`.
3. The image listens on `$PORT` (Render’s default is `10000`). Local Compose pins `PORT=8000`.
4. Set environment variables on the service:
   - `OPENROUTER_API_KEY`
   - `CLAUSEGUARD_MODEL` (default `openai/gpt-4o-mini`)
   - `CLAUSEGUARD_CORS_ORIGINS` — start with `http://localhost:3000,http://127.0.0.1:3000`, then add the Vercel origin after the UI deploy
5. Copy the public URL (for example `https://clauseguard-api.onrender.com`).

Free web services spin down after about 15 minutes without traffic. The first request after idle is slower.

### Vercel (web)

1. Import the same repo. Set **Root Directory** to `web` (Next.js preset).
2. Set `NEXT_PUBLIC_API_URL` to the Render URL with no trailing slash (Production and Preview).
3. Deploy and copy the site URL (for example `https://your-app.vercel.app`).

`NEXT_PUBLIC_API_URL` is baked in at build time. Changing it requires a new Vercel deploy.

### Wire CORS

On Render, set `CLAUSEGUARD_CORS_ORIGINS` to the local origins plus the Vercel origin, then redeploy the API (CORS is read at process start):

```text
http://localhost:3000,http://127.0.0.1:3000,https://your-app.vercel.app
```

Preview deployments need their own origins added the same way. Smoke-check `GET /health`, then open the Vercel UI and analyze a synthetic clause.

## Layout

```text
pipeline.py      # clause analysis (Anthropic)
judge.py         # LLM-as-judge scoring
evaluate.py      # CLI entrypoint
cost.py / pricing.json / telemetry.py
prompts/         # v1, v2, v3 templates
data/            # smoke / golden clause sets
api/             # FastAPI ClauseGuard (OpenRouter)
web/             # Next.js + TypeScript UI
Dockerfile         # API image
web/Dockerfile     # Next.js UI image
docker-compose.yml # local self-host (API + web)
```

## Safety and non-goals

- Prototype only — not a substitute for a lawyer
- MVP: no PDF/DOCX ingest, no auth, no DMS integrations, no jurisdiction-specific counsel
- Public demo data should stay synthetic

## Roadmap (short)

- [x] Docker Compose for the API (+ web UI)
- [x] TypeScript/Next.js UI (`web/`)
- [ ] Zero-cost public hosting — Render API + Vercel UI (see Public hosting)
