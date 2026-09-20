FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIPELINE_MODEL=openai/gpt-4o-mini \
    PIPELINE_MAX_TOKENS=900 \
    CLAUSEGUARD_MODEL=openai/gpt-4o-mini

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# API runtime needs shared CLI modules for prompts, parsing, and cost.
COPY api/ api/
COPY prompts/ prompts/
COPY pipeline.py cost.py env_utils.py telemetry.py pricing.json ./

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
