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
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8000

# Entrypoint reads PORT at container start. A Dockerfile ${PORT} would be
# expanded at image build time, and $$ inside JSON CMD is the shell PID.
CMD ["/docker-entrypoint.sh"]
