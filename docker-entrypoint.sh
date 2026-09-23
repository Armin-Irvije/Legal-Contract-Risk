#!/bin/sh
# Start uvicorn on Render's PORT, or 8000 when PORT is unset (local Docker).
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
