FROM python:3.11-slim@sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76 AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000 FAISS_OPT_LEVEL=generic
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.7,<3" \
    && pip install --no-cache-dir -r requirements.lock
ENV HF_HOME=/app/data/models/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN useradd --create-home --uid 10001 appuser \
    && install -d -o root -g root -m 0755 /run/ai-defense \
    && install -o root -g root -m 0444 /dev/null /run/ai-defense/release-ed25519.pub.pem
COPY --chown=appuser:appuser app/__init__.py ./app/__init__.py
COPY --chown=appuser:appuser app/backend ./app/backend
COPY --chown=appuser:appuser defend ./defend
COPY --chown=appuser:appuser loop ./loop
COPY --chown=appuser:appuser contracts/event.schema.json ./contracts/event.schema.json
COPY --chown=appuser:appuser sim/__init__.py sim/mandate_twin.py sim/population.py ./sim/
COPY --chown=appuser:appuser scripts/release.py ./scripts/
COPY --chown=appuser:appuser data/events.manifest.json data/market_prices.csv ./data/
COPY --chown=appuser:appuser data/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2 ./data/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
COPY --chown=appuser:appuser results ./results
COPY --chown=appuser:appuser Dockerfile .dockerignore ./
COPY --chown=appuser:appuser app/frontend/package.json app/frontend/package-lock.json ./app/frontend/
COPY --chown=appuser:appuser app/frontend/dist ./app/frontend/dist
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + '${PORT}' + '/health', timeout=2).read()"
CMD ["sh", "-c", "exec uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT}"]
