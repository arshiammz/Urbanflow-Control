# Deployment Guide

## Local API

```bash
python -m pip install -e '.[api]'
cp .env.example .env
# Set URBANFLOW_API_KEY and the final-holdout artifact paths in .env.
uvicorn urbanflow.api.app:app --host 0.0.0.0 --port 8000
```

Readiness requires the final-holdout baseline/greedy configs, decision-time forecast Parquet, zone graph and TLC lookup file. `GET /health` is liveness; `GET /ready` loads and validates serving artifacts.

## Dashboard

```bash
python -m pip install -e '.[dashboard]'
streamlit run dashboards/app.py
```

Set `URBANFLOW_API_URL` and, if configured, `URBANFLOW_API_KEY`.

## Docker Compose

```bash
cp .env.example .env
# Change the API key. Set URBANFLOW_INSTALL_RL=1 to include PyTorch/Gymnasium inference.
docker compose up --build -d
docker compose ps
```

Services:

| Service | Local URL | Purpose |
|---|---|---|
| API | http://localhost:8000 | Forecast and plan endpoints |
| Dashboard | http://localhost:8501 | Operational UI |
| MLflow | http://localhost:5000 | Experiment tracking |
| Prometheus | http://localhost:9090 | API metrics |

The compose file mounts `data`, `artifacts`, `reports` and `configs` read-only into the API container. Keep model and data build jobs outside the serving container. The dashboard and API share the same image.

## Production hardening checklist

- Put API and dashboard behind TLS termination and an authenticated reverse proxy.
- Store API secrets in a secrets manager, not a committed `.env` file.
- Restrict CORS and network access; add rate limits and audit logging at the gateway.
- Use managed Postgres/object storage for MLflow and back up model manifests.
- Pin and scan container dependencies; sign release images.
- Add alerting on API error rate, p95 latency, artifact readiness, forecast PSI, forecast residuals and zone fairness.
- Validate model manifest hashes before every deployment and retain a known-good rollback target.
- Keep operational data and raw TLC files outside the container image.
