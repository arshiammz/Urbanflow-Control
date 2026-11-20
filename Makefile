.PHONY: install install-prod lint test build-july prepare-july final-holdout stress api dashboard docker-up docker-down monitoring-reference monitoring-report mlflow-log promote verify-manifest rollback-manifest

install:
	python -m pip install -e '.[dev,rl,api,dashboard,mlops]'

install-prod:
	python -m pip install -e '.[api,dashboard]'

lint:
	ruff check src scripts dashboards tests

test:
	pytest -q

build-july:
	python -m urbanflow.cli.main download-zone-lookup
	python -m urbanflow.cli.main build-multimonth --config configs/data/multimonth_july_holdout.yaml
	python -m urbanflow.cli.main build-features --config configs/forecasting/features_july_holdout.yaml
	python scripts/build_real_simulation_inputs.py --config configs/simulation/real_data_inputs_july_holdout.yaml
	python scripts/build_zone_centroids.py
	python scripts/build_zone_graph.py --zone-index data/processed/simulation/nyc_2024-01_2024-07_all_zones/zone_index.parquet --output data/processed/spatial/nyc_taxi_zone_graph_july_holdout.json
	python scripts/fetch_weather_forecast_archive.py --config configs/weather/open_meteo_july_holdout.yaml
	python scripts/build_spatiotemporal_features.py --config configs/forecasting/spatiotemporal_features_july_holdout.yaml
	python scripts/train_probabilistic_forecaster.py --config configs/forecasting/probabilistic_graph_weather_july_holdout.yaml
	python scripts/build_decision_time_quantile_forecasts.py --config configs/forecasting/decision_time_quantile_horizon_july_holdout.yaml

final-holdout:
	python scripts/run_final_holdout.py --config configs/evaluation/final_holdout_july.yaml

stress:
	python scripts/run_stress_scenarios.py --config configs/evaluation/stress_scenarios.yaml

api:
	uvicorn urbanflow.api.app:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboards/app.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

monitoring-reference:
	python scripts/build_monitoring_reference.py

monitoring-report:
	python scripts/run_monitoring_report.py

mlflow-log:
	python scripts/log_experiment_to_mlflow.py --report reports/evaluation/2024-07-final-holdout.json

promote:
	python scripts/create_production_manifest.py

verify-manifest:
	python scripts/verify_production_manifest.py

rollback-manifest:
	python scripts/rollback_production_manifest.py
