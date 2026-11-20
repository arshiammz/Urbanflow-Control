"""Static deployment preflight; does not require private model or data artifacts."""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import yaml


REQUIRED_FILES = (
    Path("Dockerfile"),
    Path("docker-compose.yml"),
    Path(".env.example"),
    Path("infra/prometheus/prometheus.yml"),
    Path(".github/workflows/ci.yml"),
    Path("src/urbanflow/api/app.py"),
    Path("dashboards/app.py"),
    Path("docs/phase5/README.md"),
)
REQUIRED_CONFIGS = (
    Path("configs/evaluation/final_holdout_july.yaml"),
    Path("configs/evaluation/stress_scenarios.yaml"),
    Path("configs/forecasting/probabilistic_graph_weather_july_holdout.yaml"),
    Path("configs/forecasting/decision_time_quantile_horizon_july_holdout.yaml"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    missing = [str(path) for path in (*REQUIRED_FILES, *REQUIRED_CONFIGS) if not path.exists()]
    if missing:
        raise SystemExit("Production source files are missing: " + ", ".join(missing))
    with Path("pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    extras = project["project"].get("optional-dependencies", {})
    for extra in ("api", "dashboard", "mlops", "rl", "dev"):
        if extra not in extras:
            raise SystemExit(f"Missing optional dependency group: {extra}")
    for path in REQUIRED_CONFIGS:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise SystemExit(f"Config must be a YAML mapping: {path}")
    print("Static production source checks passed")
    print("Required files:", len(REQUIRED_FILES))
    print("Required configs:", len(REQUIRED_CONFIGS))
    if not args.static_only:
        print("Runtime artifact checks are intentionally deferred until local data/model build.")


if __name__ == "__main__":
    main()
