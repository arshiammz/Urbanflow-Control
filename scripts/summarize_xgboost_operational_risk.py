from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


TABLES_DIRECTORY = Path(
    "reports/forecasting/error_analysis"
)

REPORT_PATH = Path(
    "reports/forecasting/"
    "2024-01_2024-06-xgboost-operational-risk.json"
)

NORMALIZED_ZONE_METRICS_PATH = (
    TABLES_DIRECTORY
    / "zone_metrics_operational.parquet"
)

TEST_SPLIT = "test"

MINIMUM_ACTUAL_SUM_FOR_WAPE_RANKING = 100.0

TOP_ZONE_COUNT = 20


class OperationalRiskSummaryError(RuntimeError):
    """Raised when operational forecast-risk analysis fails."""


def read_analysis_table(
    name: str,
) -> pl.DataFrame:
    """Read one previously generated error-analysis table."""
    path = TABLES_DIRECTORY / f"{name}.parquet"

    if not path.exists():
        raise OperationalRiskSummaryError(
            f"Analysis table does not exist: {path}"
        )

    if path.stat().st_size == 0:
        raise OperationalRiskSummaryError(
            f"Analysis table is empty: {path}"
        )

    try:
        return pl.read_parquet(path)
    except Exception as error:
        raise OperationalRiskSummaryError(
            f"Could not read analysis table {path}: {error}"
        ) from error


def normalize_wape_columns(
    frame: pl.DataFrame,
) -> pl.DataFrame:
    """Replace undefined or non-finite WAPE metrics with null."""
    required_columns = {
        "actual_sum",
        "model_wape",
        "baseline_wape",
        "wape_absolute_improvement",
        "wape_relative_improvement",
    }

    missing_columns = sorted(
        required_columns.difference(
            frame.columns
        )
    )

    if missing_columns:
        raise OperationalRiskSummaryError(
            "Cannot normalize WAPE; missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    wape_is_defined = (
        (
            pl.col("actual_sum")
            > 0
        )
        & pl.col(
            "model_wape"
        ).is_finite()
        & pl.col(
            "baseline_wape"
        ).is_finite()
    ).fill_null(
        False
    )

    return frame.with_columns(
        pl.when(
            wape_is_defined
        )
        .then(
            pl.col(
                "model_wape"
            )
        )
        .otherwise(
            None
        )
        .alias(
            "model_wape"
        ),

        pl.when(
            wape_is_defined
        )
        .then(
            pl.col(
                "baseline_wape"
            )
        )
        .otherwise(
            None
        )
        .alias(
            "baseline_wape"
        ),

        pl.when(
            wape_is_defined
            & pl.col(
                "wape_absolute_improvement"
            ).is_finite()
        )
        .then(
            pl.col(
                "wape_absolute_improvement"
            )
        )
        .otherwise(
            None
        )
        .alias(
            "wape_absolute_improvement"
        ),

        pl.when(
            wape_is_defined
            & pl.col(
                "wape_relative_improvement"
            ).is_finite()
        )
        .then(
            pl.col(
                "wape_relative_improvement"
            )
        )
        .otherwise(
            None
        )
        .alias(
            "wape_relative_improvement"
        ),

        wape_is_defined.alias(
            "wape_defined"
        ),
    )


def records(
    frame: pl.DataFrame,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert a frame to JSON-compatible records."""
    selected = (
        frame.head(
            limit
        )
        if limit is not None
        else frame
    )

    return selected.to_dicts()


def require_single_record(
    frame: pl.DataFrame,
    *,
    description: str,
) -> dict[str, Any]:
    """Require exactly one matching aggregate record."""
    if frame.height != 1:
        raise OperationalRiskSummaryError(
            f"Expected one {description} record, "
            f"received {frame.height}."
        )

    return frame.to_dicts()[0]


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Calculate a finite ratio with a zero-denominator fallback."""
    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def round_optional(
    value: Any,
    digits: int = 8,
) -> float | None:
    """Round a finite optional numeric value."""
    if value is None:
        return None

    numeric_value = float(
        value
    )

    if not math.isfinite(
        numeric_value
    ):
        return None

    return round(
        numeric_value,
        digits,
    )


def sanitize_json_value(
    value: Any,
) -> Any:
    """Recursively replace non-finite floats with JSON null."""
    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            return None

        return value

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): sanitize_json_value(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            sanitize_json_value(
                item
            )
            for item in value
        ]

    return value


def write_json_atomically(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write a standards-compliant JSON report atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f"{path.name}.part"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    sanitized_payload = sanitize_json_value(
        payload
    )

    try:
        temporary_path.write_text(
            json.dumps(
                sanitized_payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(
            path
        )

    except (
        OSError,
        TypeError,
        ValueError,
    ) as error:
        temporary_path.unlink(
            missing_ok=True
        )

        raise OperationalRiskSummaryError(
            "Could not write operational-risk report: "
            f"{error}"
        ) from error


def write_parquet_atomically(
    frame: pl.DataFrame,
    *,
    path: Path,
) -> None:
    """Write one normalized Parquet artifact atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f"{path.name}.part"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    try:
        frame.write_parquet(
            temporary_path,
            compression="zstd",
            statistics=True,
        )

        temporary_path.replace(
            path
        )

    except Exception as error:
        temporary_path.unlink(
            missing_ok=True
        )

        raise OperationalRiskSummaryError(
            f"Could not write normalized zone metrics: {error}"
        ) from error


def build_operational_risk_summary() -> dict[str, Any]:
    """Build a business-oriented summary of XGBoost errors."""
    overall_metrics = normalize_wape_columns(
        read_analysis_table(
            "overall_metrics"
        )
    )

    zone_metrics = normalize_wape_columns(
        read_analysis_table(
            "zone_metrics"
        )
    )

    borough_metrics = normalize_wape_columns(
        read_analysis_table(
            "borough_metrics"
        )
    )

    peak_period_metrics = normalize_wape_columns(
        read_analysis_table(
            "peak_period_metrics"
        )
    )

    demand_band_metrics = normalize_wape_columns(
        read_analysis_table(
            "demand_band_metrics"
        )
    )

    test_overall = overall_metrics.filter(
        pl.col(
            "split"
        )
        == TEST_SPLIT
    )

    test_zones = zone_metrics.filter(
        pl.col(
            "split"
        )
        == TEST_SPLIT
    )

    test_boroughs = borough_metrics.filter(
        pl.col(
            "split"
        )
        == TEST_SPLIT
    )

    test_peak_periods = peak_period_metrics.filter(
        pl.col(
            "split"
        )
        == TEST_SPLIT
    )

    test_demand_bands = demand_band_metrics.filter(
        pl.col(
            "split"
        )
        == TEST_SPLIT
    )

    overall_record = require_single_record(
        test_overall,
        description="test overall",
    )

    zero_band = require_single_record(
        test_demand_bands.filter(
            pl.col(
                "demand_band"
            )
            == "zero"
        ),
        description="zero-demand band",
    )

    high_demand_band = require_single_record(
        test_demand_bands.filter(
            pl.col(
                "demand_band"
            )
            == ">60"
        ),
        description="high-demand band",
    )

    low_demand_bands = (
        test_demand_bands.filter(
            pl.col(
                "demand_band"
            ).is_in(
                [
                    "01-05",
                    "06-15",
                ]
            )
        )
        .sort(
            "actual_mean"
        )
    )

    total_test_rows = int(
        overall_record[
            "row_count"
        ]
    )

    zero_demand_rows = int(
        zero_band[
            "row_count"
        ]
    )

    high_demand_rows = int(
        high_demand_band[
            "row_count"
        ]
    )

    zero_demand_zones = (
        test_zones.filter(
            pl.col(
                "actual_sum"
            )
            == 0
        )
        .sort(
            "model_prediction_sum",
            descending=True,
        )
        .select(
            "zone_id",
            "zone_name",
            "borough",
            "service_zone",
            "row_count",
            "actual_sum",
            "model_prediction_sum",
            "model_mae",
            "model_mean_bias",
            "baseline_mae",
            "model_beats_baseline",
            "wape_defined",
        )
    )

    sparse_demand_zones = (
        test_zones.filter(
            (
                pl.col(
                    "actual_sum"
                )
                > 0
            )
            & (
                pl.col(
                    "actual_mean"
                )
                < 1.0
            )
        )
        .sort(
            "model_wape",
            descending=True,
            nulls_last=True,
        )
        .select(
            "zone_id",
            "zone_name",
            "borough",
            "actual_sum",
            "actual_mean",
            "model_mae",
            "model_wape",
            "baseline_mae",
            "mae_relative_improvement",
            "model_mean_bias",
        )
    )

    worst_zones_by_stable_wape = (
        test_zones.filter(
            pl.col(
                "actual_sum"
            )
            >= MINIMUM_ACTUAL_SUM_FOR_WAPE_RANKING
        )
        .sort(
            "model_wape",
            descending=True,
            nulls_last=True,
        )
        .select(
            "zone_id",
            "zone_name",
            "borough",
            "service_zone",
            "row_count",
            "actual_sum",
            "actual_mean",
            "model_mae",
            "model_wape",
            "baseline_mae",
            "baseline_wape",
            "mae_relative_improvement",
            "model_mean_bias",
        )
        .head(
            TOP_ZONE_COUNT
        )
    )

    worst_zones_by_mae = (
        test_zones.sort(
            "model_mae",
            descending=True,
        )
        .select(
            "zone_id",
            "zone_name",
            "borough",
            "service_zone",
            "actual_mean",
            "model_mae",
            "model_rmse",
            "model_wape",
            "baseline_mae",
            "mae_relative_improvement",
            "model_mean_bias",
        )
        .head(
            TOP_ZONE_COUNT
        )
    )

    strongest_underforecast_zones = (
        test_zones.sort(
            "model_mean_bias"
        )
        .select(
            "zone_id",
            "zone_name",
            "borough",
            "actual_mean",
            "model_mae",
            "model_wape",
            "model_mean_bias",
            "baseline_mae",
            "mae_relative_improvement",
        )
        .head(
            TOP_ZONE_COUNT
        )
    )

    strongest_overforecast_zones = (
        test_zones.sort(
            "model_mean_bias",
            descending=True,
        )
        .select(
            "zone_id",
            "zone_name",
            "borough",
            "actual_mean",
            "model_mae",
            "model_wape",
            "model_mean_bias",
            "baseline_mae",
            "mae_relative_improvement",
        )
        .head(
            TOP_ZONE_COUNT
        )
    )

    zones_where_baseline_wins = (
        test_zones.filter(
            ~pl.col(
                "model_beats_baseline"
            )
        )
        .sort(
            "mae_relative_improvement"
        )
        .select(
            "zone_id",
            "zone_name",
            "borough",
            "actual_sum",
            "actual_mean",
            "model_mae",
            "baseline_mae",
            "mae_relative_improvement",
            "model_mean_bias",
            "wape_defined",
        )
    )

    normalized_zone_metrics = zone_metrics.sort(
        "split",
        "zone_id",
    )

    write_parquet_atomically(
        normalized_zone_metrics,
        path=NORMALIZED_ZONE_METRICS_PATH,
    )

    zero_mean_prediction = safe_ratio(
        float(
            zero_band[
                "model_prediction_sum"
            ]
        ),
        float(
            zero_band[
                "row_count"
            ]
        ),
    )

    baseline_zero_mean_prediction = safe_ratio(
        float(
            zero_band[
                "baseline_prediction_sum"
            ]
        ),
        float(
            zero_band[
                "row_count"
            ]
        ),
    )

    report: dict[str, Any] = {
        "operational_risk_version": "0.1.1",
        "generated_at": datetime.now(
            UTC
        ).isoformat(),
        "definitions": {
            "wape": (
                "WAPE is reported only when the segment's "
                "actual-demand sum is greater than zero."
            ),
            "zero_demand_risk": (
                "Positive predictions when actual demand is zero "
                "represent potential unnecessary vehicle allocation."
            ),
            "high_demand_risk": (
                "Negative mean bias in high-demand segments represents "
                "potential under-allocation and unmet demand."
            ),
        },
        "test_overall": {
            "row_count": total_test_rows,
            "actual_sum": int(
                overall_record[
                    "actual_sum"
                ]
            ),
            "model_mae": round_optional(
                overall_record[
                    "model_mae"
                ]
            ),
            "model_rmse": round_optional(
                overall_record[
                    "model_rmse"
                ]
            ),
            "model_wape": round_optional(
                overall_record[
                    "model_wape"
                ]
            ),
            "model_smape": round_optional(
                overall_record[
                    "model_smape"
                ]
            ),
            "model_mean_bias": round_optional(
                overall_record[
                    "model_mean_bias"
                ]
            ),
            "baseline_mae": round_optional(
                overall_record[
                    "baseline_mae"
                ]
            ),
            "mae_relative_improvement": round_optional(
                overall_record[
                    "mae_relative_improvement"
                ]
            ),
        },
        "zero_demand_risk": {
            "row_count": zero_demand_rows,
            "row_share": round(
                safe_ratio(
                    zero_demand_rows,
                    total_test_rows,
                ),
                8,
            ),
            "model_mean_prediction": round(
                zero_mean_prediction,
                8,
            ),
            "baseline_mean_prediction": round(
                baseline_zero_mean_prediction,
                8,
            ),
            "model_prediction_sum": round_optional(
                zero_band[
                    "model_prediction_sum"
                ]
            ),
            "baseline_prediction_sum": round_optional(
                zero_band[
                    "baseline_prediction_sum"
                ]
            ),
            "model_mae": round_optional(
                zero_band[
                    "model_mae"
                ]
            ),
            "baseline_mae": round_optional(
                zero_band[
                    "baseline_mae"
                ]
            ),
            "mae_relative_improvement": round_optional(
                zero_band[
                    "mae_relative_improvement"
                ]
            ),
            "model_smape": round_optional(
                zero_band[
                    "model_smape"
                ]
            ),
            "wape": None,
            "zero_actual_zone_count": (
                zero_demand_zones.height
            ),
        },
        "high_demand_risk": {
            "band": ">60",
            "row_count": high_demand_rows,
            "row_share": round(
                safe_ratio(
                    high_demand_rows,
                    total_test_rows,
                ),
                8,
            ),
            "actual_mean": round_optional(
                high_demand_band[
                    "actual_mean"
                ]
            ),
            "model_mae": round_optional(
                high_demand_band[
                    "model_mae"
                ]
            ),
            "model_rmse": round_optional(
                high_demand_band[
                    "model_rmse"
                ]
            ),
            "model_wape": round_optional(
                high_demand_band[
                    "model_wape"
                ]
            ),
            "model_mean_bias": round_optional(
                high_demand_band[
                    "model_mean_bias"
                ]
            ),
            "baseline_mae": round_optional(
                high_demand_band[
                    "baseline_mae"
                ]
            ),
            "mae_relative_improvement": round_optional(
                high_demand_band[
                    "mae_relative_improvement"
                ]
            ),
        },
        "low_demand_bands": records(
            low_demand_bands
        ),
        "demand_band_metrics": records(
            test_demand_bands.sort(
                "actual_mean"
            )
        ),
        "peak_period_metrics": records(
            test_peak_periods.sort(
                "peak_period"
            )
        ),
        "borough_metrics": records(
            test_boroughs.sort(
                "borough"
            )
        ),
        "zone_findings": {
            "zero_demand_zones": records(
                zero_demand_zones,
                limit=TOP_ZONE_COUNT,
            ),
            "sparse_demand_zones": records(
                sparse_demand_zones,
                limit=TOP_ZONE_COUNT,
            ),
            "worst_zones_by_stable_wape": records(
                worst_zones_by_stable_wape
            ),
            "worst_zones_by_mae": records(
                worst_zones_by_mae
            ),
            "strongest_underforecast_zones": records(
                strongest_underforecast_zones
            ),
            "strongest_overforecast_zones": records(
                strongest_overforecast_zones
            ),
            "zones_where_baseline_wins": records(
                zones_where_baseline_wins,
                limit=TOP_ZONE_COUNT,
            ),
            "zones_where_baseline_wins_count": (
                zones_where_baseline_wins.height
            ),
        },
        "artifacts": {
            "normalized_zone_metrics": {
                "path": str(
                    NORMALIZED_ZONE_METRICS_PATH
                ),
                "size_bytes": (
                    NORMALIZED_ZONE_METRICS_PATH
                    .stat()
                    .st_size
                ),
                "row_count": (
                    normalized_zone_metrics.height
                ),
            },
        },
        "invariants": {
            "overall_wape_is_defined": (
                overall_record[
                    "model_wape"
                ]
                is not None
            ),
            "zero_demand_wape_is_undefined": (
                zero_band[
                    "model_wape"
                ]
                is None
            ),
            "all_zero_actual_zone_wape_is_undefined": (
                zero_demand_zones
                .get_column(
                    "wape_defined"
                )
                .not_()
                .all()
                if zero_demand_zones.height > 0
                else True
            ),
            "test_rows_match_demand_bands": (
                int(
                    test_demand_bands[
                        "row_count"
                    ].sum()
                )
                == total_test_rows
            ),
        },
    }

    write_json_atomically(
        REPORT_PATH,
        report,
    )

    return report


def main() -> None:
    """Build and print the operational-risk summary."""
    report = build_operational_risk_summary()

    overall = report[
        "test_overall"
    ]

    zero_risk = report[
        "zero_demand_risk"
    ]

    high_risk = report[
        "high_demand_risk"
    ]

    zone_findings = report[
        "zone_findings"
    ]

    print(
        "XGBoost operational-risk summary completed"
    )

    print()

    print(
        f"Test rows: {overall['row_count']:,}"
    )

    print(
        f"Overall MAE: {overall['model_mae']:.4f}"
    )

    print(
        "Overall WAPE: "
        f"{overall['model_wape'] * 100:.2f}%"
    )

    print()

    print(
        "Zero-demand rows:",
        f"{zero_risk['row_count']:,}",
        (
            f"({zero_risk['row_share'] * 100:.2f}%)"
        ),
    )

    print(
        "Zero-demand mean prediction:",
        f"{zero_risk['model_mean_prediction']:.4f}",
    )

    print(
        "Zero-actual zones:",
        zero_risk[
            "zero_actual_zone_count"
        ],
    )

    print()

    print(
        "High-demand rows:",
        f"{high_risk['row_count']:,}",
        (
            f"({high_risk['row_share'] * 100:.2f}%)"
        ),
    )

    print(
        "High-demand MAE:",
        f"{high_risk['model_mae']:.4f}",
    )

    print(
        "High-demand mean bias:",
        f"{high_risk['model_mean_bias']:.4f}",
    )

    print()

    print(
        "Zones where lag-1 remains better:",
        zone_findings[
            "zones_where_baseline_wins_count"
        ],
    )

    print()

    print(
        f"Report: {REPORT_PATH}"
    )

    print(
        "Normalized zones:",
        NORMALIZED_ZONE_METRICS_PATH,
    )


if __name__ == "__main__":
    main()