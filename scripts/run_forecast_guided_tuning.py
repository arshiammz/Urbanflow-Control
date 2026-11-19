"""Run the complete validation-only search and print the selected parameters."""

from pathlib import Path

from urbanflow.simulation.forecast_guided_tuning import run_forecast_guided_tuning


def progress(done: int, total: int) -> None:
    if done == 1 or done % 10 == 0 or done == total:
        print(f"Candidates evaluated: {done}/{total}", flush=True)


def main() -> None:
    report = run_forecast_guided_tuning(
        Path("configs/simulation/forecast_guided_tuning.yaml"), progress_callback=progress
    )
    champion = report["champion_candidate"]
    ref = report["reference_candidate"]
    print("\nMay validation-only search completed")
    print("Feasible:", report["search_grid"]["feasible_count"], "/108")
    print("Reference:", ref["candidate_id"], "cost=", round(ref["policy_total_operating_cost"], 2))
    for name in (
        "candidate_id", "forecast_horizon_steps", "demand_buffer_multiplier",
        "minimum_zone_reserve", "maximum_relocation_steps", "maximum_relocation_fraction",
        "in_transit_supply_credit", "days_beating_baseline_cost", "days_reducing_unmet_demand",
        "policy_unmet_demand_total", "policy_service_rate", "policy_total_operating_cost",
        "worst_day_service_rate_improvement", "relocation_vehicle_steps",
    ):
        print(f"{name}: {champion[name]}")
    print("Report: reports/simulation/2024-05-02_2024-05-30-forecast-guided-tuning.json")
    print("Next: python scripts/promote_forecast_guided_champion.py")


if __name__ == "__main__":
    main()
