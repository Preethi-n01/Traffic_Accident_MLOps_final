import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "analytical" / "accident_weather.csv"
REPORT_PATH = PROJECT_ROOT / "models" / "drift_report.json"

DRIFT_THRESHOLD = 0.20

# Features selected for monitoring.
# Target, identifiers, and outcome-related fields are excluded.
FEATURES = [
    "police_force",
    "day_of_week",
    "number_of_vehicles",
    "first_road_class",
    "road_type",
    "speed_limit",
    "junction_detail",
    "junction_control",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "urban_or_rural_area",
    "temperature_mean",
    "precipitation",
    "wind_speed_max",
    "weather_condition",
]


def calculate_psi(reference, current, bins=10):
    """Calculate the Population Stability Index (PSI)."""

    reference = pd.Series(reference)
    current = pd.Series(current)

    if pd.api.types.is_numeric_dtype(reference):
        reference = pd.to_numeric(reference, errors="coerce")
        current = pd.to_numeric(current, errors="coerce")

        reference = reference.replace([np.inf, -np.inf], np.nan).dropna()
        current = current.replace([np.inf, -np.inf], np.nan).dropna()

        if reference.empty or current.empty:
            return None

        edges = np.unique(
            np.quantile(reference, np.linspace(0, 1, bins + 1))
        )

        if len(edges) < 2:
            return 0.0

        edges[0] = -np.inf
        edges[-1] = np.inf

        reference_counts = np.histogram(reference, bins=edges)[0]
        current_counts = np.histogram(current, bins=edges)[0]

    else:
        reference = reference.fillna("MISSING").astype(str)
        current = current.fillna("MISSING").astype(str)

        categories = sorted(set(reference) | set(current))

        reference_counts = (
            reference.value_counts()
            .reindex(categories, fill_value=0)
            .to_numpy()
        )

        current_counts = (
            current.value_counts()
            .reindex(categories, fill_value=0)
            .to_numpy()
        )

    # Smooth zero-frequency bins to avoid division by zero.
    reference_pct = (
        reference_counts + 0.5
    ) / (reference_counts.sum() + 0.5 * len(reference_counts))

    current_pct = (
        current_counts + 0.5
    ) / (current_counts.sum() + 0.5 * len(current_counts))

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def main():
    print("Loading accident dataset...")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Reference: January to August 2025.
    reference = df[
        (df["date"] >= "2025-01-01")
        & (df["date"] < "2025-09-01")
    ]

    # Current: September to October 2025.
    current = df[
        (df["date"] >= "2025-09-01")
        & (df["date"] < "2025-11-01")
    ]

    if reference.empty or current.empty:
        raise ValueError(
            "Reference or current period is empty. "
            "Check the date column and dataset."
        )

    available_features = [
        feature for feature in FEATURES if feature in df.columns
    ]

    if not available_features:
        raise ValueError("None of the selected monitoring features exist.")

    print(f"Reference records: {len(reference)}")
    print(f"Current records: {len(current)}")
    print(f"Features monitored: {len(available_features)}")

    results = []

    for feature in available_features:
        psi = calculate_psi(
            reference[feature],
            current[feature],
        )

        if psi is None:
            status = "INSUFFICIENT_DATA"
        elif psi >= DRIFT_THRESHOLD:
            status = "DRIFT_DETECTED"
        else:
            status = "NO_SIGNIFICANT_DRIFT"

        results.append({
            "feature": feature,
            "psi": round(psi, 6) if psi is not None else None,
            "status": status,
        })

    drifted_features = [
        item["feature"]
        for item in results
        if item["status"] == "DRIFT_DETECTED"
    ]

    report = {
        "project": "Traffic Accident Severity Analytics and Prediction",
        "method": "Population Stability Index (PSI)",
        "reference_period": "2025-01-01 to 2025-08-31",
        "current_period": "2025-09-01 to 2025-10-31",
        "reference_records": int(len(reference)),
        "current_records": int(len(current)),
        "drift_threshold": DRIFT_THRESHOLD,
        "features_monitored": len(available_features),
        "drift_detected": bool(drifted_features),
        "drifted_features": drifted_features,
        "feature_results": results,
        "interpretation": (
            "PSI below 0.10: little distribution change; "
            "0.10 to below 0.20: moderate change; "
            "0.20 or above: substantial change requiring investigation. "
            "Drift does not necessarily mean model performance has declined."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nFeature drift results:")

    for item in results:
        print(
            f"{item['feature']}: "
            f"PSI={item['psi']} | {item['status']}"
        )

    print(f"\nDrift detected: {report['drift_detected']}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()