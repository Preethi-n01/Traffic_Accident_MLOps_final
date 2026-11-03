from pathlib import Path
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import requests

from src.modeling.features import TARGET, build_features, load_labeled_data

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "analytical" / "accident_weather.csv"
MODEL_PATH = ROOT / "models" / "accident_severity_pipeline.joblib"
REPORT_PATH = ROOT / "reports" / "evaluation.json"
OUTPUT_PATH = ROOT / "reports" / "monitoring.json"
BASELINE_FRACTION = 0.7
TEST_FRACTION = 0.2
DRIFT_THRESHOLD = 0.2
CLASS_DRIFT_THRESHOLD = 0.1
FATAL_FNR_THRESHOLD = 0.2
API_FAILURE_RATE_THRESHOLD = 0.05
API_P95_LATENCY_THRESHOLD_MS = 1000.0


def total_variation_distance(reference: pd.Series, current: pd.Series) -> float:
    """Return the total variation distance between two categorical distributions."""
    reference_distribution = reference.astype("string").fillna("__MISSING__").value_counts(
        normalize=True
    )
    current_distribution = current.astype("string").fillna("__MISSING__").value_counts(
        normalize=True
    )
    categories = reference_distribution.index.union(current_distribution.index)
    return float(
        0.5
        * (
            reference_distribution.reindex(categories, fill_value=0)
            - current_distribution.reindex(categories, fill_value=0)
        ).abs().sum()
    )


def population_stability_index(reference: pd.Series, current: pd.Series) -> float:
    """Compare numeric feature distributions using reference quantile bins."""
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if not len(reference_values) or not len(current_values):
        return 0.0

    quantiles = np.quantile(reference_values, np.linspace(0, 1, 11))
    boundaries = np.unique(quantiles)
    if len(boundaries) == 1:
        return 0.0 if np.allclose(current_values, boundaries[0]) else 1.0
    if len(boundaries) == 2:
        bins = np.array([-np.inf, boundaries.mean(), np.inf])
    else:
        bins = np.concatenate(([-np.inf], boundaries[1:-1], [np.inf]))
    reference_counts = np.histogram(reference_values, bins=bins)[0].astype(float)
    current_counts = np.histogram(current_values, bins=bins)[0].astype(float)
    reference_distribution = reference_counts / reference_counts.sum()
    current_distribution = current_counts / current_counts.sum()
    epsilon = 1e-6
    return float(
        np.sum(
            (current_distribution - reference_distribution)
            * np.log(
                (current_distribution + epsilon)
                / (reference_distribution + epsilon)
            )
        )
    )


def location_cells(df: pd.DataFrame) -> pd.Series:
    latitude = (pd.to_numeric(df["latitude"], errors="coerce") * 2).round() / 2
    longitude = (pd.to_numeric(df["longitude"], errors="coerce") * 2).round() / 2
    return (
        latitude.astype("string").fillna("__MISSING__")
        + ","
        + longitude.astype("string").fillna("__MISSING__")
    )


def fetch_api_metrics(url: str) -> dict:
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        return {"status": "available", "metrics": response.json()}
    except requests.RequestException as exc:
        return {"status": "unavailable", "error": str(exc)}


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Analytical dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained inference pipeline not found: {MODEL_PATH}")

    df = load_labeled_data(DATA_PATH)
    if len(df) < 2:
        raise ValueError("At least two dated, labeled accidents are required.")

    split_idx = int(len(df) * BASELINE_FRACTION)
    test_idx = int(len(df) * (1 - TEST_FRACTION))
    if split_idx == 0 or test_idx == 0 or test_idx == len(df):
        raise ValueError("Monitoring needs non-empty baseline, recent, and test windows.")

    baseline = df.iloc[:split_idx]
    recent = df.iloc[split_idx:]
    test = df.iloc[test_idx:]
    features = build_features(test)
    actual = test[TARGET].astype(int)
    model = joblib.load(MODEL_PATH)

    started = time.perf_counter()
    predictions = model.predict(features)
    prediction_seconds = time.perf_counter() - started
    labels = sorted(df[TARGET].astype(int).unique().tolist())

    from sklearn.metrics import confusion_matrix, f1_score, recall_score

    matrix = confusion_matrix(actual, predictions, labels=labels)
    false_negative_rates = {}
    for index, label in enumerate(labels):
        positives = int(matrix[index, :].sum())
        false_negative_rates[str(label)] = (
            float((positives - matrix[index, index]) / positives)
            if positives
            else None
        )

    class_drift = total_variation_distance(
        baseline[TARGET], recent[TARGET]
    )
    location_drift = total_variation_distance(
        location_cells(baseline), location_cells(recent)
    )

    numeric_candidates = [
        "latitude",
        "longitude",
        "speed_limit",
        "number_of_vehicles",
        "number_of_casualties",
        "temperature_mean",
        "precipitation",
        "wind_speed_max",
    ]
    numeric_drift = {
        name: population_stability_index(baseline[name], recent[name])
        for name in numeric_candidates
        if name in baseline.columns
    }
    categorical_candidates = [
        "weather_condition",
        "urban_or_rural_area",
        "road_type",
        "light_conditions",
        "weather_conditions",
    ]
    categorical_drift = {
        name: total_variation_distance(baseline[name], recent[name])
        for name in categorical_candidates
        if name in baseline.columns
    }

    report = {
        "rows_checked": int(len(df)),
        "baseline_rows": int(len(baseline)),
        "recent_rows": int(len(recent)),
        "test_rows": int(len(test)),
        "date_range": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        },
        "input_quality": {
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values_total": int(df.isna().sum().sum()),
            "missing_values_by_column": {
                name: int(count)
                for name, count in df.isna().sum().items()
                if count > 0
            },
        },
        "severity_distribution": {
            str(label): int(count)
            for label, count in df[TARGET].value_counts().sort_index().items()
        },
        "drift": {
            "comparison": "first 70% chronological baseline vs latest 30% data",
            "class_tvd": class_drift,
            "location_grid_tvd": location_drift,
            "numeric_feature_psi": numeric_drift,
            "categorical_feature_tvd": categorical_drift,
        },
        "model_performance": {
            "evaluation_window": "latest 20% of labeled records",
            "macro_f1": float(f1_score(actual, predictions, average="macro", zero_division=0)),
            "macro_recall": float(
                recall_score(actual, predictions, average="macro", zero_division=0)
            ),
            "classes": labels,
            "confusion_matrix": matrix.tolist(),
            "false_negative_rate_by_class": false_negative_rates,
            "prediction_batch_latency_ms": round(prediction_seconds * 1000, 3),
            "prediction_latency_ms_per_row": round(
                prediction_seconds * 1000 / len(test), 6
            ),
        },
        "retraining_criteria": {
            "class_tvd_threshold": CLASS_DRIFT_THRESHOLD,
            "location_grid_tvd_threshold": DRIFT_THRESHOLD,
            "numeric_feature_psi_threshold": DRIFT_THRESHOLD,
            "fatal_false_negative_rate_threshold": FATAL_FNR_THRESHOLD,
        },
        "alerts": [],
        "retraining_reasons": [],
    }

    alerts = report["alerts"]
    retraining_reasons = report["retraining_reasons"]
    if class_drift > CLASS_DRIFT_THRESHOLD:
        reason = "Severity-class distribution drift exceeds the threshold."
        alerts.append(reason)
        retraining_reasons.append(reason)
    if location_drift > DRIFT_THRESHOLD:
        reason = "Accident-location grid drift exceeds the threshold."
        alerts.append(reason)
        retraining_reasons.append(reason)
    drifting_features = [
        name for name, value in numeric_drift.items() if value > DRIFT_THRESHOLD
    ]
    drifting_features.extend(
        name for name, value in categorical_drift.items() if value > DRIFT_THRESHOLD
    )
    if drifting_features:
        reason = (
            "Input feature drift exceeds the threshold for: "
            + ", ".join(sorted(drifting_features))
            + "."
        )
        alerts.append(reason)
        retraining_reasons.append(reason)
    fatal_fnr = false_negative_rates.get("1")
    if fatal_fnr is not None and fatal_fnr > FATAL_FNR_THRESHOLD:
        reason = (
            f"Fatal-severity false-negative rate ({fatal_fnr:.3f}) exceeds "
            f"{FATAL_FNR_THRESHOLD:.3f}."
        )
        alerts.append(reason)
        retraining_reasons.append(reason)
    if report["input_quality"]["duplicate_rows"]:
        reason = "Duplicate labeled records detected."
        alerts.append(reason)
        retraining_reasons.append(reason)

    metrics_url = os.getenv("INFERENCE_METRICS_URL")
    if metrics_url:
        report["api_runtime"] = fetch_api_metrics(metrics_url)
        if report["api_runtime"]["status"] == "unavailable":
            alerts.append("Configured inference API metrics endpoint is unavailable.")
        else:
            runtime_metrics = report["api_runtime"]["metrics"]
            failure_rate = runtime_metrics["request_failure_rate"]
            p95_latency = runtime_metrics["latency_p95_ms"]
            if failure_rate > API_FAILURE_RATE_THRESHOLD:
                alerts.append(
                    f"Inference API request-failure rate ({failure_rate:.3f}) exceeds "
                    f"{API_FAILURE_RATE_THRESHOLD:.3f}."
                )
            if p95_latency > API_P95_LATENCY_THRESHOLD_MS:
                alerts.append(
                    f"Inference API p95 latency ({p95_latency:.1f} ms) exceeds "
                    f"{API_P95_LATENCY_THRESHOLD_MS:.1f} ms."
                )
    else:
        report["api_runtime"] = {"status": "not_configured"}

    report["runtime_alert_thresholds"] = {
        "api_request_failure_rate": API_FAILURE_RATE_THRESHOLD,
        "api_p95_latency_ms": API_P95_LATENCY_THRESHOLD_MS,
    }
    report["retraining_recommended"] = bool(retraining_reasons)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nMonitoring report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
