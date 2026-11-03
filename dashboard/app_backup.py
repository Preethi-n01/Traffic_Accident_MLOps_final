
from pathlib import Path
import json
import os

import joblib
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "accident_severity_pipeline.joblib"
DATA_PATH = ROOT / "data" / "analytical" / "accident_weather.csv"
REPORT_PATH = ROOT / "reports" / "evaluation.json"
MONITORING_REPORT_PATH = ROOT / "reports" / "monitoring.json"
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "").rstrip("/")

st.set_page_config(
    page_title="Traffic Accident Analytics",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Traffic Accident Severity Analytics")
st.caption("Machine Learning • MLOps • Weather-Aware Accident Analysis")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run src/modeling/train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Overview
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Accidents", f"{len(df):,}")
    c2.metric("Features", str(df.shape[1] - 1))
    c3.metric("Year", str(df["collision_year"].mode().iloc[0]))

    if "collision_severity" in df.columns:
        st.subheader("Accident Severity Distribution")
        severity_counts = df["collision_severity"].value_counts().sort_index()
        severity_counts.index = [
            {1: "Fatal", 2: "Serious", 3: "Slight"}.get(int(x), str(x))
            for x in severity_counts.index
        ]
        st.bar_chart(severity_counts)

    if REPORT_PATH.exists():
        import json

        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        st.subheader("Model Evaluation")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c2.metric("Macro F1-Score", f"{metrics['macro_f1']:.2%}")
        c3.metric("Macro Recall", f"{metrics['macro_recall']:.2%}")

    if MONITORING_REPORT_PATH.exists():
        with open(MONITORING_REPORT_PATH, "r", encoding="utf-8") as f:
            monitoring = json.load(f)

        st.subheader("Model and Data Monitoring")
        status_column, class_column, location_column, fnr_column = st.columns(4)
        status_column.metric(
            "Retraining",
            "Recommended" if monitoring.get("retraining_recommended") else "Not triggered",
        )
        drift = monitoring.get("drift", {})
        class_column.metric("Severity class drift (TVD)", f"{drift.get('class_tvd', 0):.3f}")
        location_column.metric(
            "Location grid drift (TVD)", f"{drift.get('location_grid_tvd', 0):.3f}"
        )
        fatal_fnr = (
            monitoring.get("model_performance", {})
            .get("false_negative_rate_by_class", {})
            .get("1")
        )
        fnr_column.metric(
            "Fatal false-negative rate",
            f"{fatal_fnr:.1%}" if fatal_fnr is not None else "Unavailable",
        )
        if monitoring.get("alerts"):
            for alert in monitoring["alerts"]:
                st.warning(alert)

    if INFERENCE_API_URL:
        try:
            api_metrics = requests.get(
                f"{INFERENCE_API_URL}/monitoring/metrics", timeout=2
            )
            api_metrics.raise_for_status()
            runtime = api_metrics.json()
            st.subheader("Inference API Runtime")
            c1, c2, c3 = st.columns(3)
            c1.metric("Requests", runtime["requests_total"])
            c2.metric("Mean latency", f"{runtime['latency_mean_ms']:.2f} ms")
            c3.metric("Request failures", runtime["request_failures_total"])
        except requests.RequestException as exc:
            st.warning(f"Inference API metrics are unavailable: {exc}")

# Prediction form
st.divider()
st.subheader("Predict Accident Severity")
st.write("Enter the accident details below to generate a prediction.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        speed_limit = st.selectbox("Speed Limit (mph)", [20, 30, 40, 50, 60, 70], index=2)
        number_of_vehicles = st.number_input("Number of Vehicles", 1, 20, 2)
        number_of_casualties = st.number_input("Number of Casualties", 1, 50, 1)
        road_type = st.number_input("Road Type Code", 1, 9, 3)
        first_road_class = st.number_input("First Road Class", 1, 6, 3)
        second_road_class = st.number_input("Second Road Class", 0, 6, 3)
        day_of_week = st.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7], index=2)

    with col2:
        hour = st.slider("Hour of Accident", 0, 23, 12)
        month = st.selectbox("Month", list(range(1, 13)), index=5)
        latitude = st.number_input("Latitude", value=54.70743, format="%.5f")
        longitude = st.number_input("Longitude", value=-1.23770, format="%.5f")
        light_conditions = st.number_input("Light Conditions Code", 1, 7, 1)
        weather_conditions = st.number_input("Weather Conditions Code", 1, 9, 1)
        road_surface_conditions = st.number_input("Road Surface Code", 1, 9, 1)

    with col3:
        temperature_mean = st.number_input("Mean Temperature (°C)", value=10.0)
        temperature_max = st.number_input("Maximum Temperature (°C)", value=14.0)
        temperature_min = st.number_input("Minimum Temperature (°C)", value=6.0)
        precipitation = st.number_input("Precipitation (mm)", min_value=0.0, value=0.0)
        rain = st.number_input("Rain (mm)", min_value=0.0, value=0.0)
        snowfall = st.number_input("Snowfall (cm)", min_value=0.0, value=0.0)
        wind_speed_max = st.number_input("Maximum Wind Speed", min_value=0.0, value=15.0)

    submitted = st.form_submit_button("Predict Severity")

if submitted:
    # Start with valid dataset defaults so all model features are supplied.
    row = pd.read_csv(DATA_PATH, nrows=1)
    row = row.drop(
        columns=[
            "collision_severity",
            "enhanced_severity_collision",
            "collision_injury_based",
            "collision_adjusted_severity_serious",
            "collision_adjusted_severity_slight",
            "collision_index",
            "collision_ref_no",
            "lsoa_of_accident_location",
            "date",
            "time",
        ],
        errors="ignore",
    )

    # Override the fields provided by the user.
    values = {
        "speed_limit": speed_limit,
        "number_of_vehicles": number_of_vehicles,
        "number_of_casualties": number_of_casualties,
        "road_type": road_type,
        "first_road_class": first_road_class,
        "second_road_class": second_road_class,
        "day_of_week": day_of_week,
        "hour": hour,
        "month": month,
        "latitude": latitude,
        "longitude": longitude,
        "light_conditions": light_conditions,
        "weather_conditions": weather_conditions,
        "road_surface_conditions": road_surface_conditions,
        "temperature_mean": temperature_mean,
        "temperature_max": temperature_max,
        "temperature_min": temperature_min,
        "precipitation": precipitation,
        "rain": rain,
        "snowfall": snowfall,
        "wind_speed_max": wind_speed_max,
    }

    for column, value in values.items():
        if column in row.columns:
            row.loc[0, column] = value

    try:
        prediction = int(model.predict(row)[0])
        labels = {1: "Fatal", 2: "Serious", 3: "Slight"}
        label = labels.get(prediction, "Unknown")

        st.subheader("Prediction Result")
        st.metric("Predicted Severity", label, f"Class {prediction}")
        st.warning(
            "This is a machine-learning estimate, not a reliable real-world "
            "safety assessment. Do not use it alone for emergency decisions."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
