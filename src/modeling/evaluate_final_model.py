
import os
import json
import joblib
import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = "data/analytical/accident_weather.csv"
MODEL_PATH = "models/accident_severity_xgboost.joblib"
METRICS_PATH = "models/final_test_metrics.json"

# --------------------------------------------------
# 2. Load dataset and saved model
# --------------------------------------------------

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "collision_severity"])
df = df.sort_values("date").reset_index(drop=True)

print("Dataset shape:", df.shape)

print("\nLoading saved XGBoost model...")
pipeline = joblib.load(MODEL_PATH)

# --------------------------------------------------
# 3. Recreate the features used during training
# --------------------------------------------------

df["month_feature"] = df["date"].dt.month
df["day_feature"] = df["date"].dt.day
df["weekday_feature"] = df["date"].dt.dayofweek

if "time" in df.columns:
    parsed_time = pd.to_datetime(
        df["time"].astype("string"),
        format="%H:%M",
        errors="coerce"
    )

    df["time_hour"] = parsed_time.dt.hour
    df["time_minute"] = parsed_time.dt.minute

TARGET = "collision_severity"

excluded_columns = [
    TARGET,
    "collision_index",
    "collision_ref_no",
    "date",
    "time",
    "collision_year",
    "year",
    "number_of_casualties",
    "enhanced_severity_collision",
    "collision_injury_based",
    "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight",
    "lsoa_of_accident_location",
    "local_authority_ons_district",
    "local_authority_highway_current",
]

X = df.drop(
    columns=[
        col for col in excluded_columns
        if col in df.columns
    ],
    errors="ignore"
)

y = df[TARGET].astype(int)

categorical_candidates = [
    "police_force",
    "day_of_week",
    "local_authority_district",
    "local_authority_highway",
    "first_road_class",
    "first_road_number",
    "road_type",
    "speed_limit",
    "junction_detail_historic",
    "junction_detail",
    "junction_control",
    "second_road_class",
    "second_road_number",
    "pedestrian_crossing_human_control",
    "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic",
    "pedestrian_crossing",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "special_conditions_at_site",
    "carriageway_hazards_historic",
    "carriageway_hazards",
    "urban_or_rural_area",
    "did_police_officer_attend_scene_of_accident",
    "trunk_road_flag",
    "weather_code_api",
    "weather_condition",
    "weather_available",
    "month_feature",
    "day_feature",
    "weekday_feature",
    "hour",
    "month",
    "day",
    "weekday",
]

categorical_columns = [
    col for col in categorical_candidates
    if col in X.columns
]

numeric_columns = [
    col for col in X.columns
    if col not in categorical_columns
]

X = X[categorical_columns + numeric_columns]

# --------------------------------------------------
# 4. Select the untouched chronological test period
# --------------------------------------------------

test_mask = df["date"] >= "2025-11-01"

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print("\nTest records:", len(X_test))
print("Test period: November-December 2025")

if len(X_test) == 0:
    raise ValueError("No test records found for November-December 2025.")

# --------------------------------------------------
# 5. Evaluate the saved model
# --------------------------------------------------

print("\nGenerating predictions...")

# The model was trained with labels 1, 2, 3 encoded as 0, 1, 2.
y_pred = pipeline.predict(X_test).astype(int) + 1

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    labels=[1, 2, 3],
    average="macro",
    zero_division=0
)

fatal_recall = recall_score(
    y_test,
    y_pred,
    labels=[1],
    average="macro",
    zero_division=0
)

serious_recall = recall_score(
    y_test,
    y_pred,
    labels=[2],
    average="macro",
    zero_division=0
)

print("\n========== FINAL TEST RESULTS ==========")
print("Accuracy:", round(accuracy, 4))
print("Macro-F1:", round(macro_f1, 4))
print("Fatal recall:", round(fatal_recall, 4))
print("Serious recall:", round(serious_recall, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[1, 2, 3],
        target_names=["Fatal", "Serious", "Slight"],
        zero_division=0
    )
)

print("Confusion Matrix (Fatal, Serious, Slight):")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[1, 2, 3]
    )
)

# --------------------------------------------------
# 6. Save final test metrics
# --------------------------------------------------

metrics = {
    "model": "XGBClassifier",
    "test_period": "2025-11-01 to 2025-12-31",
    "test_records": int(len(X_test)),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "fatal_recall": float(fatal_recall),
    "serious_recall": float(serious_recall),
}

os.makedirs("models", exist_ok=True)

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

# --------------------------------------------------
# 7. Log final test metrics in MLflow
# --------------------------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Traffic Accident Severity")

with mlflow.start_run(run_name="XGBoost_Final_Test_Evaluation"):
    mlflow.log_params({
        "model": "XGBClassifier",
        "test_period": metrics["test_period"],
        "test_records": metrics["test_records"],
    })

    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_fatal_recall": fatal_recall,
        "test_serious_recall": serious_recall,
    })

    mlflow.log_artifact(METRICS_PATH)

print("\nSaved test metrics to:", METRICS_PATH)
print("Final test metrics logged to MLflow.")
