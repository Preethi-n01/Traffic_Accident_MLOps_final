
import os
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    accuracy_score
)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

DATA_PATH = "data/analytical/accident_weather.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR, "accident_severity_pipeline.joblib"
)

METRICS_PATH = os.path.join(
    MODEL_DIR, "evaluation_metrics.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(
    df["date"], errors="coerce"
)

df = df.dropna(
    subset=["date", "collision_severity"]
)

df = df.sort_values("date").reset_index(drop=True)

print("Dataset shape:", df.shape)

print("\nSeverity distribution:")
print(df["collision_severity"].value_counts().sort_index())

# --------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------

# Extract date-related features.
df["month_feature"] = df["date"].dt.month
df["day_feature"] = df["date"].dt.day
df["weekday_feature"] = df["date"].dt.dayofweek

# Convert time strings such as 22:05 into numeric features.
if "time" in df.columns:
    parsed_time = pd.to_datetime(
        df["time"].astype("string"),
        format="%H:%M",
        errors="coerce"
    )

    df["time_hour"] = parsed_time.dt.hour
    df["time_minute"] = parsed_time.dt.minute

# --------------------------------------------------
# 4. Select target and features
# --------------------------------------------------

TARGET = "collision_severity"

# Exclude identifiers, raw date/time, and outcome-derived fields.
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
        column
        for column in excluded_columns
        if column in df.columns
    ],
    errors="ignore"
)

y = df[TARGET].astype(int)

# Treat coded categories as categorical features.
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
    column
    for column in categorical_candidates
    if column in X.columns
]

# Numeric hour and minute are handled by numeric preprocessing.
numeric_columns = [
    column
    for column in X.columns
    if column not in categorical_columns
]

# Keep only selected feature columns.
X = X[categorical_columns + numeric_columns]

print("\nNumber of features:", X.shape[1])
print("Categorical features:", len(categorical_columns))
print("Numeric features:", len(numeric_columns))

print("\nNumeric columns:")
print(numeric_columns)

# --------------------------------------------------
# 5. Chronological train/validation/test split
# --------------------------------------------------

train_mask = df["date"] < "2025-09-01"

validation_mask = (
    (df["date"] >= "2025-09-01")
    & (df["date"] < "2025-11-01")
)

test_mask = df["date"] >= "2025-11-01"

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[validation_mask]
y_val = y.loc[validation_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print("\nSplit sizes:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

for name, labels in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test),
]:
    print(f"\n{name} class distribution:")
    print(
        labels.value_counts(normalize=True)
        .sort_index()
        .round(4)
    )

# --------------------------------------------------
# 6. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

# --------------------------------------------------
# 7. Build Random Forest pipeline
# --------------------------------------------------

model = BalancedRandomForestClassifier(
    n_estimators=200,
    sampling_strategy="all",
    replacement=True,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# --------------------------------------------------
# 8. Train model
# --------------------------------------------------

print("\nTraining Balanced Random Forest...")

pipeline.fit(X_train, y_train)

print("Training completed.")

# --------------------------------------------------
# 9. Validation evaluation
# --------------------------------------------------

val_predictions = pipeline.predict(X_val)

val_accuracy = accuracy_score(
    y_val, val_predictions
)

val_macro_f1 = f1_score(
    y_val,
    val_predictions,
    average="macro",
    zero_division=0
)

val_fatal_recall = recall_score(
    y_val,
    val_predictions,
    labels=[1],
    average="macro",
    zero_division=0
)

print("\n========== VALIDATION RESULTS ==========")

print("Accuracy:", round(val_accuracy, 4))
print("Macro-F1:", round(val_macro_f1, 4))
print("Fatal recall:", round(val_fatal_recall, 4))

print("\nClassification report:")

print(classification_report(
    y_val,
    val_predictions,
    labels=[1, 2, 3],
    target_names=["Fatal", "Serious", "Slight"],
    zero_division=0
))

print("Confusion matrix (Fatal, Serious, Slight):")

print(confusion_matrix(
    y_val,
    val_predictions,
    labels=[1, 2, 3]
))

# --------------------------------------------------
# 10. Final test evaluation
# --------------------------------------------------

test_predictions = pipeline.predict(X_test)

test_accuracy = accuracy_score(
    y_test, test_predictions
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

test_fatal_recall = recall_score(
    y_test,
    test_predictions,
    labels=[1],
    average="macro",
    zero_division=0
)

print("\n========== TEST RESULTS ==========")

print("Accuracy:", round(test_accuracy, 4))
print("Macro-F1:", round(test_macro_f1, 4))
print("Fatal recall:", round(test_fatal_recall, 4))

print("\nClassification report:")

print(classification_report(
    y_test,
    test_predictions,
    labels=[1, 2, 3],
    target_names=["Fatal", "Serious", "Slight"],
    zero_division=0
))

print("Confusion matrix (Fatal, Serious, Slight):")

print(confusion_matrix(
    y_test,
    test_predictions,
    labels=[1, 2, 3]
))

# --------------------------------------------------
# 11. Save model pipeline and metrics
# --------------------------------------------------

joblib.dump(pipeline, MODEL_PATH)

metrics = {
    "model": "BalancedRandomForestClassifier",
    "random_state": 42,
    "features_count": int(X.shape[1]),
    "train_records": int(len(X_train)),
    "validation_records": int(len(X_val)),
    "test_records": int(len(X_test)),
    "validation_accuracy": float(val_accuracy),
    "validation_macro_f1": float(val_macro_f1),
    "validation_fatal_recall": float(val_fatal_recall),
    "test_accuracy": float(test_accuracy),
    "test_macro_f1": float(test_macro_f1),
    "test_fatal_recall": float(test_fatal_recall),
}

with open(
    METRICS_PATH, "w", encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

print("\nSaved model pipeline to:", MODEL_PATH)
print("Saved evaluation metrics to:", METRICS_PATH)
print("\nTraining and evaluation completed.")
