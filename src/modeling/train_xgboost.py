import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
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
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "accident_severity_xgboost.joblib"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_evaluation_metrics.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Traffic Accident Severity")

# --------------------------------------------------
# 2. Load and sort the dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
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

# --------------------------------------------------
# 4. Select features and target
# --------------------------------------------------

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
        column
        for column in excluded_columns
        if column in df.columns
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
    column
    for column in categorical_candidates
    if column in X.columns
]

numeric_columns = [
    column
    for column in X.columns
    if column not in categorical_columns
]

X = X[categorical_columns + numeric_columns]

print("\nNumber of features:", X.shape[1])
print("Categorical features:", len(categorical_columns))
print("Numeric features:", len(numeric_columns))

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

print("\nSplit sizes:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", int(test_mask.sum()))

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
# 7. XGBoost model
# --------------------------------------------------

model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Convert labels 1, 2, 3 to 0, 1, 2 for XGBoost.
y_train_encoded = y_train - 1

# Give more training weight to underrepresented classes.
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train_encoded
)

# --------------------------------------------------
# 8. Train and track the experiment
# --------------------------------------------------

print("\nTraining XGBoost...")

with mlflow.start_run(run_name="XGBoost_Balanced_Weights"):

    mlflow.log_params({
        "model": "XGBoost",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weighting": "balanced_sample_weights",
        "random_state": 42,
        "features_count": X.shape[1],
        "train_records": len(X_train),
        "validation_records": len(X_val),
    })

    pipeline.fit(
        X_train,
        y_train_encoded,
        classifier__sample_weight=sample_weights
    )

    print("Training completed.")

    # --------------------------------------------------
    # 9. Validation evaluation
    # --------------------------------------------------

    val_predictions = pipeline.predict(X_val).astype(int) + 1

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    val_macro_f1 = f1_score(
        y_val,
        val_predictions,
        labels=[1, 2, 3],
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

    val_serious_recall = recall_score(
        y_val,
        val_predictions,
        labels=[2],
        average="macro",
        zero_division=0
    )

    mlflow.log_metrics({
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
        "validation_fatal_recall": val_fatal_recall,
        "validation_serious_recall": val_serious_recall,
    })

    print("\n========== VALIDATION RESULTS ==========")
    print("Accuracy:", round(val_accuracy, 4))
    print("Macro-F1:", round(val_macro_f1, 4))
    print("Fatal recall:", round(val_fatal_recall, 4))
    print("Serious recall:", round(val_serious_recall, 4))

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

    # Save this experiment separately.
    joblib.dump(pipeline, MODEL_PATH)

    metrics = {
        "model": "XGBClassifier",
        "random_state": 42,
        "features_count": int(X.shape[1]),
        "train_records": int(len(X_train)),
        "validation_records": int(len(X_val)),
        "test_records": int(test_mask.sum()),
        "validation_accuracy": float(val_accuracy),
        "validation_macro_f1": float(val_macro_f1),
        "validation_fatal_recall": float(val_fatal_recall),
        "validation_serious_recall": float(val_serious_recall),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    mlflow.log_artifact(METRICS_PATH)

    mlflow.sklearn.log_model(
    pipeline,
    name="xgboost_pipeline",
    skops_trusted_types=[
        "numpy.dtype",
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier"
    ]
)

    print("\nMLflow run ID:", mlflow.active_run().info.run_id)

print("\nSaved model to:", MODEL_PATH)
print("Saved metrics to:", METRICS_PATH)
print("Experiment tracking completed.")