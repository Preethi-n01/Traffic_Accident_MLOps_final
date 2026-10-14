
import os
import joblib
import pandas as pd

from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_PATH = "models/accident_severity_xgboost.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Run the training script first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Traffic Accident Severity Prediction API",
    description=(
        "API for estimating accident severity using "
        "a trained XGBoost classification pipeline."
    ),
    version="1.0.0"
)

# These are the labels used by the original dataset.
SEVERITY_LABELS = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}


# --------------------------------------------------
# 2. Request schema
# --------------------------------------------------

class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description=(
            "Accident feature values as key-value pairs. "
            "Use the same feature names as the training dataset."
        )
    )


# --------------------------------------------------
# 3. Prepare incoming features
# --------------------------------------------------

def prepare_features(features: dict[str, Any]) -> pd.DataFrame:
    """
    Prepare one input record using the feature names
    expected by the saved training pipeline.
    """

    row = dict(features)

    # Recreate the date-based features used during training.
    if row.get("date"):
        try:
            accident_date = pd.to_datetime(
                row["date"],
                errors="raise"
            )

            row["month_feature"] = accident_date.month
            row["day_feature"] = accident_date.day
            row["weekday_feature"] = accident_date.dayofweek

        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422,
                detail="Invalid date. Use YYYY-MM-DD."
            )

    # Recreate time-based features.
    if row.get("time"):
        try:
            parsed_time = datetime.strptime(
                str(row["time"]),
                "%H:%M"
            )

            row["time_hour"] = parsed_time.hour
            row["time_minute"] = parsed_time.minute

        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="Invalid time. Use HH:MM format."
            )

    # Match the feature columns used to train the saved pipeline.
    expected_features = list(model.feature_names_in_)

    input_row = {
        feature: row.get(feature)
        for feature in expected_features
    }

    return pd.DataFrame(
        [input_row],
        columns=expected_features
    )


# --------------------------------------------------
# 4. Health-check endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Traffic Accident Severity Prediction API",
        "status": "running",
        "model": "XGBoost",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# --------------------------------------------------
# 5. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        input_data = prepare_features(request.features)

        # The training script encoded labels 1, 2, 3
        # as 0, 1, 2 for XGBoost.
        encoded_prediction = int(
            model.predict(input_data)[0]
        )

        severity_code = encoded_prediction + 1

        if severity_code not in SEVERITY_LABELS:
            raise HTTPException(
                status_code=500,
                detail="The model returned an unknown class."
            )

        # Obtain probabilities for all three classes.
        probabilities = model.predict_proba(input_data)[0]

        return {
            "predicted_severity": SEVERITY_LABELS[severity_code],
            "severity_code": severity_code,
            "class_probabilities": {
                SEVERITY_LABELS[index + 1]: round(
                    float(probabilities[index]), 4
                )
                for index in range(len(probabilities))
            },
            "model": "XGBoost",
            "note": (
                "Experimental prediction. This model is not "
                "validated for safety-critical decisions."
            )
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
