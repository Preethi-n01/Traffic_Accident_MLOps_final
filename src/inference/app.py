from collections import Counter, deque
from math import ceil
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    os.getenv(
        "ACCIDENT_MODEL_PATH",
        str(ROOT / "models" / "accident_severity_pipeline.joblib"),
    )
)

app = FastAPI(
    title="Traffic Accident Severity API",
    description="Predicts accident severity using the trained ML pipeline.",
    version="2.0.0",
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained inference pipeline not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
model.set_params(classifier__n_jobs=1)

request_count = 0
request_failure_count = 0
server_error_count = 0
status_counts: Counter[str] = Counter()
recent_latencies_ms: deque[float] = deque(maxlen=1000)
metrics_lock = Lock()


class AccidentInput(BaseModel):
    features: dict[str, float | int | str | None]


@app.middleware("http")
async def track_request_metrics(request: Request, call_next: Any):
    global request_count, request_failure_count, server_error_count
    started = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        if request.url.path != "/monitoring/metrics":
            elapsed_ms = (perf_counter() - started) * 1000
            with metrics_lock:
                request_count += 1
                status_counts[str(status_code)] += 1
                if status_code >= 400:
                    request_failure_count += 1
                if status_code >= 500:
                    server_error_count += 1
                recent_latencies_ms.append(elapsed_ms)


@app.get("/")
def home():
    return {
        "message": "Traffic Accident Severity API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/monitoring/metrics")
def monitoring_metrics():
    with metrics_lock:
        latency_values = sorted(recent_latencies_ms)
        count = request_count
        failures = request_failure_count
        server_errors = server_error_count
        counts = dict(status_counts)
    p95_index = max(0, ceil(0.95 * len(latency_values)) - 1)
    return {
        "requests_total": count,
        "request_failures_total": failures,
        "request_failure_rate": failures / count if count else 0.0,
        "server_errors_total": server_errors,
        "status_counts": counts,
        "latency_window_size": len(latency_values),
        "latency_mean_ms": (
            sum(latency_values) / len(latency_values) if latency_values else 0.0
        ),
        "latency_p95_ms": latency_values[p95_index] if latency_values else 0.0,
    }


@app.post("/predict")
def predict(data: AccidentInput):
    try:
        input_df = pd.DataFrame([data.features])
        prediction = int(model.predict(input_df)[0])
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="Features are invalid or do not match the trained model schema.",
        ) from exc

    severity_labels = {1: "Fatal", 2: "Serious", 3: "Slight"}
    return {
        "predicted_severity": prediction,
        "severity_label": severity_labels.get(prediction, "Unknown"),
    }
