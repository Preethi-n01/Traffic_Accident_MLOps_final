# UK Traffic Accident Analytics and MLOps

## 1. Project Overview

This project develops a data pipeline and analytics system for UK road traffic accidents by combining public accident records with historical weather information.

The project focuses on data ingestion, validation, transformation, weather enrichment, analytical database loading, and interactive visualization.

The project also includes an experimental accident severity classification model, model evaluation, MLflow experiment tracking, a FastAPI prediction service, Docker containerization, data drift monitoring, and a model retraining workflow.

---

## 2. Objectives

- Integrate UK road accident and weather datasets.
- Build a repeatable ETL pipeline.
- Validate and clean accident records.
- Enrich accident data with historical weather information.
- Store the analytical dataset in PostgreSQL.
- Develop an interactive Streamlit dashboard.
- Provide analytical insights into accident severity, time, weather, road type, and location.
- Develop machine learning models for accident severity classification.
- Evaluate model performance using chronological training, validation, and test datasets.
- Track model training experiments using MLflow.
- Expose model predictions through a FastAPI service.
- Containerize the prediction API using Docker.
- Monitor changes in feature distributions using Population Stability Index (PSI).
- Implement a model retraining workflow with a retraining log.

---

## 3. Data Sources

### UK Road Safety Open Data

The accident dataset is obtained from the UK Department for Transport Road Safety Open Data.

The data includes collision, vehicle, and casualty information.

### Open-Meteo Historical Weather API

Historical weather information is obtained using the Open-Meteo Historical Weather API.

Weather attributes include temperature, precipitation, rain, snowfall, wind speed, and weather conditions.

More details are available in:

`docs/data_sources.md`

---

## 4. Project Architecture

The pipeline follows this flow:

Data Sources → Ingestion → Raw Data → Staging → Transformation & Validation → PostgreSQL → Streamlit Dashboard

The architecture diagram is available at:

`docs/architecture_diagram.png`

---

## 5. ETL Pipeline

The pipeline consists of the following stages:

1. Accident data ingestion
2. Data validation
3. Data transformation and cleaning
4. Weather data preparation and enrichment
5. Analytical dataset creation
6. PostgreSQL database loading
7. Dashboard visualization

The pipeline is orchestrated using Apache Airflow.

---

## 6. Airflow Pipeline

The Airflow DAG is:

`traffic_accident_pipeline`

The pipeline executes the following tasks:

- `ingest_data`
- `validate_data`
- `transform_data`
- `load_database`
- `verify_analytical_data`
- `pipeline_complete`

The DAG can be triggered from the Airflow web interface.

---

## 7. Database

PostgreSQL is used as the analytical database.

Main table:

`accident_weather_mart`

The analytical table contains 101,523 accident records with accident, location, time, road, and weather attributes.

The database schema is available at:

`sql/schema.sql`

---

## 8. Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard provides:

- Key accident indicators
- Accident severity distribution
- Accidents by hour
- Accidents by weekday
- Weather condition analysis
- Road type analysis
- Monthly accident trends
- Accident location map
- Analytical data preview
- Severity and weather filters

Dashboard source:

`dashboard/app.py`

---

---

## 9. Machine Learning Model

The project includes machine learning models for classifying road accidents into three severity categories:

* **Fatal:** Severity class 1
* **Serious:** Severity class 2
* **Slight:** Severity class 3

The dataset contains 101,523 records. A chronological split was used for training, validation, and testing.

| Dataset    | Records |
| ---------- | ------: |
| Training   |  66,738 |
| Validation |  17,194 |
| Testing    |  17,591 |

XGBoost and Balanced Random Forest models were evaluated. XGBoost was selected for the prediction API based on its validation results.

### Final Test Results

| Metric                  | Result |
| ----------------------- | -----: |
| Accuracy                | 56.37% |
| Macro F1-score          | 0.3900 |
| Fatal accident recall   | 41.30% |
| Serious accident recall | 47.21% |

The model has limited predictive performance, particularly for fatal accidents. Predictions are experimental and must not be used as the sole basis for safety-critical decisions.

The trained models and evaluation results are stored in the `models/` directory.

---

## 10. Experiment Tracking

MLflow is used to track XGBoost training experiments and evaluation metrics.

* **Training script:** `src/modeling/train_xgboost.py`
* **Experiment name:** `Traffic Accident Severity`

---

## 11. Prediction API

FastAPI exposes the trained XGBoost model through an HTTP API.

**API source:** `src/api/main.py`

Available endpoints:

* `GET /` — Returns API information.
* `GET /health` — Checks API and model status.
* `POST /predict` — Returns a predicted accident severity and class probabilities.

To start the API in a configured Python environment, run:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to access the interactive Swagger documentation.

---

## 12. Docker Deployment

The prediction API is containerized using Docker.

Build the Docker image from the project root:

```bash
docker build -t traffic-accident-api .
```

Run the container:

```bash
docker run -d --name traffic-accident-container -p 8000:8000 traffic-accident-api
```

Check the API health endpoint at `http://localhost:8000/health`.

The container includes the API, its dependencies, and the saved XGBoost model.

---

## 13. Data Drift Monitoring

The project includes a feature-distribution monitoring script based on the Population Stability Index (PSI).

**Script:** `src/monitoring/monitor_drift.py`

Run the monitoring script:

```bash
python src/monitoring/monitor_drift.py
```

The script compares feature distributions between the reference and current datasets and saves the results to `models/drift_report.json`.

The current monitoring run detected distribution shifts in `temperature_mean` and `weather_condition`. These shifts indicate changes in feature distributions and do not necessarily mean model performance has declined.

---

## 14. Model Retraining Workflow

A retraining workflow is provided to rerun the existing XGBoost training script and record successful runs.

**Script:** `src/modeling/retrain_model.py`

Run it using:


python src/modeling/retrain_model.py


The workflow uses the existing dataset and training configuration. Retraining history is saved to `models/retraining_log.json`.

Retraining is currently initiated manually. Drift detection does not automatically trigger retraining. The new model's evaluation results should be reviewed before deployment.

---

## 15. Data Validation

Validation checks include:

* Required collision identifiers
* Duplicate records
* Valid dates and times
* Valid latitude and longitude ranges
* Non-negative vehicle and casualty counts
* Valid accident severity values
* Valid speed limits
* Missing weather information
* Data type consistency
* Analytical row-count verification

Rejected records are stored separately in the `data/rejected/` directory.

Detailed validation rules are documented in `docs/data_dictionary.md`.

---

## 16. Project Structure

```text
Traffic_Accident_MLOps/
├── dags/
│   └── traffic_accident_pipeline.py
├── data/
│   ├── raw/
│   ├── staging/
│   ├── cleaned/
│   ├── rejected/
│   └── analytical/
├── dashboard/
│   └── app.py
├── docs/
│   ├── architecture_diagram.png
│   ├── dashboard_overview.png
│   ├── airflow_pipeline_success.png
│   ├── database_record_count.png
│   ├── data_sources.md
│   └── data_dictionary.md
├── models/
│   ├── accident_severity_pipeline.joblib
│   ├── accident_severity_xgboost.joblib
│   ├── evaluation_metrics.json
│   ├── xgboost_evaluation_metrics.json
│   ├── final_test_metrics.json
│   ├── drift_report.json
│   └── retraining_log.json
├── sql/
│   ├── schema.sql
│   └── analytical_queries.sql
├── src/
│   ├── api/
│   │   └── main.py
│   ├── ingestion/
│   ├── modeling/
│   │   ├── train_model.py
│   │   ├── train_xgboost.py
│   │   ├── evaluate_final_model.py
│   │   └── retrain_model.py
│   ├── monitoring/
│   │   └── monitor_drift.py
│   ├── transformation/
│   ├── validation/
│   └── load_to_database.py
├── Dockerfile
├── requirements-api.txt
├── requirements.txt
└── README.md
```

*Note: This structure highlights the main project files; other files and directories may also be present.*
