# UK Traffic Accident Analytics and MLOps

## 1. Project Overview

This project develops a data pipeline and analytics system for UK road traffic accidents by combining public accident records with historical weather information.

The project focuses on data ingestion, validation, transformation, weather enrichment, analytical database loading, and interactive visualization.

The machine learning and MLOps components for accident severity prediction are planned for Part 2.

---

## 2. Objectives

- Integrate UK road accident and weather datasets.
- Build a repeatable ETL pipeline.
- Validate and clean accident records.
- Enrich accident data with historical weather information.
- Store the analytical dataset in PostgreSQL.
- Develop an interactive Streamlit dashboard.
- Provide analytical insights into accident severity, time, weather, road type, and location.
- Prepare the data pipeline for future accident severity prediction.

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

## 9. Data Validation

Validation checks include:

- Required collision identifiers
- Duplicate records
- Valid dates and times
- Valid latitude and longitude ranges
- Non-negative vehicle and casualty counts
- Valid accident severity values
- Valid speed limits
- Missing weather information
- Data type consistency
- Analytical row-count verification

Rejected records are stored separately in the `data/rejected/` directory.

Detailed validation rules are documented in:

`docs/data_dictionary.md`

---

## 10. Project Structure

```text
Traffic_Accident_MLOps/
│
├── dags/
│   └── traffic_accident_pipeline.py
│
├── data/
│   ├── raw/
│   ├── staging/
│   ├── cleaned/
│   ├── rejected/
│   └── analytical/
│
├── dashboard/
│   └── app.py
│
├── docs/
│   ├── architecture_diagram.png
│   ├── dashboard_overview.png
│   ├── airflow_pipeline_success.png
│   ├── database_record_count.png
│   ├── data_sources.md
│   └── data_dictionary.md
│
├── sql/
│   ├── schema.sql
│   └── analytical_queries.sql
│
├── src/
│   ├── ingestion/
│   ├── transformation/
│   ├── validation/
│   └── load_to_database.py
│
├── requirements.txt
└── README.md