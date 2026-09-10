# Dataset Source Information

## 1. UK Road Safety Open Data

**Source:** UK Department for Transport (DfT)

The primary accident dataset used in this project is the UK Road Safety Open Data published by the Department for Transport. It provides record-level information about personal injury road collisions in Great Britain, together with vehicle and casualty information. The project uses the 2025 collision, vehicle, and casualty datasets.

**Access:** The datasets can be downloaded as CSV files from the official DfT Road Safety Open Data page.

**Usage in this project:**

* Collision records provide accident date, time, location, severity, road type, speed limit, junction information, and environmental conditions.
* Vehicle records provide vehicle-related information.
* Casualty records provide information about casualties resulting from collisions.
* These datasets form the primary accident data layer of the pipeline.

**Licence:** UK Open Government Licence (OGL).

## 2. Open-Meteo Historical Weather API

**Source:** Open-Meteo

Open-Meteo Historical Weather API is used to enrich accident records with historical weather information based on accident location and time. The API provides historical variables including temperature, precipitation, rain, snowfall, wind speed, and weather codes.

**Access:** Weather data is obtained programmatically through the Open-Meteo Historical Weather API using Python requests.

**Usage in this project:**

* Historical weather data is requested for the locations and dates represented in the accident dataset.
* Weather information is joined with accident records using location and time-related information.
* Weather variables are included in the analytical accident-weather dataset.

## 3. Data Processing

The downloaded DfT CSV files are retained in the project's raw data layer. Weather API responses are stored in the staging layer before transformation and joining.

The resulting processed data is stored in the analytical layer as:

`data/analytical/accident_weather.csv`

The final analytical dataset contains **101,523 accident records** and combines accident-related variables with weather information.

## 4. Access Instructions

To reproduce the accident dataset:

1. Visit the official DfT Road Safety Open Data page.
2. Download the required 2025 collision, vehicle, and casualty CSV files.
3. Place the files in the project's raw data directory.
4. Run the ingestion and ETL pipeline through the provided Apache Airflow DAG.

To reproduce the weather enrichment:

1. Access the Open-Meteo Historical Weather API.
2. Generate weather requests based on the accident locations and dates.
3. Store the returned data in the staging layer.
4. Run the transformation pipeline to combine accident and weather information.
