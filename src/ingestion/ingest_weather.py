import pandas as pd
import requests
import os
import time
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/staging/weather_grid.csv"

OUTPUT_DIR = "data/staging"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "weather_data.csv"
)

WEATHER_LOG_FILE = "logs/weather_ingestion_log.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


# Open-Meteo historical weather API
API_URL = "https://archive-api.open-meteo.com/v1/archive"


# ============================================================
# LOAD WEATHER GRID
# ============================================================

print("=" * 60)
print("OPEN-METEO WEATHER INGESTION")
print("=" * 60)

print("\nLoading weather grid...")

grid = pd.read_csv(INPUT_FILE)

print("Weather grid points:", len(grid))


# ============================================================
# WEATHER VARIABLES
# ============================================================

DAILY_VARIABLES = [
    "weather_code",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max"
]


# ============================================================
# STORAGE
# ============================================================

all_weather = []
ingestion_logs = []


# ============================================================
# PROCESS EACH GRID POINT
# ============================================================

for index, row in grid.iterrows():

    latitude = row["latitude"]
    longitude = row["longitude"]

    print(
        f"\n[{index + 1}/{len(grid)}] "
        f"Fetching weather for "
        f"({latitude}, {longitude})"
    )

    start_time = datetime.now()

    try:

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "daily": ",".join(DAILY_VARIABLES),
            "timezone": "Europe/London"
        }

        response = requests.get(
            API_URL,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        # ----------------------------------------------------
        # Extract daily weather
        # ----------------------------------------------------

        if "daily" not in data:
            raise ValueError(
                "No daily weather data returned"
            )

        daily = data["daily"]

        weather_df = pd.DataFrame(daily)

        # Add grid location
        weather_df["latitude"] = latitude
        weather_df["longitude"] = longitude

        all_weather.append(weather_df)

        # ----------------------------------------------------
        # Log success
        # ----------------------------------------------------

        ingestion_logs.append({
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "latitude": latitude,
            "longitude": longitude,
            "status": "SUCCESS",
            "rows": len(weather_df),
            "error": ""
        })

        print(
            f"SUCCESS - {len(weather_df)} daily records"
        )

    except Exception as e:

        ingestion_logs.append({
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "latitude": latitude,
            "longitude": longitude,
            "status": "FAILED",
            "rows": 0,
            "error": str(e)
        })

        print(
            f"FAILED - {str(e)}"
        )

    # Small delay between requests
    time.sleep(0.2)


# ============================================================
# COMBINE WEATHER DATA
# ============================================================

if all_weather:

    weather_data = pd.concat(
        all_weather,
        ignore_index=True
    )

    # Rename time column
    weather_data = weather_data.rename(
        columns={
            "time": "date"
        }
    )

    # Save weather data
    weather_data.to_csv(
        OUTPUT_FILE,
        index=False
    )

else:

    print(
        "\nNo weather data was successfully retrieved."
    )


# ============================================================
# SAVE INGESTION LOG
# ============================================================

log_df = pd.DataFrame(
    ingestion_logs
)

log_df.to_csv(
    WEATHER_LOG_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("WEATHER INGESTION COMPLETED")
print("=" * 60)

print(
    "Successful locations:",
    (log_df["status"] == "SUCCESS").sum()
)

print(
    "Failed locations:",
    (log_df["status"] == "FAILED").sum()
)

if all_weather:

    print(
        "Weather records:",
        len(weather_data)
    )

    print(
        "Weather data saved to:",
        OUTPUT_FILE
    )

print(
    "Weather ingestion log saved to:",
    WEATHER_LOG_FILE
)