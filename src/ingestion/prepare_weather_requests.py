import pandas as pd
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/cleaned/collision_cleaned.csv"
OUTPUT_DIR = "data/staging"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "weather_requests.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD CLEANED COLLISION DATA
# ============================================================

print("Loading cleaned collision data...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Accident records:", len(df))


# ============================================================
# PREPARE DATE AND TIME
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# Extract accident hour
df["hour"] = pd.to_numeric(
    df["time"].astype(str).str.split(":").str[0],
    errors="coerce"
)


# ============================================================
# CREATE SPATIAL GRID
# ============================================================

# Group nearby accidents into approximately 10 km grid cells.
#
# 0.1 degree latitude/longitude is approximately 10 km.
#
# This prevents us from making one weather API request
# for every individual accident location.

df["latitude_grid"] = (
    df["latitude"] / 0.1
).round() * 0.1

df["longitude_grid"] = (
    df["longitude"] / 0.1
).round() * 0.1


# Round floating point values
df["latitude_grid"] = df["latitude_grid"].round(1)
df["longitude_grid"] = df["longitude_grid"].round(1)


# ============================================================
# CREATE UNIQUE WEATHER REQUESTS
# ============================================================

weather_requests = df[
    [
        "date",
        "hour",
        "latitude_grid",
        "longitude_grid"
    ]
].drop_duplicates()


# Remove incomplete records
weather_requests = weather_requests.dropna(
    subset=[
        "date",
        "hour",
        "latitude_grid",
        "longitude_grid"
    ]
)


# ============================================================
# SAVE WEATHER REQUEST DATASET
# ============================================================

weather_requests.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

original_count = len(df)
request_count = len(weather_requests)

reduction = (
    (1 - request_count / original_count) * 100
)

print("\n" + "=" * 60)
print("WEATHER REQUEST PREPARATION")
print("=" * 60)

print(
    "Original accident records:",
    original_count
)

print(
    "Unique weather requests:",
    request_count
)

print(
    "Reduction:",
    round(reduction, 2),
    "%"
)

print(
    "\nWeather request file saved to:",
    OUTPUT_FILE
)