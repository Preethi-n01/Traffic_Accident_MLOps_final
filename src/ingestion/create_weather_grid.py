import pandas as pd
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/cleaned/collision_cleaned.csv"
OUTPUT_DIR = "data/staging"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "weather_grid.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD ACCIDENT DATA
# ============================================================

print("Loading collision data...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=["latitude", "longitude"],
    low_memory=False
)


# ============================================================
# CREATE GRID
# ============================================================

# Use approximately 0.5 degree grid cells.
#
# This gives a manageable number of representative
# locations across the accident-data coverage area.

LAT_STEP = 0.5
LON_STEP = 0.5

lat_min = df["latitude"].min()
lat_max = df["latitude"].max()

lon_min = df["longitude"].min()
lon_max = df["longitude"].max()


latitudes = pd.Series(
    range(
        int(lat_min / LAT_STEP),
        int(lat_max / LAT_STEP) + 1
    )
) * LAT_STEP

longitudes = pd.Series(
    range(
        int(lon_min / LON_STEP),
        int(lon_max / LON_STEP) + 1
    )
) * LON_STEP


# Create all combinations
grid = pd.MultiIndex.from_product(
    [latitudes, longitudes],
    names=["latitude", "longitude"]
).to_frame(index=False)


# Round coordinates
grid["latitude"] = grid["latitude"].round(2)
grid["longitude"] = grid["longitude"].round(2)


# ============================================================
# KEEP ONLY GRID POINTS NEAR ACCIDENT LOCATIONS
# ============================================================

# Instead of requesting weather for every possible
# UK grid cell, keep only cells that actually contain
# accident records.

df["latitude_grid"] = (
    df["latitude"] / LAT_STEP
).round() * LAT_STEP

df["longitude_grid"] = (
    df["longitude"] / LON_STEP
).round() * LON_STEP

df["latitude_grid"] = df["latitude_grid"].round(2)
df["longitude_grid"] = df["longitude_grid"].round(2)


used_grid = (
    df[
        [
            "latitude_grid",
            "longitude_grid"
        ]
    ]
    .drop_duplicates()
    .rename(
        columns={
            "latitude_grid": "latitude",
            "longitude_grid": "longitude"
        }
    )
)


# ============================================================
# SAVE GRID
# ============================================================

used_grid.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("WEATHER GRID CREATION")
print("=" * 60)

print(
    "Accident records:",
    len(df)
)

print(
    "Weather grid points:",
    len(used_grid)
)

print(
    "\nLatitude range:",
    lat_min,
    "to",
    lat_max
)

print(
    "Longitude range:",
    lon_min,
    "to",
    lon_max
)

print(
    "\nWeather grid saved to:",
    OUTPUT_FILE
)