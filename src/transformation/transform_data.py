import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

COLLISION_FILE = "data/cleaned/collision_cleaned.csv"
WEATHER_FILE = "data/staging/weather_data.csv"

OUTPUT_DIR = "data/analytical"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "accident_weather.csv"
)

# ============================================================
# MAIN TRANSFORMATION
# ============================================================

def main():

    print("=" * 70)
    print("ACCIDENT + WEATHER TRANSFORMATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load collision data
    # --------------------------------------------------------

    print("\nLoading collision data...")

    collision = pd.read_csv(
        COLLISION_FILE,
        low_memory=False
    )

    print("Collision records:", len(collision))

    # --------------------------------------------------------
    # 2. Load weather data
    # --------------------------------------------------------

    print("\nLoading weather data...")

    weather = pd.read_csv(
        WEATHER_FILE
    )

    print("Weather records:", len(weather))

    # --------------------------------------------------------
    # 3. Prepare collision date
    # --------------------------------------------------------

    collision["date"] = pd.to_datetime(
        collision["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 4. Create same spatial grid used for weather
    # --------------------------------------------------------

    collision["latitude_grid"] = (collision["latitude"]*2).round()/2
    collision["longitude_grid"] = (collision["longitude"]*2).round()/2

    weather["date"] = pd.to_datetime(
        weather["date"],
        errors="coerce"
    )

    weather["latitude_grid"] = (weather["latitude"]*2).round()/2
    weather["longitude_grid"] = (weather["longitude"]*2).round()/2

    # --------------------------------------------------------
    # 5. Remove duplicate weather keys
    # --------------------------------------------------------

    weather = weather.drop_duplicates(
        subset=[
            "date",
            "latitude_grid",
            "longitude_grid"
        ]
    )

    # --------------------------------------------------------
    # 6. Rename weather columns
    # --------------------------------------------------------

    weather = weather.rename(
        columns={
            "weather_code": "weather_code_api",
            "temperature_2m_mean": "temperature_mean",
            "temperature_2m_max": "temperature_max",
            "temperature_2m_min": "temperature_min",
            "precipitation_sum": "precipitation",
            "rain_sum": "rain",
            "snowfall_sum": "snowfall",
            "wind_speed_10m_max": "wind_speed_max"
        }
    )

    # --------------------------------------------------------
    # 7. Select weather columns
    # --------------------------------------------------------

    weather_columns = [
        "date",
        "latitude_grid",
        "longitude_grid",
        "weather_code_api",
        "temperature_mean",
        "temperature_max",
        "temperature_min",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_max"
    ]

    weather = weather[weather_columns]

    # --------------------------------------------------------
    # 8. Merge accident + weather
    # --------------------------------------------------------

    print("\nJoining accident and weather data...")

    original_count = len(collision)

    enriched = collision.merge(
        weather,
        on=[
            "date",
            "latitude_grid",
            "longitude_grid"
        ],
        how="left"
    )

    # --------------------------------------------------------
    # 9. Weather availability flag
    # --------------------------------------------------------

    enriched["weather_available"] = (
        enriched["weather_code_api"].notna()
    ).astype(int)

    # --------------------------------------------------------
    # 10. Weather condition category
    # --------------------------------------------------------

    def classify_weather(code):

        if pd.isna(code):
            return "Unknown"

        code = int(code)

        if code == 0:
            return "Clear"

        elif code in [1, 2, 3]:
            return "Cloudy"

        elif code in range(45, 49):
            return "Fog"

        elif code in range(51, 68):
            return "Rain"

        elif code in range(71, 78):
            return "Snow"

        elif code in range(80, 83):
            return "Rain Showers"

        elif code in range(85, 87):
            return "Snow Showers"

        elif code in range(95, 100):
            return "Thunderstorm"

        else:
            return "Other"

    enriched["weather_condition"] = (
        enriched["weather_code_api"]
        .apply(classify_weather)
    )

    # --------------------------------------------------------
    # 11. Extract time features
    # --------------------------------------------------------

    enriched["hour"] = pd.to_numeric(
        enriched["time"].astype(str).str[:2],
        errors="coerce"
    )

    enriched["year"] = enriched["date"].dt.year
    enriched["month"] = enriched["date"].dt.month
    enriched["day"] = enriched["date"].dt.day

    enriched["weekday"] = enriched["date"].dt.day_name()

    # --------------------------------------------------------
    # 12. Remove unnecessary temporary columns
    # --------------------------------------------------------

    enriched = enriched.drop(
        columns=["latitude_grid", "longitude_grid"],
        errors="ignore"
    )

    # --------------------------------------------------------
    # 13. Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 14. Save analytical dataset
    # --------------------------------------------------------

    enriched.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 15. Validation summary
    # --------------------------------------------------------

    weather_matches = enriched[
        "weather_available"
    ].sum()

    weather_missing = (
        len(enriched) - weather_matches
    )

    print("\n" + "=" * 70)
    print("TRANSFORMATION COMPLETED")
    print("=" * 70)

    print("\nOriginal accident records:", original_count)
    print("Final records:", len(enriched))

    print(
        "Weather matched:",
        weather_matches
    )

    print(
        "Weather unavailable:",
        weather_missing
    )

    print(
        "Weather match rate:",
        round(
            weather_matches / len(enriched) * 100,
            2
        ),
        "%"
    )

    print(
        "\nOutput saved to:",
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()