import pandas as pd
import os
from datetime import datetime

RAW_PATH = "data/raw"
STAGING_PATH = "data/staging"
LOG_PATH = "logs"

os.makedirs(STAGING_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)


def ingest_file(filename, source_name):
    """Read a raw CSV and save it to the staging layer."""

    file_path = os.path.join(RAW_PATH, filename)

    start_time = datetime.now()

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(
            file_path,
            encoding="latin1",
            low_memory=False
        )

        # Save unchanged data to staging
        staging_path = os.path.join(
            STAGING_PATH,
            filename
        )

        df.to_csv(
            staging_path,
            index=False
        )

        end_time = datetime.now()

        log_entry = {
            "extraction_date": start_time.strftime("%Y-%m-%d"),
            "extraction_time": start_time.strftime("%H:%M:%S"),
            "source": source_name,
            "file": filename,
            "status": "SUCCESS",
            "row_count": len(df),
            "column_count": len(df.columns),
            "completed_at": end_time.strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"SUCCESS: {filename}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")

        return log_entry

    except Exception as e:

        log_entry = {
            "extraction_date": start_time.strftime("%Y-%m-%d"),
            "extraction_time": start_time.strftime("%H:%M:%S"),
            "source": source_name,
            "file": filename,
            "status": "FAILED",
            "row_count": 0,
            "column_count": 0,
            "completed_at": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "error": str(e)
        }

        print(f"FAILED: {filename}")
        print(f"Error: {e}")

        return log_entry


# Dataset files
datasets = [
    (
        "dft-road-casualty-statistics-collision-2025 (1).csv",
        "UK Department for Transport - Road Safety Data"
    ),
    (
        "dft-road-casualty-statistics-vehicle-2025.csv",
        "UK Department for Transport - Road Safety Data"
    ),
    (
        "dft-road-casualty-statistics-casualty-2025.csv",
        "UK Department for Transport - Road Safety Data"
    )
]


# Run ingestion
logs = []

for filename, source in datasets:
    result = ingest_file(filename, source)
    logs.append(result)


# Save ingestion log
log_df = pd.DataFrame(logs)

log_file = os.path.join(
    LOG_PATH,
    "ingestion_log.csv"
)

log_df.to_csv(
    log_file,
    index=False
)

print("\nIngestion completed.")
print(f"Log saved to: {log_file}")