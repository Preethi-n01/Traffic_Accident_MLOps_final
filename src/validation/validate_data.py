import pandas as pd
import os
from datetime import datetime


# ============================================================
# PATH CONFIGURATION
# ============================================================

STAGING_PATH = "data/staging"
CLEANED_PATH = "data/cleaned"
REJECTED_PATH = "data/rejected"
LOG_PATH = "logs"

os.makedirs(CLEANED_PATH, exist_ok=True)
os.makedirs(REJECTED_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

validation_logs = []


# ============================================================
# COLLISION DATA VALIDATION
# ============================================================

def validate_collision_data():

    print("\nProcessing collision data...")

    file = os.path.join(
        STAGING_PATH,
        "dft-road-casualty-statistics-collision-2025 (1).csv"
    )

    df = pd.read_csv(
        file,
        encoding="latin1",
        low_memory=False
    )

    original_count = len(df)

    rejected_records = []

    # --------------------------------------------------------
    # 1. Duplicate collision records
    # --------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=["collision_index"],
        keep="first"
    )

    duplicate_records = df[duplicate_mask].copy()

    for _, row in duplicate_records.iterrows():
        rejected_records.append({
            "collision_index": row["collision_index"],
            "rejection_reason": "Duplicate collision_index"
        })

    df = df[~duplicate_mask].copy()

    # --------------------------------------------------------
    # 2. Latitude validation
    # --------------------------------------------------------

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    invalid_latitude = (
        df["latitude"].isna()
        | (df["latitude"] < -90)
        | (df["latitude"] > 90)
    )

    invalid_latitude_records = df[invalid_latitude].copy()

    for _, row in invalid_latitude_records.iterrows():
        rejected_records.append({
            "collision_index": row["collision_index"],
            "rejection_reason": "Missing or invalid latitude"
        })

    df = df[~invalid_latitude].copy()

    # --------------------------------------------------------
    # 3. Longitude validation
    # --------------------------------------------------------

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    invalid_longitude = (
        df["longitude"].isna()
        | (df["longitude"] < -180)
        | (df["longitude"] > 180)
    )

    invalid_longitude_records = df[invalid_longitude].copy()

    for _, row in invalid_longitude_records.iterrows():
        rejected_records.append({
            "collision_index": row["collision_index"],
            "rejection_reason": "Missing or invalid longitude"
        })

    df = df[~invalid_longitude].copy()

    # --------------------------------------------------------
    # 4. Date validation
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    invalid_date = df["date"].isna()

    invalid_date_records = df[invalid_date].copy()

    for _, row in invalid_date_records.iterrows():
        rejected_records.append({
            "collision_index": row["collision_index"],
            "rejection_reason": "Missing or invalid date"
        })

    df = df[~invalid_date].copy()

    # --------------------------------------------------------
    # 5. Severity validation
    # --------------------------------------------------------

    valid_severities = [1, 2, 3]

    invalid_severity = ~df["collision_severity"].isin(
        valid_severities
    )

    invalid_severity_records = df[invalid_severity].copy()

    for _, row in invalid_severity_records.iterrows():
        rejected_records.append({
            "collision_index": row["collision_index"],
            "rejection_reason": "Invalid collision severity"
        })

    df = df[~invalid_severity].copy()

    # --------------------------------------------------------
    # 6. Save cleaned collision data
    # --------------------------------------------------------

    output_file = os.path.join(
        CLEANED_PATH,
        "collision_cleaned.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # 7. Save rejected records with reasons
    # --------------------------------------------------------

    rejected_df = pd.DataFrame(rejected_records)

    rejected_file = os.path.join(
        REJECTED_PATH,
        "collision_rejected.csv"
    )

    rejected_df.to_csv(
        rejected_file,
        index=False
    )

    # --------------------------------------------------------
    # 8. Validation statistics
    # --------------------------------------------------------

    cleaned_count = len(df)
    rejected_count = len(rejected_df)

    validation_logs.append({
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "dataset": "collision",
        "original_records": original_count,
        "cleaned_records": cleaned_count,
        "rejected_records": rejected_count,
        "rejection_rate_percent": round(
            (rejected_count / original_count) * 100,
            4
        ),
        "status": "SUCCESS"
    })

    print("Original records:", original_count)
    print("Cleaned records:", cleaned_count)
    print("Rejected records:", rejected_count)


# ============================================================
# VEHICLE DATA VALIDATION
# ============================================================

def validate_vehicle_data():

    print("\nProcessing vehicle data...")

    file = os.path.join(
        STAGING_PATH,
        "dft-road-casualty-statistics-vehicle-2025.csv"
    )

    df = pd.read_csv(
        file,
        encoding="latin1",
        low_memory=False
    )

    original_count = len(df)

    duplicate_mask = df.duplicated(
        subset=[
            "collision_index",
            "vehicle_reference"
        ],
        keep="first"
    )

    rejected = df[duplicate_mask].copy()

    df = df[~duplicate_mask].copy()

    # Save cleaned
    output_file = os.path.join(
        CLEANED_PATH,
        "vehicle_cleaned.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # Save rejected
    rejected_file = os.path.join(
        REJECTED_PATH,
        "vehicle_rejected.csv"
    )

    rejected.to_csv(
        rejected_file,
        index=False
    )

    cleaned_count = len(df)
    rejected_count = len(rejected)

    validation_logs.append({
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "dataset": "vehicle",
        "original_records": original_count,
        "cleaned_records": cleaned_count,
        "rejected_records": rejected_count,
        "rejection_rate_percent": round(
            (rejected_count / original_count) * 100,
            4
        ),
        "status": "SUCCESS"
    })

    print("Original records:", original_count)
    print("Cleaned records:", cleaned_count)
    print("Rejected records:", rejected_count)


# ============================================================
# CASUALTY DATA VALIDATION
# ============================================================

def validate_casualty_data():

    print("\nProcessing casualty data...")

    file = os.path.join(
        STAGING_PATH,
        "dft-road-casualty-statistics-casualty-2025.csv"
    )

    df = pd.read_csv(
        file,
        encoding="latin1",
        low_memory=False
    )

    original_count = len(df)

    duplicate_mask = df.duplicated(
        subset=[
            "collision_index",
            "vehicle_reference",
            "casualty_reference"
        ],
        keep="first"
    )

    rejected = df[duplicate_mask].copy()

    df = df[~duplicate_mask].copy()

    # Save cleaned
    output_file = os.path.join(
        CLEANED_PATH,
        "casualty_cleaned.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # Save rejected
    rejected_file = os.path.join(
        REJECTED_PATH,
        "casualty_rejected.csv"
    )

    rejected.to_csv(
        rejected_file,
        index=False
    )

    cleaned_count = len(df)
    rejected_count = len(rejected)

    validation_logs.append({
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "dataset": "casualty",
        "original_records": original_count,
        "cleaned_records": cleaned_count,
        "rejected_records": rejected_count,
        "rejection_rate_percent": round(
            (rejected_count / original_count) * 100,
            4
        ),
        "status": "SUCCESS"
    })

    print("Original records:", original_count)
    print("Cleaned records:", cleaned_count)
    print("Rejected records:", rejected_count)


# ============================================================
# MAIN PIPELINE
# ============================================================

print("=" * 60)
print("UK ROAD SAFETY DATA VALIDATION")
print("=" * 60)

validate_collision_data()
validate_vehicle_data()
validate_casualty_data()


# ============================================================
# SAVE VALIDATION LOG
# ============================================================

validation_log_df = pd.DataFrame(validation_logs)

validation_log_file = os.path.join(
    LOG_PATH,
    "validation_log.csv"
)

validation_log_df.to_csv(
    validation_log_file,
    index=False
)

print("\nValidation completed successfully.")
print(
    f"Validation log saved to: {validation_log_file}"
)