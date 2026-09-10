import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

CSV_PATH = "data/analytical/accident_weather.csv"

DB_CONFIG = {
    "host": "172.24.144.1",
    "port": 8080,
    "database": "traffic_accidents",
    "user": "postgres",
    "password": "traffic",
}

TABLE_NAME = "accident_weather_mart"


def clean_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def main():
    print("Reading analytical CSV...")

    df = pd.read_csv(CSV_PATH)

    print(f"Records to load: {len(df)}")

    columns = list(df.columns)

    rows = [
        tuple(clean_value(value) for value in row)
        for row in df.itertuples(index=False, name=None)
    ]

    print("Connecting to PostgreSQL...")

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        print(f"Refreshing table: {TABLE_NAME}")

        # Full refresh so Airflow reruns do not create duplicate records
        cursor.execute(f"TRUNCATE TABLE {TABLE_NAME}")

        column_names = ", ".join(f'"{col}"' for col in columns)

        insert_query = f"""
            INSERT INTO {TABLE_NAME} ({column_names})
            VALUES %s
        """

        print("Loading data into PostgreSQL...")

        execute_values(
            cursor,
            insert_query,
            rows,
            page_size=1000
        )

        conn.commit()

        print(
            f"Successfully loaded {len(rows)} records "
            f"into {TABLE_NAME}."
        )

    except Exception as e:
        conn.rollback()
        print("Database load failed.")
        print(e)
        raise

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()