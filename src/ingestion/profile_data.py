import pandas as pd
import glob
import os

RAW_PATH = "data/raw"

for file in glob.glob(os.path.join(RAW_PATH, "*.csv")):

    print("\n" + "=" * 70)
    print("FILE:", os.path.basename(file))
    print("=" * 70)

    df = pd.read_csv(
        file,
        encoding="latin1",
        low_memory=False
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumn names:")
    for column in df.columns:
        print(" -", column)

    print("\nMissing values - Top 10:")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )