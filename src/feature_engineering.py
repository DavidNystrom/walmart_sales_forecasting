# src/feature_engineering.py

import os
import logging
import pandas as pd
from src import config

# ─── Setup Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def build_features():
    # 1. Load the cleaned & merged data
    in_path = os.path.join(config.PROCESSED_DATA_DIR, "combined.csv")
    logging.info(f"Loading {in_path}")
    df = pd.read_csv(in_path, parse_dates=["Date"])
    logging.info(f"  → combined.csv shape: {df.shape}")

    # 2. Create date-based features
    logging.info("Creating date-based features")
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["DayOfWeek"]  = df["Date"].dt.dayofweek
    df["DayOfYear"]  = df["Date"].dt.dayofyear

    # 3. Encode categorical variables
    logging.info("Encoding Store Type as one-hot")
    df = pd.get_dummies(
        df,
        columns=["Type"],
        prefix="StoreType",
        drop_first=True,
    )

    logging.info("Converting IsHoliday to integer flag")
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    # 4. Drop columns not needed for modeling
    logging.info("Dropping original Date column")
    df.drop(columns=["Date"], inplace=True)

    # 5. Save to features.csv
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    out_path = os.path.join(config.PROCESSED_DATA_DIR, "features.csv")
    df.to_csv(out_path, index=False)
    logging.info(f"Saved features to {out_path} with shape {df.shape}")

if __name__ == "__main__":
    build_features()
