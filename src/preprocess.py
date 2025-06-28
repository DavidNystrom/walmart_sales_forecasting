# src/preprocess.py

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

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV into a pandas DataFrame."""
    logging.info(f"Loading {path}")
    df = pd.read_csv(path)
    logging.info(f"  -> shape: {df.shape}")
    return df

def clean_and_merge():
    # 1. Load raw files
    train = load_csv(os.path.join(config.RAW_DATA_DIR, "train.csv"))
    stores = load_csv(os.path.join(config.RAW_DATA_DIR, "stores.csv"))
    feats = load_csv(os.path.join(config.RAW_DATA_DIR, "features.csv"))

    # 2. Parse dates
    logging.info("Parsing Date columns to datetime")
    train["Date"] = pd.to_datetime(train["Date"])
    feats["Date"] = pd.to_datetime(feats["Date"])

    # 3. Inspect missing values (top 3 for each)
    for df, name in [(train, "train"), (stores, "stores"), (feats, "features")]:
        missing = df.isna().sum().sort_values(ascending=False).head(3)
        logging.info(f"{name} top missing:\n{missing}")

    # 4. Merge DataFrames
    logging.info("Merging train + stores")
    data = train.merge(stores, on="Store", how="left")
    logging.info("Merging result + features")
    data = data.merge(feats, on=["Store", "Date", "IsHoliday"], how="left")
    logging.info(f"Merged dataframe shape: {data.shape}")

    # 5. Handle missing values
    #   - Markdown columns: NaN → 0
    markdown_cols = [c for c in feats.columns if "MarkDown" in c]
    logging.info(f"Filling Markdown columns {markdown_cols} NaN → 0")
    data[markdown_cols] = data[markdown_cols].fillna(0)

    #   - Numeric columns: forward-fill then zero-fill
    num_cols = data.select_dtypes(include=["number"]).columns
    logging.info("Forward-fill numeric columns, then fill NaN → 0")
    data[num_cols] = data[num_cols].ffill().fillna(0)

    #   - Categorical columns: NaN → 'Unknown'
    cat_cols = data.select_dtypes(include=["object"]).columns
    logging.info("Filling categorical columns NaN → 'Unknown'")
    data[cat_cols] = data[cat_cols].fillna("Unknown")

    # 6. Save cleaned, merged dataset
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    out_path = os.path.join(config.PROCESSED_DATA_DIR, "combined.csv")
    data.to_csv(out_path, index=False)
    logging.info(f"Saved combined data to {out_path}")

if __name__ == "__main__":
    clean_and_merge()
