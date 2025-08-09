"""
OANDA OHLCV Cleaning Script
---------------------------
Cleans raw OANDA EUR/USD 30-minute OHLCV data into a standardized, modeling-ready CSV.

Input : data/raw/oanda_EURUSD_M30_2019-2021.csv
Output: data/processed/oanda_EURUSD_M30_clean.csv

References (Harvard):
OANDA (2024) v20 REST API – Instruments: Candles. Available at:
https://developer.oanda.com/rest-live-v20/instrument-ep/ (Accessed: 9 August 2025)
Pandas (2024) pandas.to_datetime. Available at:
https://pandas.pydata.org/ (Accessed: 9 August 2025)
"""

import pandas as pd
from pathlib import Path

RAW_PATH   = Path("data/raw/oanda_EURUSD_M30_2019-2021.csv")
CLEAN_PATH = Path("data/processed/oanda_EURUSD_M30_clean.csv")

# === PERSONAL PROJECT NOTES ===
# This cleaning step standardizes my OANDA forex data to match my Binance crypto dataset
# in column structure and timestamp format. This ensures my machine learning models can
# train on both datasets without special-case handling, enabling fair comparisons between
# BTC/USDT and EUR/USD performance in the same RL environment.
# ===============================

def process_oanda(raw_path: Path = RAW_PATH, save_path: Path = CLEAN_PATH):
    # 1) Load raw CSV
    df = pd.read_csv(raw_path)

    # 2) Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # 3) Ensure timestamp column exists and convert to datetime
    if "timestamp" not in df.columns:
        raise ValueError("Expected 'timestamp' in OANDA raw CSV.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # 4) Keep only relevant columns
    keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[keep_cols]

    # 5) Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # 6) Drop invalid rows, sort, remove duplicates
    df = (
        df.dropna(subset=["timestamp"])
          .sort_values("timestamp")
          .drop_duplicates(subset=["timestamp"])
          .reset_index(drop=True)
    )

    # 7) Save cleaned file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Cleaned OANDA saved to {save_path} with {len(df)} rows.")

if __name__ == "__main__":
    process_oanda()
