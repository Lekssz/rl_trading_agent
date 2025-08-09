"""
Binance OHLCV Cleaning Script
-----------------------------
Cleans raw Binance BTC/USDT 30-minute OHLCV data into a standardized, modeling-ready CSV.

Input : data/raw/binance_BTCUSDT_30m_2019-2021.csv
Output: data/processed/binance_BTCUSDT_30m_clean.csv

References (Harvard):
Binance (2023) Kline/Candlestick Data. Available at:
https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
(Accessed: 6 August 2025)
Pandas (2024) pandas.to_datetime. Available at:
https://pandas.pydata.org/ (Accessed: 6 August 2025)
"""

import pandas as pd
from pathlib import Path

RAW_PATH   = Path("data/raw/binance_BTCUSDT_30m_2019-2021.csv")
CLEAN_PATH = Path("data/processed/binance_BTCUSDT_30m_clean.csv")

# === PERSONAL PROJECT NOTES ===
# This cleaning script ensures my Binance data matches the structure and format of
# my OANDA EUR/USD dataset, so I can directly compare crypto and forex price action
# using the same modeling pipeline. This includes consistent column names, timestamp
# formats, and removal of duplicates/missing values.
# ===============================

def process_binance(raw_path: Path = RAW_PATH, save_path: Path = CLEAN_PATH):
    # 1) Load raw CSV
    df = pd.read_csv(raw_path)

    # 2) Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # 3) Ensure timestamp column exists and convert to datetime
    if "open_time" in df.columns:
        df.rename(columns={"open_time": "timestamp"}, inplace=True)
    if "timestamp" not in df.columns:
        raise ValueError("Expected 'timestamp' in Binance raw CSV.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")

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
    print(f"✅ Cleaned Binance saved to {save_path} with {len(df)} rows.")

if __name__ == "__main__":
    process_binance()
