"""
Binance OHLCV Data Ingestion Script
-----------------------------------
Downloads BTC/USDT 30-minute candlestick data from Binance for 2019-2023
and saves it as a CSV file.

Reference:
Binance (2023) Kline/Candlestick Data. Available at:
https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
(Accessed: 14 October 2025)
"""

import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import time

# ====== CONFIGURATION ======
SYMBOL = "BTCUSDT"
INTERVAL = "30m"
START_DATE = "2019-01-01"
END_DATE = "2023-12-31"
LIMIT = 1000  # Binance's max candles per call
RAW_PATH = Path("data/raw/ohlcv/binance_BTCUSDT_30m_2019-2023.csv")
BASE_URL = "https://api.binance.com/api/v3/klines"

# === PERSONAL PROJECT NOTES ===
# I chose 30-minute candles because they balance intraday resolution 
# with manageable API request sizes for backtesting.
# Higher frequency like 1-minute would be too heavy for my available storage 
# and training time, and lower frequency like 4h/D1 would lose important price action details.
# ============================

def date_to_millis(date_str: str) -> int:
    """Convert YYYY-MM-DD date string to milliseconds."""
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)

def fetch_binance_ohlcv(symbol: str, interval: str, start_ts: int, end_ts: int, limit: int = LIMIT):
    """Fetch a chunk of OHLCV data from Binance."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ts,
        "endTime": end_ts,
        "limit": limit
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

def get_all_data(symbol: str, interval: str, start_date: str, end_date: str):
    """Loop through all candles between start_date and end_date."""
    start_ts = date_to_millis(start_date)
    end_ts = date_to_millis(end_date)
    all_candles = []

    while start_ts < end_ts:
        print(f"🔄 Fetching Binance {symbol} candles from {datetime.utcfromtimestamp(start_ts/1000)}...")
        candles = fetch_binance_ohlcv(symbol, interval, start_ts, end_ts)
        if not candles:
            break
        all_candles.extend(candles)
        start_ts = candles[-1][6] + 1  # Next start time = last close time + 1ms
        time.sleep(0.5)  # Sleep to avoid API rate limits

    return all_candles

def save_to_csv(candles, filepath: Path):
    """Save the data to CSV with proper column names."""
    df = pd.DataFrame(candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Saved {len(df)} rows to {filepath}")

if __name__ == "__main__":
    candles = get_all_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    save_to_csv(candles, RAW_PATH)
