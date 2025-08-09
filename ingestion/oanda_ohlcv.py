"""
OANDA OHLCV Ingestion Script
----------------------------
Downloads EUR/USD 30-minute candlesticks from the OANDA API and saves
a raw CSV.

Reference:
OANDA (2024) v20 REST API – Instruments: Candles. Available at:
https://developer.oanda.com/rest-live-v20/instrument-ep/ (Accessed: 9 August 2025)
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

# ===== CONFIG =====
INSTRUMENT  = "EUR_USD"                   # Currency pair to fetch
GRANULARITY = "M30"                       # Candle size (M1, M5, M15, M30, H1, D)
START_DATE  = "2019-01-01"                 # Data start date (UTC)
END_DATE    = "2021-12-31"                 # Data end date (UTC)
PRICE_TYPE  = "M"                          # Mid prices (M = mid, A = ask, B = bid)
OUT_PATH    = Path("data/raw/oanda_EURUSD_M30_2019-2021.csv")
BASE_URL    = "https://api-fxpractice.oanda.com/v3"

# === PERSONAL PROJECT NOTES ===
# I selected 30-minute candles to match my Binance BTCUSDT data, so both
# crypto and forex datasets have identical time granularity for easier comparison
# and model training. Using mid prices ("M") removes bid/ask spread noise and
# ensures a fairer representation of the underlying market movements.
# ===============================

# === Time Helper Functions ===
def iso(dt: datetime) -> str:
    """Convert Python datetime to OANDA-friendly UTC ISO string."""
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def parse_time(s: str) -> datetime:
    """Convert OANDA timestamp (with possible nanoseconds) into Python datetime."""
    s = s.replace("Z", "+00:00")
    if "." in s:
        head, tail = s.split(".", 1)
        frac, tz = tail.split("+00:00")
        s = f"{head}.{frac[:6]}+00:00"
    return datetime.fromisoformat(s)

def step_for(gran: str) -> timedelta:
    """Get time step size from granularity code."""
    table = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "D": 1440}
    if gran not in table:
        raise ValueError(f"Unsupported granularity: {gran}")
    return timedelta(minutes=table[gran])

# === API Fetch Function ===
def fetch(session: requests.Session, instrument: str, gran: str, t0: datetime, t1: datetime, price: str):
    """Fetch one chunk of candles between t0 and t1."""
    url = f"{BASE_URL}/instruments/{instrument}/candles"
    params = {
        "granularity": gran,
        "price": price,
        "from": iso(t0),
        "to": iso(t1),
        "includeFirst": "true"
    }
    r = session.get(url, params=params, timeout=30)
    if r.status_code in (401, 403):
        raise RuntimeError("Auth error from OANDA (401/403). Check API token/account ID.")
    r.raise_for_status()
    return r.json().get("candles", [])

# === Main Logic ===
def main():
    # Load your API credentials
    load_dotenv()
    api_key = os.getenv("OANDA_API_KEY")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    if not api_key or not account_id:
        raise RuntimeError("Missing OANDA_API_KEY or OANDA_ACCOUNT_ID in .env")

    # Prepare date range
    start = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # make data/raw if missing

    rows = []
    step = step_for(GRANULARITY)
    window = step * 5000  # large range; OANDA will cap

    with requests.Session() as s:
        s.headers.update({"Authorization": f"Bearer {api_key}"})
        t0 = start
        while t0 < end:
            print(f"🔄 Downloading OANDA {INSTRUMENT} candles from {t0} to {min(t0 + window, end)}...")
            # Try up to 3 times if there’s a temporary error
            for tries in range(4):
                try:
                    candles = fetch(s, INSTRUMENT, GRANULARITY, t0, min(t0 + window, end), PRICE_TYPE)
                    break
                except requests.HTTPError as e:
                    if tries < 3 and e.response is not None and e.response.status_code in (429, 500, 502, 503, 504):
                        time.sleep(1.5 * (tries + 1))
                        continue
                    raise

            # If no candles returned, move on
            if not candles:
                t0 = t0 + window
                continue

            # Store each candle in our target format
            for c in candles:
                if not c.get("complete", False):
                    continue
                mid = c.get("mid", {})
                ts = parse_time(c["time"])
                rows.append({
                    "timestamp": ts.isoformat(),
                    "open": float(mid.get("o", "nan")),
                    "high": float(mid.get("h", "nan")),
                    "low": float(mid.get("l", "nan")),
                    "close": float(mid.get("c", "nan")),
                    "volume": int(c.get("volume", 0)),
                })

            # Move the time pointer forward
            last_ts = parse_time(candles[-1]["time"])
            t0 = last_ts + step
            time.sleep(0.3)  # avoid hitting OANDA too fast

    # Save to CSV
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved {len(df)} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()
