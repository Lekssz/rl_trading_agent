#loads each CSV,
#shifts timestamp += 30min (bar start → bar end),
#sorts, dedupes, clips to 2019–2023,
#saves Parquet.
import pandas as pd
import numpy as np
from pathlib import Path

# Inputs (your files)
btc_csv = Path("data/processed/ohlcv/binance_BTCUSDT_30m_2019-2023_processed.csv")
eur_csv = Path("data/processed/ohlcv/oanda_EURUSD_M30_2019-2023_PROCESSED.csv")

# Outputs (Parquet)
out_dir = Path("data/processed/ohlcv/parquet"); out_dir.mkdir(parents=True, exist_ok=True)
btc_parq = out_dir / "btcusd_30m_2019_2023.parquet"
eur_parq = out_dir / "eurusd_30m_2019_2023.parquet"

CLIP_START = "2019-01-01 00:00:00+00:00"
CLIP_END   = "2023-12-31 23:59:59+00:00"

def load_shift_save(csv_path: Path, out_path: Path, name: str):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    # basic schema check
    req = ["open","high","low","close","volume"]
    miss = [c for c in req if c not in df.columns]
    if miss: raise ValueError(f"{name}: missing cols {miss}")

    # bar START -> bar END
    before = df["timestamp"].iloc[:3].tolist()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True) + pd.Timedelta(minutes=30)
    after = df["timestamp"].iloc[:3].tolist()

    # keep / clean
    df = (df[["timestamp"] + req]
          .sort_values("timestamp")
          .drop_duplicates(subset=["timestamp"], keep="last"))
    # clip to 2019–2023
    df = df[(df["timestamp"] >= CLIP_START) & (df["timestamp"] <= CLIP_END)].reset_index(drop=True)

    # spacing info (just a quick warning if gaps exist)
    diffs = df["timestamp"].diff().dropna()
    bad = int((diffs != pd.Timedelta("30min")).sum())
    if bad:
        print(f"⚠️  {name}: {bad} gaps not equal to 30min.")
    else:
        print(f"✅ {name}: uniform 30min spacing after bar-end shift.")

    # save
    df.to_parquet(out_path, index=False)
    print(f"Saved {name} → {out_path}")
    print(f"First 3 timestamps (before shift) : {before}")
    print(f"First 3 timestamps (after shift)  : {after}")
    print(f"Range: {df['timestamp'].min()} → {df['timestamp'].max()} | rows={len(df)}\n")

load_shift_save(btc_csv, btc_parq, "BTCUSDT 30m")
load_shift_save(eur_csv, eur_parq, "EURUSD M30")
