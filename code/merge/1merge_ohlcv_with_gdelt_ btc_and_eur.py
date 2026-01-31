import pandas as pd
import numpy as np
from pathlib import Path
import json

# ---------- INPUTS ----------
# OHLCV (cleaned outputs)
BTC_OHLCV = Path("data/processed/ohlcv/parquet/btcusd_30m_2019_2023_filled.parquet")   # has is_gap_fill
EUR_OHLCV = Path("data/processed/ohlcv/parquet/eurusd_30m_2019_2023_cleaned.parquet")

# GDELT (processed outputs, scaled from Step 4A)
BTC_NEWS  = Path("data/processed/features/btcusd_gdelt_news_30m_scaled_2019_2023.parquet")
EUR_NEWS  = Path("data/processed/features/eurusd_gdelt_news_30m_scaled_2019_2023.parquet")

# OUTPUTS
OUT_DIR   = Path("data/processed/state")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BTC_OUT   = OUT_DIR / "btcusd_state_30m_2019_2023.parquet"
EUR_OUT   = OUT_DIR / "eurusd_state_30m_2019_2023.parquet"
BTC_META  = OUT_DIR / "btcusd_state_30m_2019_2023.meta.json"
EUR_META  = OUT_DIR / "eurusd_state_30m_2019_2023.meta.json"

# Train window for thresholds and CV
TRAIN_START = pd.Timestamp("2019-01-01", tz="UTC")
TRAIN_END   = pd.Timestamp("2021-12-31 23:59:59", tz="UTC")

# ---------- HELPERS ----------

def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

def load_news(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def merge_and_lag(ohlcv: pd.DataFrame, news: pd.DataFrame):
    # Identify news feature columns (everything except timestamp)
    news_cols = [c for c in news.columns if c != "timestamp"]

    # Merge on timestamp; keep OHLCV timeline (left join)
    df = ohlcv.merge(news, on="timestamp", how="left")

    # Fill missing news with 0.0 (no news in that bin)
    df[news_cols] = df[news_cols].fillna(0.0).astype("float64")

    # Compute burst_flag_24h from TRAIN quantile on burst_ratio_24h (pre-lag)
    burst_flag_col = None
    if "burst_ratio_24h" in df.columns:
        mask_train = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
        thr = df.loc[mask_train, "burst_ratio_24h"].quantile(0.99)
        df["burst_flag_24h"] = (df["burst_ratio_24h"] >= thr).astype("int8")
        burst_flag_col = "burst_flag_24h"
        news_cols_with_flags = news_cols + [burst_flag_col]
    else:
        thr = None
        news_cols_with_flags = news_cols

    # Lag all news features (and burst flag, if present) by +1 bar
    df[news_cols_with_flags] = df[news_cols_with_flags].shift(1)

    # After shift, first row(s) become NaN -> fill
    flag_cols = []
    for c in news_cols_with_flags:
        if c.endswith("_flag"):
            df[c] = df[c].fillna(0).astype("int8")
            flag_cols.append(c)
        else:
            df[c] = df[c].fillna(0.0).astype("float64")

    return df, news_cols, flag_cols, thr

def make_meta(df: pd.DataFrame,
              news_cols: list[str],
              flag_cols: list[str],
              burst_thr: float | None,
              tag: str) -> dict:
    meta = {
        "asset": tag,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "timestamp_min": str(df["timestamp"].min()),
        "timestamp_max": str(df["timestamp"].max()),
        "train_start": str(TRAIN_START),
        "train_end": str(TRAIN_END),
        "news_feature_columns": news_cols,
        "news_flag_columns": flag_cols,
        "burst_ratio_24h_p99_train": float(burst_thr) if burst_thr is not None else None,
        "notes": (
            "OHLCV merged with scaled GDELT features. "
            "All news features (and flags) are lagged by +1 bar to avoid look-ahead. "
            "Missing news values treated as 0.0. "
            "novelty_flag_30d is used as a continuous scaled feature (no extra binary)."
        ),
    }
    return meta

def run_one(ohlcv_path: Path, news_path: Path, out_path: Path, meta_path: Path, tag: str):
    print(f"\n=== MERGE {tag} ===")
    ohlcv = load_ohlcv(ohlcv_path)
    news  = load_news(news_path)
    merged, news_cols, flag_cols, burst_thr = merge_and_lag(ohlcv, news)

    # Basic sanity checks
    assert merged["timestamp"].is_monotonic_increasing, "Timestamps not sorted!"
    if merged["timestamp"].dt.tz is None:
        raise ValueError("Timestamps must be tz-aware (UTC).")

    # Print quick info
    print(f"Rows: {len(merged):,} | Columns: {len(merged.columns)}")
    print(f"News feature columns: {len(news_cols)}")
    print(f"News flag columns: {flag_cols}")
    print("Sample columns:", merged.columns[:10].tolist())
    print("Date range:", merged['timestamp'].min(), "→", merged['timestamp'].max())

    # Save parquet
    merged.to_parquet(out_path, index=False)
    print("Saved state →", out_path)

    # Save meta
    meta = make_meta(merged, news_cols, flag_cols, burst_thr, tag)
    meta_path.write_text(json.dumps(meta, indent=2))
    print("Saved meta  →", meta_path)

if __name__ == "__main__":
    run_one(BTC_OHLCV, BTC_NEWS, BTC_OUT, BTC_META, tag="BTCUSD")
    run_one(EUR_OHLCV, EUR_NEWS, EUR_OUT, EUR_META, tag="EURUSD")
