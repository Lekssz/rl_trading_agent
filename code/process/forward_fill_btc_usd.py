# Forward-fill prices, set volume=0, 
# adds a flag for gap-filled bars in OHLCV data.

import pandas as pd

PATH_IN  = "data/processed/ohlcv/parquet/btcusd_30m_2019_2023.parquet"
PATH_OUT = "data/processed/ohlcv/parquet/btcusd_30m_2019_2023_filled.parquet"

df = pd.read_parquet(PATH_IN).sort_values("timestamp")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# make a continuous 30-min timeline
full_index = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="30min", tz="UTC")
df_full = df.set_index("timestamp").reindex(full_index).rename_axis("timestamp").reset_index()

# mark fills
df_full["is_gap_fill"] = df_full["open"].isna().astype(int)

# forward-fill OHLC, set volume=0 for filled rows
for col in ["open", "high", "low", "close"]:
    df_full[col] = df_full[col].ffill()
df_full["volume"] = df_full["volume"].fillna(0.0)

print(f"Total bars: {len(df_full)}")
print(f"Gap-filled bars: {df_full['is_gap_fill'].sum()} ({100 * df_full['is_gap_fill'].mean():.3f}% of data)")

df_full.to_parquet(PATH_OUT, index=False)
print(f"Saved → {PATH_OUT}")
