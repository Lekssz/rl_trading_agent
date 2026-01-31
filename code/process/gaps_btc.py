import pandas as pd
import numpy as np

PATH = "data/processed/ohlcv/parquet/btcusd_30m_2019_2023.parquet"
OUT  = "data/processed/ohlcv/parquet/btcusd_30m_2019_2023_gap_report.csv"

# Load and sort
df = pd.read_parquet(PATH).sort_values("timestamp").reset_index(drop=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Check duplicates
dupes = int(df["timestamp"].duplicated().sum())
print(f"Duplicates: {dupes}")

# Calculate diffs
t = df["timestamp"].to_numpy()
diffs = np.diff(t)

# Find where spacing != 30 minutes
gap_mask = diffs != np.timedelta64(30, "m")

# Extract arrays of previous/next timestamps
prev_arr = t[:-1][gap_mask]
curr_arr = t[1:][gap_mask]

# Convert duration array explicitly to timedelta64[ns]
dur_arr = np.array(curr_arr - prev_arr, dtype='timedelta64[ns]')

# Compute number of missing 30-min bars
miss_cnt = (dur_arr / np.timedelta64(30, "m")).astype(int) - 1

# Build report
rep = pd.DataFrame({
    "gap_start_at": pd.to_datetime(prev_arr),
    "gap_end_at": pd.to_datetime(curr_arr),
    "gap_duration": pd.to_timedelta(dur_arr),
    "missing_30m_bars": miss_cnt
}).sort_values("gap_start_at").reset_index(drop=True)

print(f"Total gaps (non-30min): {len(rep)}")
print(rep.head(10))

# Optional: save full report
rep.to_csv(OUT, index=False)
print(f"Saved gap report → {OUT}")
