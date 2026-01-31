import pandas as pd
import numpy as np

PATH = "data/processed/ohlcv/parquet/eurusd_30m_2019_2023.parquet"

# Load & prep
df = pd.read_parquet(PATH).sort_values("timestamp").reset_index(drop=True)
ts = pd.to_datetime(df["timestamp"], utc=True)
t = ts.to_numpy()
# gaps where != 30min
diff = np.diff(t)
gap_mask = diff != np.timedelta64(30, "m")

# arrays of gap endpoints
prev_arr = t[:-1][gap_mask]
curr_arr = t[1:][gap_mask]
dur_arr  = curr_arr - prev_arr  # numpy timedeltas

def is_weekend_like(td: np.timedelta64) -> bool:
    # Weekend closure ≈ 48h; allow tolerance for DST/venue differences
    # 47h (169200000000000 ns) to 50h (180000000000000 ns) in timedelta64[ns]
    hours = td / np.timedelta64(1, "h")
    return 47.0 <= hours <= 50.0

weekend_like_flags = [is_weekend_like(td) for td in dur_arr]

total_gaps       = int(gap_mask.sum())
weekend_like     = int(sum(weekend_like_flags))
non_weekend_gaps = total_gaps - weekend_like

print(f"Total gaps (any): {total_gaps}")
print(f"Weekend-like gaps (≈2d ±1h): {weekend_like}")
print(f"Remaining gaps (non-weekend): {non_weekend_gaps}")

# Optional: show first 10 non-weekend gaps to inspect
print("\nFirst 10 non-weekend gaps:")
shown = 0
for a, b, td, wk in zip(prev_arr, curr_arr, dur_arr, weekend_like_flags):
    if not wk:
        print(f"- At {pd.Timestamp(b)}  gap = {pd.to_timedelta(td)}")
        shown += 1
        if shown >= 10:
            break
