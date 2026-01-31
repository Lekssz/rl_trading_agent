import pandas as pd
import numpy as np

PATH_IN  = "data/processed/ohlcv/parquet/eurusd_30m_2019_2023.parquet"
PATH_OUT = "data/processed/ohlcv/parquet/eurusd_30m_2019_2023_cleaned.parquet"

df = pd.read_parquet(PATH_IN).sort_values("timestamp").reset_index(drop=True)
ts = pd.to_datetime(df["timestamp"], utc=True).to_numpy()

# find gaps
diff = np.diff(ts)
gap_mask = diff != np.timedelta64(30, "m")

prev_arr = ts[:-1][gap_mask]
curr_arr = ts[1:][gap_mask]
dur_arr  = curr_arr - prev_arr

def is_weekend_like(td):
    hrs = td / np.timedelta64(1, "h")
    return 47.0 <= hrs <= 50.0  # ≈ weekend closure

non_wk_endings = [pd.Timestamp(b) for b,td in zip(curr_arr, dur_arr) if not is_weekend_like(td)]
print("Non-weekend gap bar-ends (to drop):", len(non_wk_endings))

# drop those bars
before = len(df)
df = df[~df["timestamp"].isin(non_wk_endings)].reset_index(drop=True)
after = len(df)
print(f"Dropped {before-after} rows. New rows={after}")

df.to_parquet(PATH_OUT, index=False)
print("Saved →", PATH_OUT)
