import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PARQ_PATH = Path("data/processed/ohlcv/parquet/btcusd_30m_2019_2023.parquet")
REPORT_CSV = Path("data/processed/ohlcv/parquet/btcusd_30m_gaps_report.csv")
FIG_PATH   = Path("data/processed/ohlcv/parquet/btcusd_30m_gaps_timeline.png")

# Load and prep
df = pd.read_parquet(PARQ_PATH).sort_values("timestamp").reset_index(drop=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Compute gaps
t = df["timestamp"].to_numpy()
diffs = np.diff(t)
gap_mask = diffs != np.timedelta64(30, "m")

prev_arr = t[:-1][gap_mask]
curr_arr = t[1:][gap_mask]
dur_arr  = np.array(curr_arr - prev_arr, dtype='timedelta64[ns]')
miss_cnt = (dur_arr / np.timedelta64(30, "m")).astype(int) - 1

# Build report
rep = pd.DataFrame({
    "gap_start_at": pd.to_datetime(prev_arr),
    "gap_end_at":   pd.to_datetime(curr_arr),
    "gap_duration": pd.to_timedelta(dur_arr),
    "missing_30m_bars": miss_cnt
}).sort_values("gap_start_at").reset_index(drop=True)

# Save CSV
REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
rep.to_csv(REPORT_CSV, index=False)

# Plot (single chart, matplotlib, no custom colors)
plt.figure(figsize=(12, 4))
plt.scatter(rep["gap_end_at"], rep["missing_30m_bars"])
plt.title("BTC/USDT 30-min Gaps Timeline (y = missing 30-min bars)")
plt.xlabel("Gap end time (UTC)")
plt.ylabel("Missing 30-min bars")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=150)

print(f"Total gaps: {len(rep)}")
print(f"Saved CSV → {REPORT_CSV}")
print(f"Saved PNG → {FIG_PATH}")
print(rep.head(10))
