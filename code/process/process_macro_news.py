import pandas as pd
import numpy as np
from pathlib import Path
import json

# ---------- PATHS ----------
RAW_MACRO_DIR = Path("data/raw/macro")
PROC_MACRO_DIR = Path("data/processed/macro")
PROC_MACRO_DIR.mkdir(parents=True, exist_ok=True)

ECB_PATH = RAW_MACRO_DIR / "ecb_deposit_facility.csv"
CPI_PATH = RAW_MACRO_DIR / "euro_cpi_yoy_2019_2023.xlsx"
FED_PATH = RAW_MACRO_DIR / "fedfunds_2019_2023.xlsx"
VIX_PATH = RAW_MACRO_DIR / "vix_index_2019_2023.xlsx"

OUT_PARQ = PROC_MACRO_DIR / "macro_30m_2019_2023.parquet"
OUT_PROV = PROC_MACRO_DIR / "macro_30m_2019_2023.provenance.json"

START = pd.Timestamp("2019-01-01", tz="UTC")
END   = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")

# ---------- GENERIC LOADER ----------

def read_any(path: Path) -> pd.DataFrame:
    """
    Read CSV or XLSX, guess date column and value column.
    Handles common Bloomberg/FRED formats:
    - Date / Observation Date
    - PX_LAST / LAST PRICE / Close / Value
    """
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        df = pd.read_excel(path, sheet_name=0)

    # Normalise column names
    df.columns = [str(c).strip() for c in df.columns]
    cols_lower = {c.lower(): c for c in df.columns}

    # Pick date column
    date_candidates = [
        "date", "observation date", "obs date", "obs_date", "timestamp"
    ]
    date_col = None
    for key in date_candidates:
        if key in cols_lower:
            date_col = cols_lower[key]
            break
    if date_col is None:
        # fallback: first column
        date_col = df.columns[0]

    # Pick value column
    value_candidates = [
        "px_last", "last price", "close", "value", "index",
        "rate", "upper bound", "lower bound", "target", "vix"
    ]
    value_col = None
    for c in df.columns:
        if c == date_col:
            continue
        if c.lower() in value_candidates:
            value_col = c
            break
    if value_col is None:
        # fallback: second column
        value_col = [c for c in df.columns if c != date_col][0]

    out = df[[date_col, value_col]].copy()
    out.columns = ["timestamp", "value"]

    # Parse timestamp & numeric values
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Strip % signs and commas
    out["value"] = (
        out["value"].astype(str)
                     .str.replace(",", "")
                     .str.replace("%", "")
    )
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    return out

def normalize_units(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Convert units so that:
    - ecb_dep_fac, fedfunds: percent (e.g., 3.75)
    - euro_cpi_yoy: percent
    - vix: index level
    """
    s = df["value"].copy()
    median = s.median()

    if name in ["ecb_dep_fac", "fedfunds"]:
        # If looks like basis points (e.g., 375), convert to percent
        if median is not None and median > 20:
            s = s / 100.0

    df[name] = s.astype(float)
    return df[["timestamp", name]]

def effective_next_bar(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Make macro changes effective from the NEXT 30-min bar.
    Avoids using the new value inside its own release bar.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["timestamp"] = out["timestamp"] + pd.Timedelta("30min")
    return out[["timestamp", col]]

def vix_to_next_midnight_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treat each VIX daily close as effective from next UTC midnight.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["timestamp"] = out["timestamp"].dt.normalize() + pd.Timedelta("1D")
    return out[["timestamp", "vix"]]

def to_30m_ffill(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Reindex to full 30-min UTC grid and forward-fill.
    """
    full_idx = pd.date_range(START, END, freq="30min", tz="UTC")
    s = df.set_index("timestamp")[col].sort_index()
    s = s.loc[(s.index >= START) & (s.index <= END)]
    s = s.reindex(full_idx).ffill()
    return s.to_frame(col).reset_index().rename(columns={"index": "timestamp"})

# ---------- BUILD SERIES ----------

# ECB deposit facility (from FRED, but normalized as ECB series)
ecb_raw = read_any(ECB_PATH)
ecb = normalize_units(ecb_raw, "ecb_dep_fac")
ecb_eff = effective_next_bar(ecb, "ecb_dep_fac")
ecb_30m = to_30m_ffill(ecb_eff, "ecb_dep_fac")

# Fed funds
fed_raw = read_any(FED_PATH)
fed = normalize_units(fed_raw, "fedfunds")
fed_eff = effective_next_bar(fed, "fedfunds")
fed_30m = to_30m_ffill(fed_eff, "fedfunds")

# Euro CPI YoY
cpi_raw = read_any(CPI_PATH)
cpi = normalize_units(cpi_raw, "euro_cpi_yoy")
cpi_eff = effective_next_bar(cpi, "euro_cpi_yoy")
cpi_30m = to_30m_ffill(cpi_eff, "euro_cpi_yoy")

# VIX
vix_raw = read_any(VIX_PATH)
vix = normalize_units(vix_raw, "vix")
vix_eff = vix_to_next_midnight_utc(vix)
vix_30m = to_30m_ffill(vix_eff, "vix")

# ---------- MERGE & FLAGS ----------

macro = (
    ecb_30m.merge(fed_30m, on="timestamp", how="outer")
           .merge(cpi_30m, on="timestamp", how="outer")
           .merge(vix_30m, on="timestamp", how="outer")
).sort_values("timestamp").reset_index(drop=True)

# Forward-fill any leading NaNs from start
for c in ["ecb_dep_fac", "fedfunds", "euro_cpi_yoy", "vix"]:
    macro[c] = macro[c].ffill()

# Add simple change flags (1 if level changed at that 30m bar)
for c in ["ecb_dep_fac", "fedfunds", "euro_cpi_yoy", "vix"]:
    macro[f"{c}_chg_flag"] = (macro[c].diff().fillna(0) != 0).astype("int8")

# ---------- SAVE ----------

OUT_PARQ.parent.mkdir(parents=True, exist_ok=True)
macro.to_parquet(OUT_PARQ, index=False)

prov = {
    "inputs": {
        "ecb_deposit_facility": str(ECB_PATH),
        "euro_cpi_yoy": str(CPI_PATH),
        "fedfunds": str(FED_PATH),
        "vix_index": str(VIX_PATH),
    },
    "range_utc": [str(START), str(END)],
    "effective_rules": {
        "ecb_dep_fac": "next 30m bar",
        "fedfunds": "next 30m bar",
        "euro_cpi_yoy": "next 30m bar",
        "vix": "next UTC midnight",
    },
    "notes": (
        "Raw macro series (FRED/Bloomberg) resampled to 30-min UTC grid with forward-fill. "
        "Levels treated as final vintages. *_chg_flag marks bars where the level changed."
    ),
}
OUT_PROV.write_text(json.dumps(prov, indent=2))

print("Saved macro block →", OUT_PARQ)
print("Saved macro provenance →", OUT_PROV)
print("Rows:", len(macro), "| Columns:", len(macro.columns))
