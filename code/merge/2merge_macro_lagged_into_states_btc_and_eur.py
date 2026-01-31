import pandas as pd
import numpy as np
from pathlib import Path
import json

# ---------- PATHS ----------
STATE_DIR = Path("data/processed/state")
MACRO_PARQ = Path("data/processed/macro/macro_30m_2019_2023.parquet")

BTC_STATE_IN  = STATE_DIR / "btcusd_state_30m_2019_2023.parquet"
EUR_STATE_IN  = STATE_DIR / "eurusd_state_30m_2019_2023.parquet"

BTC_STATE_OUT = STATE_DIR / "btcusd_state_with_macro_30m_2019_2023.parquet"
EUR_STATE_OUT = STATE_DIR / "eurusd_state_with_macro_30m_2019_2023.parquet"

BTC_META_OUT  = STATE_DIR / "btcusd_state_with_macro_30m_2019_2023.meta.json"
EUR_META_OUT  = STATE_DIR / "eurusd_state_with_macro_30m_2019_2023.meta.json"


def load_state(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_macro(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # ensure numeric
    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_and_lag_macro(state_df: pd.DataFrame,
                        macro_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    # macro columns (all except timestamp)
    macro_cols = [c for c in macro_df.columns if c != "timestamp"]

    # split into levels vs flags
    flag_cols = [c for c in macro_cols if c.endswith("_chg_flag")]
    level_cols = [c for c in macro_cols if c not in flag_cols]

    # merge on timestamp (keep state timeline)
    df = state_df.merge(macro_df, on="timestamp", how="left")

    # forward-fill macro levels first, then flags (NaN → 0)
    if level_cols:
        df[level_cols] = df[level_cols].ffill()
    if flag_cols:
        df[flag_cols] = df[flag_cols].fillna(0.0)

    # lag all macro cols by +1 bar
    df[macro_cols] = df[macro_cols].shift(1)

    # after shift, first row(s) become NaN -> fill:
    # - levels: 0.0 (means “no macro info yet” at very first bar)
    # - flags : 0 (no event)
    if level_cols:
        df[level_cols] = df[level_cols].fillna(0.0).astype("float64")
    if flag_cols:
        df[flag_cols] = df[flag_cols].fillna(0).astype("int8")

    return df, level_cols, flag_cols


def make_meta(df: pd.DataFrame,
              macro_level_cols: list[str],
              macro_flag_cols: list[str],
              tag: str) -> dict:
    meta = {
        "asset": tag,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "timestamp_min": str(df["timestamp"].min()),
        "timestamp_max": str(df["timestamp"].max()),
        "macro_level_columns": macro_level_cols,
        "macro_flag_columns": macro_flag_cols,
        "notes": (
            "State includes OHLCV, lagged GDELT news, and macro series. "
            "Macro series (levels and *_chg_flag) merged from macro_30m_2019_2023.parquet "
            "and lagged by +1 bar to avoid look-ahead."
        ),
    }
    return meta


def run_one(state_in: Path,
            state_out: Path,
            meta_out: Path,
            macro_df: pd.DataFrame,
            tag: str):
    print(f"\n=== MERGE MACRO INTO {tag} ===")
    state_df = load_state(state_in)
    merged, macro_levels, macro_flags = merge_and_lag_macro(state_df, macro_df)

    # sanity checks
    assert merged["timestamp"].is_monotonic_increasing, "Timestamps not sorted!"
    if merged["timestamp"].dt.tz is None:
        raise ValueError("Timestamps must be tz-aware (UTC).")

    print(f"Rows: {len(merged):,} | Columns: {len(merged.columns)}")
    print("Macro level columns:", macro_levels)
    print("Macro flag columns :", macro_flags)
    print("Date range:", merged["timestamp"].min(), "→", merged["timestamp"].max())

    # save parquet
    merged.to_parquet(state_out, index=False)
    print("Saved state+macro →", state_out)

    # save meta
    meta = make_meta(merged, macro_levels, macro_flags, tag)
    meta_out.write_text(json.dumps(meta, indent=2))
    print("Saved meta →", meta_out)


if __name__ == "__main__":
    macro_df = load_macro(MACRO_PARQ)

    run_one(
        state_in=BTC_STATE_IN,
        state_out=BTC_STATE_OUT,
        meta_out=BTC_META_OUT,
        macro_df=macro_df,
        tag="BTCUSD",
    )
    run_one(
        state_in=EUR_STATE_IN,
        state_out=EUR_STATE_OUT,
        meta_out=EUR_META_OUT,
        macro_df=macro_df,
        tag="EURUSD",
    )
