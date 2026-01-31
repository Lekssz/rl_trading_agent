import pandas as pd
import json
from pathlib import Path

# ---------- Paths ----------
STATE_IN  = Path("data/processed/state/eurusd_state_with_macro_30m_2019_2023.parquet")
STATE_META_IN = Path("data/processed/state/eurusd_state_with_macro_30m_2019_2023.meta.json")

CNN_CSV = Path("data/model/cnn_price_eurusd/cnn_price_embeddings_EURUSD.csv")

STATE_OUT = Path("data/processed/state/eurusd_state_with_macro_cnn_30m_2019_2023.parquet")
STATE_META_OUT = Path("data/processed/state/eurusd_state_with_macro_cnn_30m_2019_2023.meta.json")


def main():
    # --- Load base state ---
    print("Loading state:", STATE_IN)
    df_state = pd.read_parquet(STATE_IN)
    df_state["timestamp"] = pd.to_datetime(df_state["timestamp"], utc=True)
    df_state = df_state.sort_values("timestamp").reset_index(drop=True)

    # --- Load CNN embeddings ---
    print("Loading CNN embeddings:", CNN_CSV)
    df_cnn = pd.read_csv(CNN_CSV)

    # Parse time and shift by +30min to align with bar-end timestamps
    df_cnn["time"] = pd.to_datetime(df_cnn["time"], utc=True)
    df_cnn["timestamp"] = df_cnn["time"] + pd.Timedelta(minutes=30)

    # Keep only timestamp + CNN features
    cnn_cols = [c for c in df_cnn.columns if c.startswith("z")]  # embeddings
    cols_keep = ["timestamp", "prob_up_1h"] + cnn_cols
    df_cnn = df_cnn[cols_keep].sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    print(f"CNN rows: {len(df_cnn):,} | Embedding dims: {len(cnn_cols)}")

    # --- Merge (left join on state timeline) ---
    df_merged = df_state.merge(df_cnn, on="timestamp", how="left")

    print("Merged rows:", len(df_merged))
    print("Nulls in prob_up_1h after merge:", df_merged["prob_up_1h"].isna().sum())

    # Optional: fill early NaNs (before CNN starts) with 0.5 and 0 embeddings
    df_merged["prob_up_1h"] = df_merged["prob_up_1h"].fillna(0.5)
    for c in cnn_cols:
        df_merged[c] = df_merged[c].fillna(0.0)

    # --- Save merged state ---
    STATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(STATE_OUT, index=False)
    print("Saved merged state →", STATE_OUT)

    # --- Update meta JSON ---
    try:
        with open(STATE_META_IN, "r") as f:
            meta = json.load(f)
    except FileNotFoundError:
        meta = {}

    meta["cnn_price_features"] = {
        "prob_col": "prob_up_1h",
        "emb_cols": cnn_cols,
        "instrument": "EURUSD",
        "source": str(CNN_CSV)
    }

    with open(STATE_META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved meta →", STATE_META_OUT)


if __name__ == "__main__":
    main()
