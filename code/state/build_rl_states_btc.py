# build_rl_states_btc.py
# Converts frozen BTC Parquet state → RL-ready NumPy states with ablations
# Train-scaler period: 2019–2021
# PPO training: 2019–2022
# Test: 2023

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json

# ============================================================
# INPUT
# ============================================================

STATE_IN = Path(
    "data/processed/state/btc/btcusd_state_with_macro_cnn_30m_2019_2023.parquet"
)

TRAIN_START = pd.Timestamp("2019-01-01", tz="UTC")
TRAIN_END   = pd.Timestamp("2021-12-31 23:59:59", tz="UTC")

# ============================================================
# OUTPUT
# ============================================================

OUT_BASE = Path("data/rl_states/btc")
OUT_NPY  = OUT_BASE / "npy"
OUT_SPEC = OUT_BASE / "spec"

OUT_NPY.mkdir(parents=True, exist_ok=True)
OUT_SPEC.mkdir(parents=True, exist_ok=True)

OUT_PRICE_NPY = OUT_NPY / "close.npy"
OUT_TIME_NPY  = OUT_NPY / "timestamps.npy"
OUT_SPEC_JSON = OUT_SPEC / "state_spec.json"

# ============================================================
# PRICE FEATURE ENGINEERING
# ============================================================

def build_price_features(df):
    df = df.sort_values("timestamp").copy()

    df["log_close"] = np.log(df["close"].replace(0, np.nan))
    df["ret_1"] = df["log_close"].diff().fillna(0.0)

    df["rng_rel"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["rng_rel"] = df["rng_rel"].replace([np.inf, -np.inf], 0).fillna(0)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["clv"] = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng)
    df["clv"] = df["clv"].replace([np.inf, -np.inf], 0).fillna(0)

    df["log_vol"] = np.log1p(df["volume"].clip(lower=0))

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=1).mean().fillna(0.0)

    df["rv_1d"] = (
        df["ret_1"].pow(2).rolling(48, min_periods=1).sum()
    ).pow(0.5).fillna(0.0)

    m = df["close"].rolling(20, min_periods=1).mean()
    s = df["close"].rolling(20, min_periods=1).std().replace(0, np.nan)
    df["zclose20"] = ((df["close"] - m) / s).replace([np.inf, -np.inf], 0).fillna(0)

    ts = pd.to_datetime(df["timestamp"], utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = ts.dt.weekday
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    price_cols = [
        "ret_1",
        "rng_rel",
        "clv",
        "log_vol",
        "atr14",
        "rv_1d",
        "zclose20",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    return df, price_cols

# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading frozen BTC state...")
    df = pd.read_parquet(STATE_IN, engine="fastparquet").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # ---------------- PRICE ----------------
    print("Building price features...")
    df, price_cols = build_price_features(df)

    # ---------------- NEWS (GDELT) ----------------
    news_cols = [
        c
        for c in df.columns
        if (
            c.startswith("art_")
            or c.startswith("uniq_domains")
            or "tone" in c
            or "theme_" in c
        )
    ]
    print(f"Detected {len(news_cols)} news columns")

    # ---------------- MACRO ----------------
    macro_level_cols = ["ecb_dep_fac", "fedfunds", "euro_cpi_yoy", "vix"]
    macro_flag_cols = [
        "ecb_dep_fac_chg_flag",
        "fedfunds_chg_flag",
        "euro_cpi_yoy_chg_flag",
        "vix_chg_flag",
    ]

    # ---------------- CNN ----------------
    cnn_prob_col = "prob_up_1h"
    cnn_emb_cols = [c for c in df.columns if c.startswith("z")]
    print(f"CNN embedding dims: {len(cnn_emb_cols)}")

    # ---------------- FLAGS ----------------
    mask_train = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)

    thr = df.loc[mask_train, "burst_ratio_24h"].quantile(0.99)
    df["burst_flag_24h"] = (df["burst_ratio_24h"] >= thr).astype("float32")

    flag_cols = ["is_gap_fill", "burst_flag_24h"] + macro_flag_cols

    # ---------------- SCALING ----------------
    print("Fitting scalers on 2019–2021 (train period only)...")

    scaler_price = StandardScaler().fit(df.loc[mask_train, price_cols])
    scaler_macro = StandardScaler().fit(df.loc[mask_train, macro_level_cols])
    scaler_cnn   = StandardScaler().fit(df.loc[mask_train, cnn_emb_cols])

    df[price_cols] = scaler_price.transform(df[price_cols])
    df[macro_level_cols] = scaler_macro.transform(df[macro_level_cols])
    df[cnn_emb_cols] = scaler_cnn.transform(df[cnn_emb_cols])

    # DO NOT SCALE NEWS
    # DO NOT SCALE FLAGS
    # DO NOT SCALE prob_up_1h

    # ============================================================
    # ABLATIONS
    # ============================================================

    ABLATIONS = {
        "price": price_cols,
        "price_cnn": price_cols + [cnn_prob_col] + cnn_emb_cols,
        "price_gdelt": price_cols + news_cols,
        "price_cnn_gdelt": price_cols + news_cols + [cnn_prob_col] + cnn_emb_cols,
        "price_macro": price_cols + macro_level_cols + macro_flag_cols,
        "full": (
            price_cols
            + news_cols
            + macro_level_cols
            + macro_flag_cols
            + [cnn_prob_col]
            + cnn_emb_cols
            + flag_cols
        ),
    }

    # ---------------- SAVE SHARED ARRAYS ----------------
    prices = df["close"].to_numpy(np.float32)
    times = df["timestamp"].astype("int64").to_numpy()

    np.save(OUT_PRICE_NPY, prices)
    np.save(OUT_TIME_NPY, times)

    # ---------------- SAVE ABLATIONS ----------------
    print("Saving RL state matrices...")
    ablation_dims = {}

    for name, cols in ABLATIONS.items():
        X = df[cols].to_numpy(np.float32)
        out_path = OUT_NPY / f"state_{name}.npy"
        np.save(out_path, X)
        ablation_dims[name] = X.shape[1]
        print(f"  → {name}: {X.shape}")

    # ---------------- SPEC ----------------
    spec = {
        "asset": "BTCUSD",
        "source_parquet": str(STATE_IN),
        "train_scaler_start": str(TRAIN_START),
        "train_scaler_end": str(TRAIN_END),
        "price_cols": price_cols,
        "news_cols": news_cols,
        "macro_level_cols": macro_level_cols,
        "macro_flag_cols": macro_flag_cols,
        "cnn_prob_col": cnn_prob_col,
        "cnn_emb_cols": cnn_emb_cols,
        "flag_cols": flag_cols,
        "ablations": {
            name: {"dim": ablation_dims[name], "cols": cols}
            for name, cols in ABLATIONS.items()
        },
    }

    with open(OUT_SPEC_JSON, "w") as f:
        json.dump(spec, f, indent=2)

    print("\nBTC RL states successfully built:")
    print(" →", OUT_BASE)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
