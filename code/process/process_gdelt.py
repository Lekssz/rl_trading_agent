
#train/holdout/forward splits (2019–2021 / 2022 / 2023)
#winsorize p99 on TRAIN only for counts
#log1p the capped counts
#derive a few compact signals (burst, tone surprise, domain diversity)
#RobustScaler (median/IQR) fit on TRAIN only
#saves processed Parquet + a scaler JSON for provenance

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# ------------------------------------------------------------------
# YOUR FILES (explicit paths)
# ------------------------------------------------------------------
BTC_FILE = Path("data/processed/news/gdeltv2/btc_gkg_30m_dense_features_2019_2023_fullgrid_roll.parquet")
EUR_FILE = Path("data/processed/news/gdeltv2/eurusd_gkg_30m_dense_features_2019_2023_fullgrid_roll.parquet")

OUT_DIR = Path("data/processed/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# STUDY WINDOWS
# ------------------------------------------------------------------
TRAIN_START = pd.Timestamp("2019-01-01", tz="UTC")
TRAIN_END   = pd.Timestamp("2021-12-31 23:59:59", tz="UTC")
HOLD_START  = pd.Timestamp("2022-01-01", tz="UTC")
HOLD_END    = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
FWD_START   = pd.Timestamp("2023-01-01", tz="UTC")
FWD_END     = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")
ALL_START, ALL_END = TRAIN_START, FWD_END

# ------------------------------------------------------------------
# FEATURE DEFINITIONS
# ------------------------------------------------------------------
REQUIRED_COLS = [
    "timestamp",
    "art_count","mean_tone","median_tone","uniq_domains",
    "art_24h","art_7d","mean_tone_24h","mean_tone_7d",
    "novelty_flag_30d",
]

THEMES = {
    "BTCUSD": ["theme_regulation","theme_exchange_hack_outage","theme_stablecoin","theme_network_miner"],
    "EURUSD": ["theme_rates_cbank","theme_inflation","theme_labor","theme_growth"],
}

BASE_NEWS_COLS = [
    "art_count_cap_log1p", "art_24h_cap_log1p",
    "burst_ratio_24h",
    "tone_surprise_24h", "mean_tone", "mean_tone_24h",
    "domain_diversity",
    "novelty_flag_30d",
]

# ------------------------------------------------------------------
def clip_at_quantile_train_only(s, ts, q=0.99):
    mask_train = (ts >= TRAIN_START) & (ts <= TRAIN_END)
    cap = s[mask_train].quantile(q)
    return np.minimum(s, cap), float(cap)

def build_gdelt_features(df_g, theme_cols):
    df = df_g.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df["timestamp"] >= ALL_START) & (df["timestamp"] <= ALL_END)]
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Winsorize counts on TRAIN only
    df["art_count_cap"],    p99_cnt  = clip_at_quantile_train_only(df["art_count"],    df["timestamp"])
    df["art_24h_cap"],      p99_24h  = clip_at_quantile_train_only(df["art_24h"],      df["timestamp"])
    df["art_7d_cap"],       p99_7d   = clip_at_quantile_train_only(df["art_7d"],       df["timestamp"])
    df["uniq_domains_cap"], p99_uniq = clip_at_quantile_train_only(df["uniq_domains"], df["timestamp"])

    # log1p
    for c in ["art_count_cap","art_24h_cap","art_7d_cap","uniq_domains_cap"]:
        df[c+"_log1p"] = np.log1p(df[c])

    # engineered signals
    df["burst_ratio_24h"]   = df["art_count_cap"] / (df["art_24h_cap"] / 48.0).replace(0, np.nan)
    df["tone_surprise_24h"] = df["mean_tone"] - df["mean_tone_24h"]
    df["domain_diversity"]  = df["uniq_domains_cap"] / df["art_count_cap"].clip(lower=1)

    feat_cols = BASE_NEWS_COLS + theme_cols
    df[feat_cols] = df[feat_cols].fillna(0.0)
    df[feat_cols] = df[feat_cols].astype("float64") 
    caps = {"p99_art_count": p99_cnt, "p99_art_24h": p99_24h, "p99_art_7d": p99_7d, "p99_uniq_domains": p99_uniq}
    return df[["timestamp"] + feat_cols].copy(), feat_cols, caps

def robust_scale_train_only(df_feat: pd.DataFrame, feat_cols: list[str], scaler_json_path: Path) -> pd.DataFrame:
    mask_train = (df_feat["timestamp"] >= TRAIN_START) & (df_feat["timestamp"] <= TRAIN_END)
    df_feat.loc[:, feat_cols] = df_feat[feat_cols].astype("float64")
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75))
    scaler.fit(df_feat.loc[mask_train, feat_cols])
    scaled = pd.DataFrame(
    scaler.transform(df_feat[feat_cols]),
    columns=feat_cols,
    index=df_feat.index
    )
    df_feat.loc[:, feat_cols] = scaled


    params = {
        "type": "RobustScaler",
        "quantile_range": [25, 75],
        "center_": scaler.center_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "feature_names": feat_cols,
        "train_start": str(TRAIN_START),
        "train_end": str(TRAIN_END),
    }
    scaler_json_path.write_text(json.dumps(params, indent=2))
    return df_feat

def process_asset(symbol, file_path, theme_cols):
    print(f"\n=== Processing {symbol} ===")
    df = pd.read_parquet(file_path)
    gdelt_feat, news_cols, caps = build_gdelt_features(df, theme_cols)

    scaler_path = OUT_DIR / f"{symbol.lower()}_gdelt_news_robust_scaler_2019_2021.json"
    gdelt_feat = robust_scale_train_only(gdelt_feat, news_cols, scaler_path)

    out_path = OUT_DIR / f"{symbol.lower()}_gdelt_news_30m_scaled_2019_2023.parquet"
    gdelt_feat.to_parquet(out_path, index=False)

    prov = {
        "asset": symbol,
        "source_file": str(file_path),
        "train_range": [str(TRAIN_START), str(TRAIN_END)],
        "holdout_range": [str(HOLD_START), str(HOLD_END)],
        "forward_range": [str(FWD_START), str(FWD_END)],
        "news_columns": news_cols,
        "winsor_caps_train_only": caps,
        "scaler_file": scaler_path.name,
        "notes": "Counts winsorized at p99 (train only), log1p, engineered ratios, RobustScaler fitted on 2019–2021. Lag applied later.",
    }
    (OUT_DIR / f"{symbol.lower()}_gdelt_news_30m_scaled_2019_2023.provenance.json").write_text(json.dumps(prov, indent=2))

    print("Saved features →", out_path)
    print("Saved scaler   →", scaler_path)
    print("Caps           →", caps)

# ------------------------------------------------------------------
if __name__ == "__main__":
    process_asset("BTCUSD", BTC_FILE, THEMES["BTCUSD"])
    process_asset("EURUSD", EUR_FILE, THEMES["EURUSD"])
