import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

STATE_DIR = Path("data/processed/state")

BTC_PATH = STATE_DIR / "btcusd_state_with_macro_30m_2019_2023.parquet"
EUR_PATH = STATE_DIR / "eurusd_state_with_macro_30m_2019_2023.parquet"

SUMMARY_BTC = STATE_DIR / "btcusd_state_summary.txt"
SUMMARY_EUR = STATE_DIR / "eurusd_state_summary.txt"

CORR_BTC = STATE_DIR / "btcusd_corr.png"
CORR_EUR = STATE_DIR / "eurusd_corr.png"


def analyze(df: pd.DataFrame, tag: str):
    """
    Generate:
    - basic stats
    - missing values
    - feature grouping
    - correlation plot
    - dissertation-ready text summary
    """

    # ---- Feature categories ----
    price_cols = ["open", "high", "low", "close", "volume"]
    price_cols = [c for c in price_cols if c in df.columns]

    # News = everything from Step 4B except timestamp, OHLCV, macro
    macro_cols = [c for c in df.columns if c.startswith("ecb_") or
                                             c.startswith("fedfunds") or
                                             c.startswith("euro_cpi_yoy") or
                                             c.startswith("vix")]
    macro_cols = sorted(macro_cols)

    known_cols = price_cols + macro_cols + ["timestamp"]
    news_cols = sorted([c for c in df.columns if c not in known_cols])

    # ---- Missing values ----
    missing_summary = df.isna().sum()

    # ---- Correlation matrix ----
    # numeric only
    num_df = df.drop(columns=["timestamp"], errors="ignore")
    num_df = num_df.select_dtypes(include=[np.number])

    plt.figure(figsize=(16, 12))
    sns.heatmap(num_df.corr(), cmap="coolwarm", center=0)
    plt.title(f"{tag} – Feature Correlation Heatmap")
    plt.tight_layout()

    if tag == "BTCUSD":
        plt.savefig(CORR_BTC)
    else:
        plt.savefig(CORR_EUR)

    # ---- Summary text ----
    summary = []
    summary.append(f"===== {tag} STATE SUMMARY =====")
    summary.append(f"Rows: {len(df):,}")
    summary.append(f"Columns: {len(df.columns):,}")
    summary.append(f"Date Range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    summary.append("")

    summary.append("---- FEATURE CATEGORIES ----")
    summary.append(f"OHLCV Columns ({len(price_cols)}): {price_cols}")
    summary.append(f"News Columns ({len(news_cols)}): {news_cols[:10]} ... total={len(news_cols)}")
    summary.append(f"Macro Columns ({len(macro_cols)}): {macro_cols}")
    summary.append("")

    summary.append("---- MISSING VALUES (should be zero) ----")
    summary.append(str(missing_summary[missing_summary > 0]))
    summary.append("If empty → no missing values after processing.")
    summary.append("")

    summary.append("---- MEMORY FOOTPRINT ----")
    summary.append(df.memory_usage(deep=True).to_string())
    summary.append("")

    summary.append("---- NOTES ----")
    summary.append("• All news & macro features lagged by +1 bar (look-ahead safe).")
    summary.append("• OHLCV bars aligned to bar-end (clean 30-min grid).")
    summary.append("• Macro values forward-filled between releases.")
    summary.append("• Change flags mark when macro series updated.")
    summary.append("")

    return "\n".join(summary)


def run(asset_path: Path, out_path: Path, tag: str):
    df = pd.read_parquet(asset_path)
    text = analyze(df, tag)
    out_path.write_text(text)
    print(f"Saved summary → {out_path}")


if __name__ == "__main__":
    run(BTC_PATH, SUMMARY_BTC, "BTCUSD")
    run(EUR_PATH, SUMMARY_EUR, "EURUSD")

    print("Correlation plots saved.")
