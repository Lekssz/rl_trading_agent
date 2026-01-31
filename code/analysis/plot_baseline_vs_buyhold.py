import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "visualisation/cross_asset_baseline_vs_buyhold"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Load Baseline PPO (Full Model)
# -------------------------------

btc_ppo = pd.read_csv(PROJECT_ROOT / "results/ppo_btc_baseline_ablations_full/metrics.csv")
eur_ppo = pd.read_csv(PROJECT_ROOT / "results/ppo_eur_baseline_ablations_full/metrics.csv")

# -------------------------------
# Buy & Hold (your computed values)
# -------------------------------

data = {
    "Strategy": [
        "PPO Baseline (BTC)",
        "Buy & Hold (BTC)",
        "PPO Baseline (EUR)",
        "Buy & Hold (EUR)"
    ],
    "Sharpe": [
        btc_ppo["sharpe"].iloc[0],
        2.415,   # BTC Buy & Hold
        eur_ppo["sharpe"].iloc[0],
        0.453    # EUR Buy & Hold
    ]
}

df = pd.DataFrame(data)

# -------------------------------
# Plot
# -------------------------------

plt.figure(figsize=(9, 5))
bars = plt.bar(df["Strategy"], df["Sharpe"], color="#4C72B0", edgecolor="black")

plt.axhline(0, linewidth=1)
plt.title("Baseline PPO vs Buy & Hold (2023)")
plt.ylabel("Sharpe Ratio")
plt.xticks(rotation=25)

# Value annotations
for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + (0.05 if h >= 0 else -0.05),
        f"{h:.2f}",
        ha="center",
        va="bottom" if h >= 0 else "top",
        fontsize=10
    )

plt.tight_layout()
plt.savefig(FIG_DIR / "baseline_vs_buyhold_sharpe.png", dpi=200)
plt.close()

