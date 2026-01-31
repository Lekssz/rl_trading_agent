import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "visualisation/robustness_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load metrics
btc_base = pd.read_csv(PROJECT_ROOT / "results/ppo_btc_baseline_ablations_full/metrics.csv")
btc_opt  = pd.read_csv(PROJECT_ROOT / "results/ppo_btc_optimised/metrics.csv")
eur_base = pd.read_csv(PROJECT_ROOT / "results/ppo_eur_baseline_ablations_full/metrics.csv")
eur_opt  = pd.read_csv(PROJECT_ROOT / "results/ppo_eur_optimised_from_btc/metrics.csv")

df = pd.DataFrame({
    "Model": [
        "Baseline PPO (BTC)",
        "Optimised PPO (BTC)",
        "Baseline PPO (EUR)",
        "BTC-Tuned PPO (EUR)"
    ],
    "Sharpe": [
        btc_base["sharpe"].iloc[0],
        btc_opt["sharpe"].iloc[0],
        eur_base["sharpe"].iloc[0],
        eur_opt["sharpe"].iloc[0],
    ]
})

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7),
                               gridspec_kw={"height_ratios": [1, 2]})

bars1 = ax1.bar(df["Model"], df["Sharpe"], color="#4C72B0", edgecolor="black")
bars2 = ax2.bar(df["Model"], df["Sharpe"], color="#4C72B0", edgecolor="black")

# Axis limits (adjust if needed)
ax1.set_ylim(-2, 1)        # Zoomed region (important comparisons)
ax2.set_ylim(-30, -5)      # Extreme outlier region

# Zero line (only meaningful on top axis)
ax1.axhline(0, linewidth=1, color="black")

# Titles & labels

ax2.set_ylabel("Sharpe Ratio")
plt.xticks(rotation=25)

# Diagonal break marks
d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# Value annotations (auto-select correct axis)
for bar in bars1:
    h = bar.get_height()
    ax = ax1 if h >= -2 else ax2
    ax.text(
        bar.get_x() + bar.get_width()/2,
        h + (0.05 if h >= 0 else -0.05),
        f"{h:.2f}",
        ha="center",
        va="bottom" if h >= 0 else "top",
        fontsize=10
    )



plt.tight_layout(rect=[0, 0, 1, 0.88])
fig.suptitle("Effect of Optimisation and Cross-Asset Transfer (2023 Test Period)",
             fontsize=14, fontweight="bold")

fig.text(0.5, 0.92, "Negative values indicate risk-adjusted underperformance.",
         ha="center", fontsize=11)
plt.savefig(FIG_DIR / "robustness_sharpe.png", dpi=200)
plt.close()

print("Sharpe robustness figure saved.")
