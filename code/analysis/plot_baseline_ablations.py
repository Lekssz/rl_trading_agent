import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# CONFIG
# ==============================

ASSET = "btc"   # change to "eur" when needed
EXPERIMENT = f"ppo_{ASSET}_baseline_ablations"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

FIG_DIR = PROJECT_ROOT / f"visualisation/{ASSET}_baseline_ablation_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


PLOT_STYLE = {
    "title_size": 14,
    "label_size": 12,
    "tick_size": 11,
    "line_width": 2.2,
    "grid_alpha": 0.3,
}

ABLATIONS = [
    "price",
    "price_cnn",
    "price_gdelt",
    "price_cnn_gdelt",
    "price_macro",
    "full",
]

LABEL_MAP = {
    "price": "Price Only",
    "price_cnn": "Price + CNN",
    "price_gdelt": "Price + GDELT",
    "price_cnn_gdelt": "Price + CNN + GDELT",
    "price_macro": "Price + Macro",
    "full": "Full Model",
}

# ==============================
# LOAD DATA
# ==============================

equity_curves = {}
metrics = []

for name in ABLATIONS:
    exp_dir = RESULTS_DIR / f"{EXPERIMENT}_{name}"
    df_eq = pd.read_csv(exp_dir / "equity_curve.csv")
    df_m  = pd.read_csv(exp_dir / "metrics.csv")

    equity_curves[name] = np.exp(df_eq["equity"].values) * 100
    metrics.append(df_m.iloc[0])

metrics_df = pd.DataFrame(metrics)
time = pd.to_datetime(df_eq["time"])

# ==============================
# 1) OVERLAY EQUITY CURVES 
# ==============================

plt.figure(figsize=(11, 7))

for name in ABLATIONS:
    plt.plot(
        time,
        equity_curves[name],
        linewidth=PLOT_STYLE["line_width"],
        label=LABEL_MAP[name]

    )

plt.title(f"{ASSET.upper()}USD PPO Equity Curves by Feature Configuration (2023 TEST Period)",
          fontsize=PLOT_STYLE["title_size"], fontweight="bold")
plt.xlabel("Time (2023)", fontsize=PLOT_STYLE["label_size"], fontweight="bold")
plt.ylabel("Equity (% of Initial Capital)", fontsize=PLOT_STYLE["label_size"], fontweight="bold")
plt.xticks(fontsize=PLOT_STYLE["tick_size"])
plt.yticks(fontsize=PLOT_STYLE["tick_size"])
plt.grid(True, alpha=PLOT_STYLE["grid_alpha"])
plt.legend(fontsize=10, frameon=True)
plt.tight_layout()
plt.savefig(FIG_DIR / f"{ASSET}_equity_overlay.png", dpi=200)
plt.close()

# ==============================
# 2) SHARPE BAR CHART
# ==============================

# Sort by Sharpe (best → worst)
metrics_sorted = metrics_df.sort_values("sharpe", ascending=False)
metrics_sorted["config"] = metrics_sorted["config"].replace(LABEL_MAP)

colors = "steelblue" 
plt.figure(figsize=(10, 5))
plt.bar(metrics_sorted["config"], metrics_sorted["sharpe"], color=colors)

plt.title(f"{ASSET.upper()}USD Sharpe Ratio by Feature Configuration (2023 Test Period)",
          fontsize=PLOT_STYLE["title_size"], fontweight="bold")
plt.ylabel("Sharpe Ratio", fontsize=PLOT_STYLE["label_size"], fontweight="bold")
plt.xticks(rotation=30, fontsize=10)
plt.yticks(fontsize=11)

# Zero baseline
plt.axhline(0, linewidth=1, color="black")
plt.grid(axis="y", alpha=0.3)

# Value labels
for i, v in enumerate(metrics_sorted["sharpe"]):
    plt.text(i, v, f"{v:.2f}",
             ha="center",
             va="bottom" if v >= 0 else "top",
             fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(FIG_DIR / f"{ASSET}_sharpe_bar_sorted.png", dpi=200)
plt.close()



# ==============================
# LOG EQUITY CHART 
# ==============================
plt.figure(figsize=(11, 7))
for name in ABLATIONS:
    plt.plot(time, np.log(equity_curves[name] / 100), linewidth=2, label=LABEL_MAP[name])

plt.title(f"{ASSET.upper()}USD PPO Log-Equity (2023 Test Period)", fontweight="bold")
plt.xlabel("Time (2023)")
plt.ylabel("Log Equity")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / f"{ASSET}_log_equity_overlay.png", dpi=200)
plt.close()

print("Figures saved successfully.")