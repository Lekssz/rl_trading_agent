

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "visualisation/robustness_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_eq(path):
    df = pd.read_csv(path)
    return pd.to_datetime(df["time"]), np.exp(df["equity"].values)

# ---- BTC ----
t_btc_base, eq_btc_base = load_eq(PROJECT_ROOT / "results/ppo_btc_baseline_ablations_full/equity_curve.csv")
t_btc_opt,  eq_btc_opt  = load_eq(PROJECT_ROOT / "results/ppo_btc_optimised/equity_curve.csv")

plt.figure(figsize=(10, 6))
plt.plot(t_btc_base, eq_btc_base, label="Baseline PPO (BTC)")
plt.plot(t_btc_opt,  eq_btc_opt,  label="Optimised PPO (BTC)")
plt.title("BTCUSD Equity Comparison (2023 Test Period)", fontweight="bold")
plt.xlabel("Time (2023)")
plt.ylabel("Equity")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "btc_equity_robustness.png", dpi=200)
plt.close()

# ---- EUR ----
t_eur_base, eq_eur_base = load_eq(PROJECT_ROOT / "results/ppo_eur_baseline_ablations_full/equity_curve.csv")
t_eur_opt,  eq_eur_opt  = load_eq(PROJECT_ROOT / "results/ppo_eur_optimised_from_btc/equity_curve.csv")

plt.figure(figsize=(10, 6))
plt.plot(t_eur_base, eq_eur_base, label="Baseline PPO (EUR)")
plt.plot(t_eur_opt,  eq_eur_opt,  label="BTC-Tuned PPO (EUR)")
plt.title("EURUSD Equity Comparison (2023 Test Period)", fontweight="bold")
plt.xlabel("Time (2023)")
plt.ylabel("Equity")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "eur_equity_robustness.png", dpi=200)
plt.close()

print("Equity robustness figures saved.")


# ---- OPTIMISED PPO: BTC vs EUR ----

t_btc_opt, eq_btc_opt = load_eq(PROJECT_ROOT / "results/ppo_btc_optimised/equity_curve.csv")
t_eur_opt, eq_eur_opt = load_eq(PROJECT_ROOT / "results/ppo_eur_optimised_from_btc/equity_curve.csv")

plt.figure(figsize=(10, 6))
plt.plot(t_btc_opt, eq_btc_opt, label="Optimised PPO (BTC)")
plt.plot(t_eur_opt, eq_eur_opt, label="BTC-Tuned PPO (EUR)")
plt.title("Optimised PPO Cross-Asset Behaviour (2023 Test Period)", fontweight="bold")
plt.xlabel("Time (2023)")
plt.ylabel("Equity")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "optimised_cross_asset_equity.png", dpi=200)
plt.close()

print("Optimised cross-asset equity figure saved.")
