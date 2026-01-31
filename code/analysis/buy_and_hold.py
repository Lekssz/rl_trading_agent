import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSET = "eur"   # change to "eur" for EURUSD

PRICE_NPY = PROJECT_ROOT / f"data/rl_states/{ASSET}/npy/close.npy"
TIME_NPY  = PROJECT_ROOT / f"data/rl_states/{ASSET}/npy/timestamps.npy"

def evaluate_buy_and_hold():
    prices = np.load(PRICE_NPY)
    times = pd.to_datetime(np.load(TIME_NPY), utc=True)

    # 2023 test slice
    mask = (times >= "2023-01-01") & (times <= "2023-12-31")
    p = prices[mask]

    c = 0.0001  # 1 bps

    equity = (p / p[0]) * (1 - c) / (1 + c)

    rets = np.diff(equity) / (equity[:-1] + 1e-12)

    final_return = equity[-1] - 1
    sharpe = np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(len(rets))

    running_max = np.maximum.accumulate(equity)
    max_drawdown = np.max(1 - equity / (running_max + 1e-12))

    print(f"\nBUY & HOLD {ASSET.upper()} (2023)")
    print(f"Return       : {final_return*100:.2f}%")
    print(f"Sharpe       : {sharpe:.3f}")
    print(f"Max Drawdown : {max_drawdown*100:.2f}%")

if __name__ == "__main__":
    evaluate_buy_and_hold()
