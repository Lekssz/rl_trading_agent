"""
Vanilla PPO Baseline on OHLCV-only data (with fees & no look-ahead)
- Trains PPO for EURUSD (2 bps) and BTCUSDT (4 bps)
- Reward uses new position; switch penalty reduces churn
- Adds distance-to-200-bar MA feature (no backfill -> no leakage)
- Observations EXCLUDE the current bar to avoid look-ahead
- Saves results to data/model/baseline/
"""

import os
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed


# --------------------
# 1) Load Data Helper
# --------------------
def load_ohlcv(path: str) -> pd.DataFrame:
    """Load OHLCV CSV, enforce UTC datetime index, and validate columns."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    # Require timestamp column
    if "timestamp" not in df.columns:
        raise ValueError(f"{path}: missing required 'timestamp' column.")

    # Parse timestamps to UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        bad = int(df["timestamp"].isna().sum())
        raise ValueError(f"{path}: {bad} bad timestamps (NaT). Clean before training.")
    df = df.set_index("timestamp").sort_index()

    # Require OHLCV columns
    cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    # Keep only OHLCV and cast to float
    df = df[cols].astype(float)

    # Check for NaNs
    if df.isna().any().any():
        raise ValueError(f"{path}: contains NaNs. Clean before training.")

    return df


# --------------------
# 2) Custom Env
# --------------------
@dataclass
class EnvConfig:
    window: int = 48                 # 1 day of 30-min bars
    fee_bps: float = 0.0             # set per asset below
    switch_penalty: float = 3e-4     # penalty per unit position change to reduce flip-flopping
    btc_mode: bool = True            # annualization convention for Sharpe
    eps: float = 1e-6                # numerical jitter


class ForexCryptoEnv(gym.Env):
    """
    State  = z-scored window of [OHLCV, dist_ma200]  (EXCLUDES current bar -> no look-ahead)
    Action = {-1, 0, +1}
    Reward = NEW position * return - trading_cost - switch_penalty
    """
    metadata = {"render_modes": []}

    def __init__(self, data: pd.DataFrame, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg

        # ---- Copy and add long-term trend feature (no backfill -> no future leakage) ----
        df = data.copy()
        ma200 = df["close"].rolling(200, min_periods=200).mean()
        df["dist_ma200"] = df["close"] / ma200 - 1.0
        df = df.dropna().copy()  # drop early rows until MA is defined
        self.df = df[["open", "high", "low", "close", "volume", "dist_ma200"]]

        # Prices & simple returns: (P_t - P_{t-1}) / P_{t-1}; returns[0] = 0
        self.prices = self.df["close"].to_numpy(dtype=np.float64)
        self.returns = np.zeros_like(self.prices, dtype=np.float64)
        self.returns[1:] = np.diff(self.prices) / np.clip(self.prices[:-1], 1e-8, None)

        # Observation & action spaces
        self.obs_shape = (self.cfg.window, self.df.shape[1])  # (48, 6)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=self.obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # Internal state
        self._pos = 0
        # Start so that the observation window [t-window, t) is valid
        self._t = max(self.cfg.window, 1)

        # Rolling stats for z-scoring (computed once; sliced per step)
        self._mean = self.df.rolling(self.cfg.window, min_periods=1).mean()
        self._std = self.df.rolling(self.cfg.window, min_periods=1).std().fillna(1.0)

    def _obs(self):
        # Use bars [t-window, t) -> exclude current bar to avoid look-ahead
        lo = self._t - self.cfg.window
        hi = self._t
        frame = self.df.iloc[lo:hi]
        z = (frame - self._mean.iloc[lo:hi]) / (self._std.iloc[lo:hi] + self.cfg.eps)
        # Clip extreme z-scores
        z = z.clip(self.observation_space.low[0, 0], self.observation_space.high[0, 0])
        return z.to_numpy(dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._pos = 0
        self._t = max(self.cfg.window, 1)
        return self._obs(), {}

    def step(self, action: int):
        # Map discrete action to target position
        pos_target = (-1, 0, +1)[int(action)]

        # Costs & penalty based on position change
        turnover = abs(pos_target - self._pos)              # 0, 1, or 2
        trading_cost = (self.cfg.fee_bps / 1e4) * turnover
        switch_penalty = self.cfg.switch_penalty * turnover

        # --- FIX: reward uses the PREVIOUS position on the bar that just closed (t-1 -> t)
        r_t = self.returns[self._t]
        reward = self._pos * r_t - trading_cost - switch_penalty

        # Now commit the NEW position for the NEXT bar (t -> t+1)
        self._pos = pos_target
        self._t += 1

        # Terminate after consuming the last available return
        terminated = self._t >= len(self.df)
        truncated = False

        info = {
            # This is the position AFTER the action (for turnover stats)
            "position": int(self._pos),
            "bar_return": float(r_t),
            "cost": float(trading_cost),
            "switch_penalty": float(switch_penalty),
        }
        return self._obs(), float(reward), terminated, truncated, info


# --------------------
# 3) Backtest Helpers
# --------------------
def run_once(env: gym.Env, model: PPO) -> pd.DataFrame:
    """Greedy backtest with the trained policy; returns per-bar metrics."""
    obs, _ = env.reset()
    rows = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        rows.append(
            {
                "time": env.df.index[env._t - 1],
                "reward": float(reward),
                "position": int(info.get("position", 0)),
                "bar_return": float(info.get("bar_return", 0.0)),
                "cost": float(info.get("cost", 0.0)),
            }
        )
        if terminated or truncated:
            break

    out = pd.DataFrame(rows).set_index("time")
    out["strategy_ret"] = out["reward"]
    out["equity"] = (1.0 + out["strategy_ret"]).cumprod()
    return out


def sharpe_annualized(returns: pd.Series, bars_per_day: int = 48, btc_mode: bool = True) -> float:
    """Annualized Sharpe from per-bar returns."""
    mu = float(returns.mean())
    sd = float(returns.std(ddof=1))
    ann_bars = bars_per_day * (365 if btc_mode else 252)
    return float(np.sqrt(ann_bars) * mu / (sd + 1e-12))


def max_drawdown(equity: pd.Series):
    """Max drawdown with start/end timestamps."""
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    mdd = float(dd.min())
    end = dd.idxmin()
    start = equity.loc[:end].idxmax()
    return mdd, start, end


# --------------------
# 4) Run Baseline for each asset
# --------------------
ASSETS = {
    "EURUSD":  "data/processed/ohlcv/oanda_EURUSD_30m_clean.csv",
    "BTCUSDT": "data/processed/ohlcv/binance_BTCUSDT_30m_clean.csv",
}
ASSET_FEES_BPS = {
    "EURUSD": 2.0,   # 2 bps FX
    "BTCUSDT": 4.0,  # 4 bps crypto
}

OUT_DIR = "data/model/baseline"
os.makedirs(OUT_DIR, exist_ok=True)

TOTAL_TIMESTEPS = int(1e5)   # baseline run
SEED = 42


def main():
    set_random_seed(SEED)
    summaries = []

    for asset, path in ASSETS.items():
        print(f"\n=== Training {asset} ===")
        df = load_ohlcv(path)

        # Sanity checks
        assert isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None, \
            "Timestamp index must be timezone-aware (UTC)."
        assert df.shape[0] > 500, f"Not enough rows in {path}."

        # Build env with asset-specific fees
        cfg = EnvConfig(
            btc_mode=(asset == "BTCUSDT"),
            fee_bps=ASSET_FEES_BPS[asset],
        )
        print(f"cfg: fee_bps={cfg.fee_bps}, switch_penalty={cfg.switch_penalty}, window={cfg.window}")

        # Train PPO with stable hyperparameters
        vec_env = DummyVecEnv([lambda: ForexCryptoEnv(df, cfg)])
        model = PPO(
            "MlpPolicy",
            vec_env,
            seed=SEED,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            learning_rate=2.5e-4,
            ent_coef=0.003,
            vf_coef=0.5,
            clip_range=0.2,
            gae_lambda=0.95,
            gamma=0.99,
            max_grad_norm=0.5,
        )
        # If using SB3 < 2.0, remove progress_bar=True
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

        # Save model
        model_path = os.path.join(OUT_DIR, f"ppo_baseline_{asset}.zip")
        model.save(model_path)

        # Evaluate (single greedy pass)
        results = run_once(ForexCryptoEnv(df, cfg), model)
        sharpe = sharpe_annualized(results["strategy_ret"], bars_per_day=48, btc_mode=(asset == "BTCUSDT"))
        mdd, mdd_start, mdd_end = max_drawdown(results["equity"])

        # Quick churn check
        avg_turnover = results["position"].diff().abs().fillna(0).mean()
        print(f"{asset} avg turnover per bar: {avg_turnover:.3f}")

        # Save artifacts
        ts_path = os.path.join(OUT_DIR, f"ppo_baseline_timeseries_{asset}.csv")
        results.to_csv(ts_path)

        summary = pd.DataFrame([{
            "asset": asset,
            "timesteps": TOTAL_TIMESTEPS,
            "seed": SEED,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "mdd_start": mdd_start,
            "mdd_end": mdd_end,
            "avg_turnover": avg_turnover,
            "model_path": model_path,
            "timeseries_path": ts_path,
        }])
        sum_path = os.path.join(OUT_DIR, f"ppo_baseline_summary_{asset}.csv")
        summary.to_csv(sum_path, index=False)
        summaries.append(summary)
        print(summary)

    # Combined summary
    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(os.path.join(OUT_DIR, "ppo_baseline_summary_all.csv"), index=False)
    print("\n=== Combined Summary ===")
    print(all_summary)


if __name__ == "__main__":
    main()