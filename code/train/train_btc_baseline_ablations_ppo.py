# PPO BTC Baseline Ablations Training Script
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "code"))

from stable_baselines3.common.vec_env import DummyVecEnv
from env.trading_env import TradingPPOEnv

# ==============================
# CONFIG
# ==============================

SEED = 42
TOTAL_TIMESTEPS = 200_000
TRADING_COST_BPS = 1.0
TURNOVER_PENALTY = 0.0

ABLATIONS = {
    "price":      "state_price.npy",
    "price_cnn":  "state_price_cnn.npy",
    "price_gdelt":"state_price_gdelt.npy",
    "price_cnn_gdelt": "state_price_cnn_gdelt.npy",
    "price_macro":"state_price_macro.npy",
    "full":       "state_full.npy",
}


EXPERIMENT_NAME = "ppo_btc_baseline_ablations"
PRICE_NPY = PROJECT_ROOT / "data/rl_states/btc/npy/close.npy"
TIME_NPY  = PROJECT_ROOT / "data/rl_states/btc/npy/timestamps.npy"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


# ==============================
# DATA
# ==============================

def load_data():
    prices = np.load(PRICE_NPY)
    times = pd.to_datetime(np.load(TIME_NPY), utc=True)
    return prices, times


# ==============================
# ENV
# ==============================

def make_env_slice(X, prices, times, start_ts, end_ts):
    idx = np.where((times >= start_ts) & (times <= end_ts))[0]
    return TradingPPOEnv(
        states=X,
        prices=prices,
        times=times.view("int64"),
        start_idx=int(idx[0]),
        end_idx=int(idx[-1]),
        trading_cost_bps=TRADING_COST_BPS,
        turnover_penalty=TURNOVER_PENALTY,
    )

# ==============================
# EVAL
# ==============================

def evaluate_strategy(model, env):
    obs, _ = env.reset()
    done = False
    records = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        records.append(info)

    df = pd.DataFrame(records)

    # Equity (stored as log equity in env)
    log_eq = df["equity"].to_numpy()
    lin_eq = np.exp(log_eq)

    rets = np.diff(lin_eq) / (lin_eq[:-1] + 1e-12)

    final_return = lin_eq[-1] / lin_eq[0] - 1

    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(len(rets)))

    running_max = np.maximum.accumulate(lin_eq)
    max_drawdown = float(np.max(1 - lin_eq / (running_max + 1e-12)))

    pos = df["position"].to_numpy()
    total_trades = int(np.sum(np.abs(np.diff(pos)) > 0))

    return (
    df,
    final_return,
    sharpe,
    max_drawdown,
    total_trades,
    )


# ==============================
# MAIN
# ==============================

def main():
    prices, times = load_data()

    for name, state_file in ABLATIONS.items():
        print(f"\n=== Running BTC Ablation: {name} ===")

        X = np.load(PROJECT_ROOT / "data/rl_states/btc/npy" / state_file)

        exp_dir = RESULTS_DIR / f"{EXPERIMENT_NAME}_{name}"
        model_dir = MODEL_DIR / f"{EXPERIMENT_NAME}_{name}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        train_env = make_env_slice(
            X, prices, times,
            pd.Timestamp("2019-01-01", tz="UTC"),
            pd.Timestamp("2022-12-31", tz="UTC"),
        )
        venv = DummyVecEnv([lambda: train_env])

        model = PPO("MlpPolicy", venv, verbose=0, seed=SEED)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(model_dir / "model.zip")

        test_env = make_env_slice(
            X, prices, times,
            pd.Timestamp("2023-01-01", tz="UTC"),
            pd.Timestamp("2023-12-31", tz="UTC"),
        )

        df, f_ret, sharpe, mdd, trades = evaluate_strategy(model, test_env)


        df.to_csv(exp_dir / "equity_curve.csv", index=False)

        pd.DataFrame([{
            "config": name,
            "final_return": f_ret,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "total_trades": trades,
        }]).to_csv(exp_dir / "metrics.csv", index=False)

        print(f"{name} done: Return={f_ret:.3f}, Sharpe={sharpe:.3f}, Trades={trades}")

    
if __name__ == "__main__":
    main()

