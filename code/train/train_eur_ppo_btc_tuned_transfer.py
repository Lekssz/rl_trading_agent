
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "code"))

from env.trading_env import TradingPPOEnv

# ==============================
# CONFIG (BTC-TUNED)
# ==============================

SEED = 42
TOTAL_TIMESTEPS = 200_000

LR = 3e-4
N_STEPS = 1024
BATCH_SIZE = 64
GAMMA = 0.99
ENT_COEF = 0.0
CLIP_RANGE = 0.2

TRADING_COST_BPS = 1.0
TURNOVER_PENALTY = 0.01   

EXPERIMENT_NAME = "ppo_eur_optimised_from_btc"

STATE_NPY = PROJECT_ROOT / "data/rl_states/eur/npy/state_full.npy"
PRICE_NPY = PROJECT_ROOT / "data/rl_states/eur/npy/close.npy"
TIME_NPY  = PROJECT_ROOT / "data/rl_states/eur/npy/timestamps.npy"
MODEL_DIR = PROJECT_ROOT / "models" / EXPERIMENT_NAME
RESULTS_DIR = PROJECT_ROOT / "results" / EXPERIMENT_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    log_eq = df["equity"].to_numpy()
    lin_eq = np.exp(log_eq)

    rets = np.diff(lin_eq) / (lin_eq[:-1] + 1e-12)

    final_return = lin_eq[-1] / lin_eq[0] - 1
    sharpe = np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(len(rets))
    mdd = np.max(1 - lin_eq / np.maximum.accumulate(lin_eq))

    trades = int(np.sum(np.abs(np.diff(df["position"])) > 0))

    return final_return, sharpe, mdd, trades, df

# ==============================
# MAIN
# ==============================

def main():
    X = np.load(STATE_NPY)
    prices = np.load(PRICE_NPY)
    times = pd.to_datetime(np.load(TIME_NPY), utc=True)

    train_env = DummyVecEnv([lambda: make_env_slice(
        X, prices, times,
        pd.Timestamp("2019-01-01", tz="UTC"),
        pd.Timestamp("2022-12-31", tz="UTC"),
    )])

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=LR,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        clip_range=CLIP_RANGE,
        verbose=1,
        seed=SEED,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_DIR / "model.zip")

    test_env = make_env_slice(
        X, prices, times,
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2023-12-31", tz="UTC"),
    )

    ret, sharpe, mdd, trades, df = evaluate_strategy(model, test_env)

    df.to_csv(RESULTS_DIR / "equity_curve.csv", index=False)
    pd.DataFrame([{
        "final_return": ret,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "total_trades": trades,
    }]).to_csv(RESULTS_DIR / "metrics.csv", index=False)

    print("\n=== EUR Optimised-from-BTC Results (2023) ===")
    print(f"Return: {ret:.3f}")
    print(f"Sharpe: {sharpe:.3f}")
    print(f"Max DD: {mdd:.3f}")
    print(f"Trades: {trades}")

if __name__ == "__main__":
    main()
