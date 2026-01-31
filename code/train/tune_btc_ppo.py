import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
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
TURNOVER_PENALTY = 0.01   # realism constraint

STATE_NPY = PROJECT_ROOT / "data/rl_states/btc/npy/state_full.npy"
PRICE_NPY = PROJECT_ROOT / "data/rl_states/btc/npy/close.npy"
TIME_NPY  = PROJECT_ROOT / "data/rl_states/btc/npy/timestamps.npy"

MODEL_DIR = PROJECT_ROOT / "models/ppo_btc_tuned"
RESULTS_DIR = PROJECT_ROOT / "results/ppo_btc_tuned"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Grid
LR = [3e-4, 1e-4]
N_STEPS = [1024, 2048]
BATCH = [64, 128]
GAMMA = [0.99, 0.995]
ENT_COEF = [0.0, 0.01]
CLIP_RANGE = [0.1, 0.2]

# ==============================
# HELPERS
# ==============================

def load_data():
    X = np.load(STATE_NPY)
    prices = np.load(PRICE_NPY)
    times = pd.to_datetime(np.load(TIME_NPY), utc=True)
    return X, prices, times

def make_env_slice(X, prices, times, start_ts, end_ts):
    idx = np.where((times >= start_ts) & (times <= end_ts))[0]
    return TradingPPOEnv(
        states=X, prices=prices, times=times.view("int64"),
        start_idx=int(idx[0]), end_idx=int(idx[-1]),
        trading_cost_bps=TRADING_COST_BPS,
        turnover_penalty=TURNOVER_PENALTY,
    )

def evaluate(model, env):
    obs, _ = env.reset()
    done = False
    eq = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        eq.append(info["equity"])

    lin = np.exp(np.array(eq))
    rets = np.diff(lin) / (lin[:-1] + 1e-12)
    sharpe = np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(len(rets))
    return sharpe

# ==============================
# MAIN
# ==============================

def main():
    X, prices, times = load_data()

    train_env = make_env_slice(X, prices, times,
                               "2019-01-01", "2021-12-31")
    val_env   = make_env_slice(X, prices, times,
                               "2022-01-01", "2022-12-31")

    venv = DummyVecEnv([lambda: train_env])

    results = []

    for lr, ns, bs, g, ent, clip in product(LR, N_STEPS, BATCH, GAMMA, ENT_COEF, CLIP_RANGE):
        print(f"Testing: lr={lr}, n_steps={ns}, batch={bs}, gamma={g}, ent={ent}, clip={clip}")

        model = PPO(
            "MlpPolicy", venv, seed=SEED, verbose=1,
            learning_rate=lr, n_steps=ns, batch_size=bs,
            gamma=g, ent_coef=ent, clip_range=clip
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        sharpe = evaluate(model, val_env)

        results.append([lr, ns, bs, g, ent, clip, sharpe])

    df = pd.DataFrame(results, columns=["lr","n_steps","batch","gamma","ent_coef","clip_range","val_sharpe"])
    df.to_csv(RESULTS_DIR / "tuning_results.csv", index=False)

    best = df.sort_values("val_sharpe", ascending=False).iloc[0]
    print("\nBEST CONFIG:\n", best)

    # Retrain best on 2019–2022
    full_train = make_env_slice(X, prices, times,
                                 "2019-01-01", "2022-12-31")
    venv = DummyVecEnv([lambda: full_train])

    model = PPO(
        "MlpPolicy", venv, seed=SEED, verbose=1,
        learning_rate=best.lr, n_steps=int(best.n_steps),
        batch_size=int(best.batch), gamma=best.gamma,
        ent_coef=best.ent_coef, clip_range=best.clip_range
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_DIR / "best_model.zip")

    # Final test on 2023
    test_env = make_env_slice(X, prices, times,
                               "2023-01-01", "2023-12-31")
    final_sharpe = evaluate(model, test_env)
    print(f"\nFinal 2023 Sharpe (Optimised BTC): {final_sharpe:.3f}")

if __name__ == "__main__":
    main()
