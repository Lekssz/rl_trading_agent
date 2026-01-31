# ✅ Project Setup Checklist

This checklist documents the environment, data, and execution steps required
to reproduce the experimental results of this dissertation.

---

## 🛠️ Environment & Setup
- [ ] Installed Miniconda / Conda
- [ ] Created environment from `environment.yml`
- [ ] Activated environment
- [ ] Verified Python and package versions
- [ ] Confirmed reproducibility on a clean environment

---

## 📂 Project Structure
- [ ] Repository cloned successfully
- [ ] Folder structure intact (`code/`, `data/`, `models/`, `results/`, `docs/`)
- [ ] All paths resolved correctly

---

## 🔗 Data Ingestion
- [ ] Retrieved Binance BTCUSDT OHLCV data
- [ ] Retrieved OANDA EURUSD OHLCV data
- [ ] Downloaded GDELT V2 GKG records
- [ ] Downloaded macroeconomic data from BLOOMBERG and FRED 
- [ ] Stored raw data in `data/raw/`

---

## 📊 Data Processing & Feature Engineering
- [ ] Cleaned and aligned OHLCV data
- [ ] Processed GDELT features (counts, tone, novelty, themes)
- [ ] Resampled macroeconomic indicators
- [ ] Applied lagging to non-price features
- [ ] Saved intermediate datasets to `data/processed/`

---

## 🧠 CNN Feature Extraction
- [ ] Trained CNN price model (BTC & EUR)
- [ ] Exported embeddings and directional probabilities
- [ ] Injected CNN outputs into RL state

---

## 🧮 RL State Construction
- [ ] Combined OHLCV, GDELT, macro, and CNN features
- [ ] Exported unified state as Parquet
- [ ] Converted Parquet states to NumPy arrays
- [ ] Stored NumPy states in `data/rl_states/`

---

## 🤖 PPO Training & Ablations
- [ ] Trained PPO baseline models (BTC & EUR)
- [ ] Executed feature ablation experiments
- [ ] Saved trained models to `models/`
- [ ] Stored evaluation metrics per model in `results/`

---

## 🔍 Hyperparameter Optimisation
- [ ] Performed grid search on BTCUSD
- [ ] Selected configuration based on Sharpe ratio
- [ ] Retrained BTC model using optimised parameters

---

## 🔁 Robustness & Cross-Asset Evaluation
- [ ] Transferred optimised BTC hyperparameters to EURUSD
- [ ] Evaluated robustness under identical settings
- [ ] Compared against Buy-and-Hold baseline
- [ ] Generated equity curves and comparison figures

---

## 📈 Results & Visualisation
- [ ] Verified metrics stored per model in `results/`
- [ ] Generated figures in `visualisation/`
- [ ] Cross-checked tables and plots against dissertation

---

## 📃 Ethics & Documentation
- [ ] Finalised `ETHICS.md`
- [ ] Finalised `DATA_LICENSE.md`
- [ ] Completed `project_summary.md`

---

✔️ This checklist reflects the **final, submitted artefact**.
