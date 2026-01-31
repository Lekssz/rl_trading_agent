# Multi-Modal Reinforcement Learning Trading Agent

This repository contains the full experimental artefact for an MSc Data Science dissertation at  
**Kingston University London**.

The project investigates whether **multi-modal data** (price, news attention, macroeconomic indicators, and CNN-derived price embeddings) improves the performance and stability of a **PPO-based reinforcement learning trading agent** across cryptocurrency (BTCUSD) and foreign exchange (EURUSD) markets.

---

## 📌 Project Scope

- Offline, **simulation-only** reinforcement learning
- No live trading or real-money execution
- Historical backtesting using 30-minute OHLCV data
- Controlled ablation studies, hyperparameter tuning, and robustness evaluation
- Cross-asset transfer from BTCUSD to EURUSD

---

## 📂 Repository Structure

A detailed overview of the project layout is available here:  
👉 [`Project Structure Overview`](Project%20Structure%20Overview.md)


---

## 📄 Documentation (`docs/`)

All supporting documentation for ethics, licensing, provenance, and reproducibility is located in:

docs/
├── project_summary.md # High-level description of the artefact and experiments
├── project_setup_checklist.md # Reproducibility and setup checklist
├── ETHICS.md # Ethics and compliance statement
├── DATA_LICENSE.md # Data source usage and licensing
└── LICENSES.md # Third-party software licenses


---

## 🧠 Method Overview

1. **Data Sources**
   - Binance (BTCUSDT OHLCV)
   - OANDA (EURUSD OHLCV)
   - GDELT V2 GKG (global news event metadata)
   - Macroeconomic indicators (FRED, Bloomberg – institutional access)

2. **Feature Engineering**
   - Price-derived indicators
   - News attention and thematic signals
   - Macroeconomic context
   - CNN-based price embeddings and short-horizon direction probability

3. **State Construction**
   - Unified multi-modal state
   - Exported as Parquet, then converted to NumPy arrays
   - NumPy states consumed directly by the RL environment

4. **Reinforcement Learning**
   - Proximal Policy Optimisation (PPO)
   - Baseline training and feature ablations (BTC & EUR)
   - Hyperparameter tuning on BTC only
   - Cross-asset robustness testing on EUR
   - Buy-and-Hold benchmark comparison

---

## 📊 Results

- Evaluation metrics (return, Sharpe ratio, drawdown, trading activity) are stored **per model** in:
results/<model_name>/

- Figures and plots used in the dissertation are stored in:
visualisation/


---

## ⚖️ Ethics and Compliance

- No human subjects
- No personal or sensitive data
- No live trading
- No financial advice
- All data used under public or institutional academic licenses

See `docs/ETHICS.md` and `docs/DATA_LICENSE.md` for full details.

---

## 🔁 Reproducibility

- Environment specification is provided in `environment.yml`
- Code, data processing, and evaluation pipelines are modularised
- Final artefact is reproducible from raw data ingestion to result generation


---

## 🎓 Academic Context

This repository accompanies an **experiment-based MSc dissertation** submitted in partial fulfilment of the requirements for the **MSc Data Science** programme at Kingston University London.

---

**Author:** Gbemileke Micah  
**Student ID:** K2457391  