# Project Summary  
**A Multi-Modal Reinforcement Learning Trading Agent**

## 1. Data Acquisition

**Market Data**
- Binance BTCUSDT OHLCV (30-minute bars, 2019–2023)
- OANDA EURUSD OHLCV (30-minute bars, 2019–2023)
- Data cleaned, deduplicated, UTC-aligned, and gap-inspected

**Macroeconomic Data**
- VIX Index (from **Bloomberg**)
- Federal Funds Rate(from **FRED**)
- ECB Deposit Facility Rate(from **FRED**)
- Euro CPI YoY  (from **Bloomberg**)
(All resampled to 30-minute resolution and forward-filled)

**News Data**
- GDELT V2 GKG events collected via BigQuery
- Aggregated into 30-minute bins:
  - Article counts
  - Tone and median tone
  - Domain diversity
  - 24h / 7d rolling attention
  - Novelty (z-score burst detection)
  - Thematic indicators (regulation, exchange events, hacks, stablecoins, network/mining)

---

## 2. Feature Engineering and Unified State Construction

**Preprocessing**
- BTC gaps interpolated where appropriate
- EURUSD weekend gaps separated from genuine missing data
- All series aligned to exact 30-minute bar closes

**Feature Sets**
- Price-derived features (returns, ATR, CLV, time encodings)
- GDELT attention-based news features (winsorised, log-transformed)
- Macro indicators and change flags
- CNN-derived price embeddings and short-horizon directional probability

**Unified State**
- OHLCV + GDELT + Macro + CNN features merged at each timestep
- All non-price features lagged by one bar (no look-ahead)
- Unified state exported as Parquet for reproducibility

---

## 3. CNN Price Feature Extractor

- Custom 1D dilated CNN with residual blocks
- Input: 128 × 30-minute historical bars
- Output:
  - 1-hour directional probability
  - 64-dimensional dense embedding
- Trained independently on BTCUSD and EURUSD
- Outputs saved and later injected into the RL state

---

## 4. RL State Preparation

- Final Parquet state files converted into **NumPy arrays**
- NumPy states used directly by the PPO trading environment
- Ensures fast loading, reproducibility, and separation of data engineering from training

---

## 5. Baseline PPO Training and Feature Ablations

Using a **fixed PPO configuration**, baseline experiments were conducted on both assets.
  **Key detail**: The baseline model had no turnover penalty, which means that the agent was not penalized for frequently trading. This is important for understanding the agent's trading behavior and performance in the absence of transaction cost considerations.
  ### Transaction Cost Configuration

  The baseline model included 1 bps for transaction costs, as set in the training script (TRADING_COST_BPS = 1.0). Although the environment constructor defaults to 5 bps, the script passes 1 bps as the configuration parameter, which was used during training and evaluation. The turnover penalty was set to 0.0 in the baseline model to avoid penalizing frequent trading.
**BTCUSD Ablations**
- Price-only
- Price + CNN
- Price + GDELT
- Price + CNN + GDELT
- Price + Macro
- Full multi-modal state

**EURUSD Ablations**
- Same ablation structure as BTCUSD

These experiments isolate the contribution of each data modality under identical training conditions.

---

## 6. Hyperparameter Optimisation (BTC Only)

- Grid search performed on BTCUSD
- Train: 2019–2021  
- Validation: 2022  
- Test: 2023 (held out)

**Search Space**
- Learning rate
- n_steps
- Batch size
- Discount factor
- Entropy coefficient
- PPO clip range

Hyperparameters selected based on **validation Sharpe ratio**, then fixed for robustness testing.

---

## 7. Robustness and Cross-Asset Transfer

- Optimised BTC configuration retrained on extended data
- Same hyperparameters transferred **unchanged** to EURUSD
- Evaluates cross-market generalisation under identical settings
- Buy-and-Hold benchmarks computed for both assets

## 8. Key Logs and Observations

This section provides a summary of key logs that capture the setup, data processing, and model training details for this project.

### 1. Project Setup and Environment Configuration

 **The Conda environment** was set up using the `environment.yml` file, ensuring the correct installation of dependencies such as **PyTorch**, **Stable-Baselines3**, and **pandas**.
- Data from **Binance**, **OANDA**, **GDELT**, **FRED**, and **Bloomberg** were ingested and processed. 
  **GDELT** data was downloaded directly from BigQuery and sent straight to the data/processed/ directory for immediate use.
  **Binance** and **OANDA** data were processed, cleaned, and stored in the data/raw/ directory before being moved to data/processed/.
  **FRED** and **Bloomberg** macroeconomic data were also processed and stored in data/processed/.


### 2. Data Processing and Feature Engineering

- **BTCUSD** and **EURUSD** datasets were merged with **GDELT event metadata** and **macroeconomic indicators**.
- Gaps in the BTCUSD data were **interpolated**, and **weekend gaps** in EURUSD were handled separately.
- Features derived from price (returns, ATR), news sentiment (from GDELT), and macroeconomic indicators were engineered and merged into the final state for training.
- Data was saved as **Parquet files**, converted to **NumPy arrays**, and fed into the PPO model.

### 3. Model Training and Evaluation

- **Baseline PPO training** was carried out on both BTCUSD and EURUSD data using fixed parameters. Ablation studies were performed to assess the impact of different data features (e.g., price-only, price + CNN, price + GDELT, etc.).
- **Hyperparameter tuning** was performed for BTCUSD using grid search, optimizing for Sharpe ratio on the 2022 validation set.
- The best-performing configuration was transferred unchanged to EURUSD for robustness testing.


### 4. Robustness and Cross-Asset Transfer

- The trained model on BTCUSD was retrained using extended data and its hyperparameters were transferred to EURUSD to evaluate cross-market generalisation under identical settings.
- **Buy-and-Hold** benchmarks were computed for both BTCUSD and EURUSD as a reference for comparison.

---

## 9. Results and Metrics

- **All evaluation metrics are stored per model in the `results/` directory**
- Each model folder contains:
  - Return
  - Sharpe ratio
  - Maximum drawdown
  - Trading activity
  - Equity time series (where applicable)

- Figures and comparisons are generated from these result files and stored separately under `visualisation/`

---

## 10. Reproducibility and Organisation

- Models stored in `models/`
- Metrics stored in `results/<model_name>/`
- Training, preprocessing, and evaluation scripts modularised under `code/`
- logs, and environment configuration included
- Project fully reproducible from raw data ingestion to final figures

---

This project summary documents the **complete experimental artefact**, aligned with the submitted dissertation.
