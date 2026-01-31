# DATA_LICENSE.md — Data Sources and Usage Terms

**Student:** Gbemileke Micah (K2457391)  
**Programme:** MSc Data Science, Kingston University London  

This document records the datasets used in the project and summarises licensing and usage compliance.
All data is used strictly for **non-commercial academic research** and **offline simulation/backtesting**.

---

## 1. Market Data

### Binance (Cryptocurrency OHLCV)
- **Data used:** BTCUSDT historical OHLCV (30-minute bars)
- **Access method:** Public Binance REST endpoints (no private endpoints; no account data)
- **Purpose:** Market price/volume series for RL environment and evaluation
- **Compliance:** Only public market data is accessed; no authentication keys or private user data are used.

### OANDA (FX OHLCV)
- **Data used:** EURUSD historical OHLCV (30-minute bars)
- **Access method:** OANDA v20 REST API (demo account)
- **Purpose:** Market price/volume series for RL environment and evaluation
- **Compliance:** Access is limited to historical market data; **no live trading**, order placement, or financial transactions are performed.

---

## 2. News and Event Data

### GDELT V2 Global Knowledge Graph (GKG)
- **Data used:** Event-level global news metadata (counts, themes, tone, attention measures)
- **Access method:** Public dataset access (e.g., BigQuery / GDELT tools)
- **Purpose:** Construction of exogenous news-attention features aligned to 30-minute intervals
- **Compliance:** GDELT is a public dataset and contains no personal user data collected by this project.

---

## 3. Macroeconomic Data

### FRED (Federal Reserve Economic Data)
- **Data used:** Public macroeconomic indicators (e.g., interest rates, inflation series)
- **Access method:** Public FRED datasets
- **Purpose:** Macroeconomic context features, resampled to 30-minute resolution
- **Compliance:** Data is openly available and used in accordance with FRED’s public data terms.

### Bloomberg
- **Data used:** Selected macro-financial indicators (e.g., VIX index, policy rates)
- **Access method:** Bloomberg Terminal access provided through Kingston University
- **Purpose:** Supplementary macroeconomic features for academic analysis
- **Compliance:** Data accessed under **institutional academic license**.
  Bloomberg data is **not redistributed** and is used solely for research purposes.

---

## 4. Data Handling and Redistribution

- All data used is historical and non-personal.
- No personally identifiable information (PII) is collected, stored, or processed.
- Raw proprietary datasets (e.g., Bloomberg) are **not redistributed**.
- Derived features and aggregated statistics are used only for academic reporting.

---

## 5. Academic Use Disclaimer

This project is an MSc dissertation artefact at Kingston University London.
It does **not** provide financial advice, does **not** execute live trading, and is **not intended for commercial use**.

---

_Last updated: [JANUARY/ 2026]_
