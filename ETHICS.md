# ETHICS.md – Ethical Compliance and Audit Log

**Project Title:** AI Trading Agent – A Fractal-Risk-Aware PPO Framework  
**Student Name:** Gbemileke Micah  
**Student ID:** K2457391  
**Institution:** Kingston University London

---

## 1. Data Sources

This project uses publicly available and authorized data from:

- **OANDA API** (EUR/USD historical Forex data)
- **Binance API** (BTC/USD cryptocurrency data)
- **Twitter API v2 (Academic Research Track)** for financial sentiment
- **NewsAPI** as a fallback sentiment source during API outages

> ⚠️ No private, sensitive, or personally identifiable information (PII) is collected or stored.

---

## 2. Data Protection & Privacy

- All data is processed and stored within **Kingston University’s OneDrive** (encrypted and access-controlled).
- **Tweet metadata** (user handles, IDs, geolocation) is stripped during preprocessing.
- No raw tweet content is stored unless anonymized and sentiment-scored.
-All project data is processed and stored within **Kingston University’s OneDrive**, which is encrypted and institutionally managed.
- Local development occurs on both:
  - **Mac** – with **FileVault enabled**
  - **Lenovo (Windows)** – with **BitLocker enabled**
- Data retention policies comply with **GDPR** and university data governance best practices.

---

## 3. Transparency & Accountability

- All **agent actions**, including trades, pattern detections, and sentiment decisions, are **timestamped** and logged in `/logs/`.
- Sentiment fallback logic is in place:  
  > If Twitter API is down, fallback to NewsAPI headlines with the same sentiment analysis pipeline.
- This file is updated whenever workflows or data policies change.

---

## 4. Model Testing and Simulations

- All trading decisions are conducted in a **simulated environment** using historical or synthetic data only.
- **No real-money trading** or live order execution is involved at any point.
- Model evaluation includes:
  - Historical backtests
  - Crisis-mode stress tests
  - Monte Carlo simulations

---

## 5. Compliance and Licensing

-- All external APIs are used under their public or academic license terms. See `DATA_LICENSE.md` for full details.

### OANDA API Access
- OANDA demo account registered using **Kingston University email** for academic verification purposes.
- API access is limited to **public market data** (OHLCV, technical indicators) via the OANDA v20 REST API.
- No live trading, order placement, or financial transactions are performed.
- All credentials (Personal Access Token, Account ID) are stored securely in `.env` files, excluded from version control via `.gitignore`.
- API usage is fully compliant with OANDA's Terms of Service and Kingston University ethics guidelines.

### Binance API Access
- Binance API is accessed in **public mode** for OHLCV historical cryptocurrency market data.
- No API keys, authentication, or private account endpoints are used.
- No account balances, private data, or live trading functions are accessed.
- API usage is fully compliant with Binance Terms of Service and Kingston University ethics guidelines.

- Third-party libraries (e.g., PyTorch, FinBERT, Stable-Baselines3, Optuna) are open source and documented in `LICENSES.md`.
- Project code is for **academic research only**.
- Final license: **MIT (to be confirmed at submission)**



---

## 6. Live Trading Restriction

> This project is for **research and educational purposes only**.  
> **No live trading** will be conducted.

---

## 7. KUREC Position

## Ethics Approval Status

KUREC ethics approval was reviewed and deemed unnecessary.

Based on the final project scope  which involves only **public, anonymized, and synthetic data**, and does **not engage human subjects or collect personal information** — this project qualifies as **low-risk** and is **exempt from formal KUREC review**.

All ethical safeguards, data compliance measures, and fallback procedures are documented in this file in accordance with Kingston University’s research ethics guidelines.


---

_Last updated: 6 August 2025_
