**Project Title:** **A Multi-Modal Reinforcement Learning Trading Agent**  
**Student Name:** Gbemileke Micah  
**Student ID:** K2457391  
**Institution:** Kingston University London  

---

## 1. Data Sources

This project uses only **publicly available, non-personal financial and news data**, specifically:

- **Binance API** – BTCUSDT historical OHLCV cryptocurrency market data
- **OANDA API** – EURUSD historical OHLCV foreign exchange data
- **GDELT V2 Global Knowledge Graph (GKG)** – public, aggregated news event metadata
- **Public macroeconomic indicators** (e.g., VIX, interest rates, CPI)

No private, sensitive, or personally identifiable information (PII) is collected, stored, or processed at any stage.

---

## 2. Data Protection & Privacy

- All datasets are **aggregated, non-personal, and anonymized at source**.
- GDELT data contains **event-level metadata only** and does not include individual user information.
- No human subjects, social media users, or identifiable entities are involved.
- Local development is conducted on encrypted devices:
  - macOS with **FileVault**
  - Windows with **BitLocker**
- No credentials or API keys are committed to version control.

This project complies with **UK GDPR** principles and Kingston University data governance guidance.

---

## 3. Experimental Scope & Transparency

- All trading decisions are executed in a **fully simulated reinforcement learning environment**.
- No live market interaction, brokerage connection, or real-money trading occurs.
- Agent actions, rewards, and evaluation metrics are logged and reproducible via stored result files.

---

## 4. Model Evaluation

- Models are evaluated using **historical backtesting only**.
- Performance metrics (e.g., return, Sharpe ratio, drawdown) are computed offline.
- Buy-and-Hold benchmarks are used solely for comparative analysis.

---

## 5. Compliance and Licensing

- All APIs are accessed under their **public or academic usage terms**.
- No private account endpoints or authenticated trading features are used.
- Third-party libraries (e.g., NumPy, PyTorch, Stable-Baselines3) are open source and listed in `LICENSES.md`.
- The project is conducted strictly for **academic research purposes**.

---

## 6. Live Trading Restriction

> This project does **not** engage in live trading.  
> It is strictly a **research and educational artefact**.

---

## 7. KUREC Ethics Position

This project does **not require formal KUREC ethics approval**.

Rationale:
- No human participants
- No personal or sensitive data
- No behavioural intervention
- No live financial activity

The project therefore qualifies as **low-risk** under Kingston University research ethics guidelines.

---

_Last updated: [JANUARY_2026]_
