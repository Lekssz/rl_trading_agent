# 🧾 Project Setup Log

> MSc Data Science Dissertation – Gbemileke Micah  
> Project: AI-Driven Financial Trading Agent  
> Repository: https://github.com/<your-username>/AI-Trading-Agent  
> Start Date: August 03, 2025

---

## ✅ ENVIRONMENT SETUP

| Step | Description | Date | Status |
|------|-------------|------|--------|
| 1. | Installed Miniconda (x86_64) | Aug 2, 2025 | ✅ |
| 2. | Verified `conda --version` | Aug 2, 2025 | ✅ |
| 3. | Created `environment.yml` with all dependencies | Aug 2, 2025 | ✅ |
| 4. | Ran `conda env create -f environment.yml` | Aug 2, 2025 | ✅ |
| 5. | Activated environment `ai-trading-agent` | Aug 2, 2025 | ✅ |

---

## 📂 PROJECT STRUCTURE

| Folder | Purpose |
|--------|---------|
| `data/` | Raw and processed data (ignored by Git) |
| `models/` | Saved models and checkpoints |
| `code/` | Main logic and RL agent training |
| `dashboard/` | Streamlit app and visualizations |
| `docs/` | Proposal, Gantt chart, logs, diagrams |
| `notebooks/` | Exploratory data analysis (Jupyter) |
| `ingestion/` | Scripts to fetch OHLCV, tweets, news |
| `feature_engineering/` | Pattern detection and indicators |
| `risk_control/` | Fractal volatility and filters |
| `utils/` | Reusable helpers and config logic |
| `tests/` | Optional unit tests |

---

## 🔐 ETHICS & LICENSES

| File | Purpose | Status |
|------|---------|--------|
| `ETHICS.md` | Ethical protocol, anonymization | ✅ Created |
| `DATA_LICENSE.md` | API terms and review log | ✅ Created |
| `LICENSES.md` | 3rd-party software licenses | ✅ Created |

---

## 🔁 GIT & VERSION CONTROL

| Action | Date | Notes |
|--------|------|-------|
| Initialized Git repo | Aug 1, 2025 | `git init` |
| Added `.gitignore` | Aug 1, 2025 | Covers models, data, env, etc. |
| Committed base structure | Aug 1, 2025 | All folders & placeholder files |
| Added `.gitkeep` to empty folders | Aug 1, 2025 | Allows structure tracking |
| Verified gitignore rules using `git check-ignore` | Aug 2, 2025 | ✅ |

---

## 🔖 NOTES / ISSUES

- Miniconda installed using `.sh` installer successfully
- `conda` wasn't recognized initially — fixed after restarting terminal
- `.gitignore` setup complete and tested
- Next step: Start ingestion + OHLCV formatting

## 📝 Auto Log Entry — 2025-08-03 21:36:06

- [ ] Ran: `conda env create -f environment.yml`
- [ ] Verified `conda activate ai-trading-agent`
- [ ] Confirmed folder structure
- [ ] Checked `.gitignore` with `git check-ignore -v`


---

## 🔗 GitHub Repository Setup

- [x] Initialized Git repository locally (`git init`)
- [x] Created `.gitignore` and added appropriate rules
- [x] Added initial project structure and `.gitkeep` files
- [x] Committed setup files and environment config
- [x] Installed and used GitHub Desktop app
- [x] Created a repository on GitHub
- [x] Published the repository using GitHub Desktop
- [ ] Verified repository privacy settings

**Repository Name:** `AI-Trading-Agent`  
**Created on:** August 3, 2025  
**Published via:** GitHub Desktop (macOS)  


## 🛠️ Conda Environment

- [x] Created `environment.yml` file
- [x] Installed Miniconda
- [x] Created conda environment from YAML
- [x] Activated environment and tested installation
*Created on:** August 3, 2025  


## 📝 Auto Log Entry — 2025-08-06 20:25:18

- [ ] Added requests to environment.yml for API ingestion scripts

## 📝 Auto Log Entry — 2025-08-06 20:30:20

- [ ] Added requests to environment.yml for API ingestion scripts

## 📝 Auto Log Entry — 2025-08-06 21:45:34

- [ ] Downloaded Binance BTCUSDT 30m OHLCV data from 2019–2021 using public API

## 📝 Auto Log Entry — 2025-08-06 22:27:28

- [x] Cleaned Binance BTCUSDT 30m OHLCV data
  - Converted open_time to UTC timestamp
  - Removed unused columns
  - Saved cleaned file to `data/processed/binance_BTCUSDT_30m_clean.csv`

## 📝 Auto Log Entry — 2025-08-09 10:27:41

- [ ] Downloaded OANDA EUR/USD M30 OHLCV data (2019–2021) and saved to data/raw/oanda_EURUSD_M30_2019-2021.csv

## 📝 Auto Log Entry — 2025-08-09 11:04:24

- [ ] Cleaned Binance BTCUSDT 30m OHLCV data and saved to data/processed/binance_BTCUSDT_30m_clean.csv

## 📝 Auto Log Entry — 2025-08-09 11:04:24

- [ ] Cleaned OANDA EUR/USD 30m OHLCV data and saved to data/processed/oanda_EURUSD_M30_clean.csv

