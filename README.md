# MSc Data Science Dissertation Project 
# AI Trading Agent

**Student:** Gbemileke Micah  
**Student ID:** K2457391  
**Course:** MSc Data Science  
**University:** Kingston University London  
**Project Period:** July – October 2025  

##📘 Project Title  
**A Fractal-Risk-Aware PPO Multimodal Framework for Cross-Asset Forex and Cryptocurrency Trading**

## 🧠 Project Overview

This project aims to design and implement a reinforcement learning (RL) agent capable of trading across both the **EUR/USD** (Forex) and **BTC/USD** (Cryptocurrency) markets. The system will use:


Key innovations include:

## 🧩 Key System Features

- 🤖 **Proximal Policy Optimization (PPO)** – A robust reinforcement learning algorithm chosen for its stability in continuous action spaces  
- 🧠 **CNN-Validated Pattern Detection** – Detects head-and-shoulders and triangle patterns using finetuned ResNet-18 on candlestick images  
- 📈 **Technical Indicators** – Integrates order blocks and 200-day moving average confluence filters to align trades with institutional flow  
- 💬 **Real-Time Sentiment Analysis** – Combines fast VADER scores with FinBERT’s financial-contextual embeddings for high-fidelity market mood  
- ⚠️ **Fractal Volatility Controls** – Implements Mandelbrot-inspired volatility caps (20-day rolling) to mitigate drawdowns during crisis regimes  
- 🔄 **Dual-Market Adaptation** – Unified PPO agent architecture capable of switching seamlessly between EUR/USD (Forex) and BTC/USD (Crypto)  
- 🎯 **Bayesian Optimization + Monte Carlo Stress Testing** – Enhances robustness through hyperparameter tuning and 150-path scenario simulations

> **This is a simulation-based research project.**  
> **No real-money trading is executed.**
>**The agent will be trained and evaluated using historical data (2019–2021) and benchmarked against vanilla PPO and buy-and-hold baselines.**

---
---

## 📁 Project Structure

├── code/ # Core RL training and evaluation logic  
├── code/src/processing/ # Scripts for cleaning and transforming raw OHLCV data into model-ready datasets  
├── configs/ # PPO agent parameters, tuning configs  
├── feature_engineering/ # CNN pattern detection, sentiment scoring  
├── ingestion/ # API scripts (OANDA, Binance, Twitter, NewsAPI)  
├── risk_control/ # Fractal volatility cap, triple-barrier labeling  
├── data/ # Raw and processed data (OneDrive only)  
├── models/ # Saved models and checkpoints  
├── notebooks/ # Exploratory notebooks and results  
├── dashboard/ # Streamlit visualization tool  
├── tests/ # Unit tests (optional)  
├── docs/ # Proposal, Gantt, ethics, diagrams  
├── ETHICS.md # Ethics protocols and audit trails  
├── DATA_LICENSE.md # API and data usage compliance  
├── LICENSES.md # 3rd-party software licenses (WIP)  
├── README.md # This file  



---

## 🧪 Status

🛠️ **Development has not yet started.**  
This repository is being set up as a secure and trackable environment for code development, literature review, and ethics compliance.

---

## 🔐 Ethics & Licensing

- **No live trading or real-money execution** will be performed
- Tweets will be anonymized (no handles, user IDs, or geolocation)
- Data is stored securely using **Kingston University OneDrive**
- Scripts will comply with:
  - Twitter API v2 Academic
  - OANDA & Binance Terms of Service
- Full documentation is in:
  - `ETHICS.md`
  - `DATA_LICENSE.md`
- Proposed code license: **MIT** (pending supervisor confirmation)
- 📄 Ethics Protocol: [ETHICS.md](./ETHICS.md)
- 📄 Data Licensing: [DATA_LICENSE.md](./DATA_LICENSE.md)

---

## ⚙️ Planned Tech Stack

| Category        | Tools                                    |
|----------------|-------------------------------------------|
| Language        | Python 3.9+                              |
| RL Framework    | Stable-Baselines3 (PPO)                  |
| Deep Learning   | PyTorch + torchvision                    |
| Optimization    | Optuna (Bayesian search + pruning)       |
| Sentiment       | VADER + FinBERT (transformers)           |
| Visualization   | Jupyter + Streamlit                      |
| APIs            | OANDA (Forex), Binance (Crypto), Twitter |
| Risk Modeling   | Fractal volatility cap, triple-barrier   |

---

## 📦 Planned Environment (via Conda)

`environment.yml` will include:

```yaml
name: ai-trading-agent
dependencies:
  - python=3.9
  - pip
  - pip:
      - torch
      - torchvision
      - stable-baselines3
      - transformers
      - vaderSentiment
      - optuna
      - streamlit
      - scikit-learn
      - pandas
      - numpy
      - scipy
      - matplotlib
      - jupyter
```      



text

---

▶️ Planned Execution
All training scripts will be organized in code/src/

Dashboard interface will run via Streamlit:

bash
Copy code
streamlit run dashboard/app.py
Model training will be triggered via:

bash
Copy code
python code/train_agent.py

2025-08-06:
- Added `requests` to environment.yml for API calls (Binance & OANDA ingestion scripts).


✍️ Author
Name: Gbemileke Micah
Programme: MSc Data Science
Institution: Kingston University London
Start Date: July 2025
Expected Submission: October 2025

I used an AI assistant to draft scaffolding for ingestion/cleaning; all code was reviewed, modified, and validated by me.



**📌Notes
This project is conducted under academic supervision.
No personal, sensitive, or commercial data will be shared or published.
All use of APIs and external models complies with applicable licenses and ethical standards.**