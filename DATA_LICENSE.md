
# DATA_LICENSE.md

## Data Sources and Licensing

This document outlines the data sources used in the AI Trading Agent project and confirms compliance with licensing and usage terms.

---

## OANDA (Forex API)

- **Official site**: https://www.oanda.com
- **Data Used**: EUR/USD OHLCV historical Forex market data
- **Access**: OANDA v20 REST API using a demo account registered with Kingston University email
- **Terms of Use**: https://www.oanda.com/us-en/legal/
- **Compliance**: API access is limited to public market data only (OHLCV and technical indicators). No live trading, order placement, or financial transactions are performed. All credentials (Personal Access Token, Account ID) are stored securely in `.env` files and excluded from version control via `.gitignore`. Usage complies with OANDA’s Terms of Service and Kingston University ethics guidelines.

---

## Binance (Crypto API)

- **Official site**: https://www.binance.com
- **Data Used**: BTC/USDT OHLCV historical market data
- **Access**: Public API (no authentication keys used)
- **Terms of Use**: https://www.binance.com/en/terms
- **Compliance**: No API keys, authentication, or private endpoints are used. Only public OHLCV market data is accessed. All data use complies with Binance’s public API usage policy.

---

## Twitter API (Academic Research Track)

- **Data Used**: Public tweets mentioning market-related terms
- **Access**: Academic Research v2 API access
- **Terms of Use**: https://developer.twitter.com/en/developer-terms/agreement-and-policy
- **Compliance**:
  - No personal identifiers (PII) are collected or stored
  - Only tweet text, timestamps, and sentiment scores are used
  - All sentiment inputs are anonymized in preprocessing

---

## NewsAPI (Sentiment Fallback)

- **Official site**: https://newsapi.org/
- **Data Used**: Financial headlines for EUR/USD and BTC/USD
- **Access**: Free tier (100 requests/day)
- **Terms of Use**: https://newsapi.org/terms
- **Compliance**: Used only when Twitter API access fails. Headlines processed similarly to tweets using the same sentiment pipeline.

---

## Storage and Handling

All project data is stored securely using:

- 🔒 **FileVault-encrypted Mac system** (personal device, enabled for full-disk encryption)
- ☁️ **Kingston University OneDrive account**, used exclusively for cloud storage and syncing

No personal or research data is stored on unencrypted or non-Kingston-approved cloud services. 
No data is uploaded to public repositories.


# DATA_LICENSE.md

> Created by Gbemileke Micah (K2457391) for CI7000 MSc Data Science Project  
> MSc Data Science, Kingston University  
> Last updated: August 1, 2025  

---

## 🔍 Data Sources and Usage Summary

This project uses data exclusively from publicly accessible APIs under research-only, non-commercial licenses. All usage complies with the individual platform terms and is documented here for transparency and ethical compliance.

- No personal or private user data is collected
- Twitter metadata (e.g., handles, IDs, locations) is anonymized
- Data is stored on a FileVault-encrypted personal Mac and synchronized only via Kingston University OneDrive
- No raw data is uploaded to GitHub or shared externally

---

## 🔗 API Terms of Service

- [OANDA Terms of Use](https://www.oanda.com/us-en/legal/)
- [Binance Terms of Use](https://www.binance.com/en/terms)
- [Twitter API Academic Access Policy](https://developer.twitter.com/en/solutions/academic-research)
- [NewsAPI Terms of Service](https://newsapi.org/terms)

All API integrations used comply with respective rate limits, terms of access, and attribution rules.

---

## 📅 Data Usage Review Log

- **Last reviewed**: August 6, 2025  
- **Review frequency**: Every 30 days or upon API terms change  
- **Responsible party**: Gbemileke Micah (MSc Data Science candidate)

---

## 🎓 Data Usage Disclaimer

All data is used exclusively for simulation, analysis, and academic reporting as part of a Kingston University MSc dissertation. This project does not involve commercial trading, redistribution of data, or storage of private user information. All work is conducted in accordance with Kingston University ethics policy and GDPR requirements.
