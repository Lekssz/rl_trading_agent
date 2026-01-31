# LICENSES.md — Third-Party Software and Licenses

**Student:** Gbemileke Micah (K2457391)  
**Programme:** MSc Data Science, Kingston University London  

This document lists the main third-party software libraries used in this project and their open-source licenses.
All software is used for **non-commercial academic research** purposes only.

> Note: The full dependency set is defined in `environment.yml`. This file summarises the primary libraries
> used for data processing, modelling, reinforcement learning, and evaluation.

---

## Core Libraries

| Component / Library | Typical Use in Project | License |
|---|---|---|
| Python | Runtime | PSF License |
| NumPy | Arrays / RL state tensors | BSD 3-Clause |
| pandas | Data processing, joining, resampling | BSD 3-Clause |
| SciPy | Numeric utilities | BSD 3-Clause |
| scikit-learn | Scaling / preprocessing utilities | BSD 3-Clause |
| PyTorch | CNN training and embedding generation | BSD 3-Clause |
| Stable-Baselines3 | PPO training and evaluation | MIT |
| Gym / Gymnasium | RL environment interface | MIT |
| Matplotlib | Charts and figures | Matplotlib License (PSF-compatible) |

---

## License Use Notes

- All listed libraries are open-source and permit academic use and modification under their respective terms.
- No proprietary software is required to reproduce the experimental artefact.
- This project does not redistribute third-party source code; it depends on standard package installation via Conda/Pip as declared in `environment.yml`.

---

_Last updated: [JANUARY/ 2026]_
