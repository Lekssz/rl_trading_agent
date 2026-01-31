Project Structure Overview

This document provides an overview of the folder structure of the **Multi-Modal Reinforcement Learning Trading Agent** Below is a description of each folder and its contents:
    ├── README.md                                    # Overview of the project, usage instructions
    ├── Project_Structure.md                         # Description of the folder structure
    ├── code/                                        # Code files for analysis, training, model, etc.
    │   ├── analysis/.                               # Scripts for analyzing and visualizing results
    │   ├── clean/
    │   ├── env/                                     # Scripts for configuring the RL environment
    │   ├── merge/                                   # Scripts for merging datasets
    │   ├── model/                                    
    │   ├── process/                                 # Data processing scripts
    │   ├── state/                                   # Scripts for managing RL states
    │   └── train/                                   # Scripts for model training
    ├── data/                                        # Data files (raw, processed, and states)
    │   ├── model/                                   # CNN price embeddings for btcusd and eurusd
    │   ├── processed/
    │   ├── raw/                                     # Raw data files
    │   └── rl_states/                               # Data for reinforcement learning states
    ├── docs/                                        # Documentation files
    │   ├── DATA_LICENSE.md                          # License information
    │   ├── ETHICS.md                                # Ethics documentation
    │   ├── LICENSES.md                              # License for code and resources
    │   ├── Project_Summary.md                       # Project summary document
    │   └── project_setup_checklist.md               # Checklist for setting up the project
    ├── environment.yml                              # Conda environment file for replicating the environment
    ├── ingestion/                                   # Scripts for data ingestion (Binance, OANDA)
    │   ├── binance_ohlcv.py                         # Binance OHLCV data fetching script
    │   └── oanda_ohlcv.py                           # OANDA OHLCV data fetching script
    ├── models/                                      # Trained PPO models                     
    │   ├── ppo_btc_baseline_ablations_full/         # The btc ppo model for the full model 
    │   ├── ppo_btc_optimised/.                      # The btc ppo hypertuned model
    │   └── (other models)
    ├── results/                                     # Results of my experiments
    │   ├── ppo_btc_baseline_ablations_full/         # Contains the results for the trained btc ppo full model 
    │   ├── ppo_btc_baseline_ablations_price/        # Contains the results for the trained btc ppo price only btc model
    │   └── (other results)
    ├── visualisation/                                # Visualization and figures
    │   ├── btc_baseline_ablation_figures/
    │   ├── cross_asset_baseline_vs_buyhold/
    │   └── (other visualizations)


**Explanation of Key Folders:**

code/: Contains all Python scripts organized by functionality, including data analysis, model training, and environment setup.

data/: Contains the raw, processed, and RL state data used throughout the project.

docs/: Contains all documentation files, including licensing, ethical considerations, and project summary.

models/: Contains pre-trained models, categorized by experiment type (e.g., baseline ppo ablations for btc & eur, optimized).

results/: Stores the results of the experiments, such as performance metrics and ablation study outcomes.

visualisation/: Contains all figures and charts generated during the project for analysis and presentation.