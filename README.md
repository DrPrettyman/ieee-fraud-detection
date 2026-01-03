# Optimizing Fraud Detection for E-Commerce

A cost-sensitive approach to transaction screening using the IEEE-CIS Fraud Detection dataset.

## Problem Statement

E-commerce companies lose billions annually to fraud, but aggressive fraud prevention creates false positives that block legitimate customers. This project builds a fraud detection system that optimizes the trade-off between fraud losses and customer friction.

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
python data/download_data.py
```

### Kaggle API Setup

Before downloading, configure your Kaggle credentials:

1. Go to https://www.kaggle.com/settings
2. Under "API Tokens", click "Create New Token"
3. Copy the `KGAT_...` token and set the environment variable:
   ```bash
   export KAGGLE_API_TOKEN="KGAT_your_token_here"
   ```
4. Accept the [competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules)

## Project Structure

```
ieee-fraud-detection/
├── data/                  # Dataset and download script
│   └── download_data.py   # Kaggle download utility
├── notebooks/             # Analysis notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_business_analysis.ipynb
├── src/                   # Reusable modules
│   ├── features.py        # Feature engineering
│   ├── model.py           # Model training/evaluation
│   └── cost_analysis.py   # Business impact calculations
└── reports/               # Executive summaries
```

## Key Differentiators

- **Cost-sensitive approach**: Translates model performance into dollar impact
- **Threshold recommendations**: Actionable decision tool for different business scenarios
- **Production thinking**: Considers deployment constraints and monitoring
- **Business narrative**: Quantified ROI, not just accuracy metrics

## Dataset

| File | Description |
|------|-------------|
| `train_transaction.csv` | 590K transactions, 394 features |
| `train_identity.csv` | Device/identity info for subset |
| `test_transaction.csv` | 500K transactions for prediction |
| `test_identity.csv` | Identity info for test set |

**Fraud rate**: 3.5% (imbalanced)

## Technologies

- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **ML**: scikit-learn, XGBoost, LightGBM
- **Imbalanced learning**: imbalanced-learn (SMOTE)
