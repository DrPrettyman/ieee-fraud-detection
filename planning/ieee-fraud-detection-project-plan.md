# IEEE-CIS Fraud Detection Portfolio Project Plan

## Project Title
**"Optimizing Fraud Detection for E-Commerce: A Cost-Sensitive Approach to Transaction Screening"**

---

## Business Problem Statement
E-commerce companies lose billions annually to fraud, but aggressive fraud prevention creates false positives that block legitimate customers—damaging revenue and trust. This project builds a fraud detection system that **optimizes the trade-off between fraud losses and customer friction**, providing actionable thresholds for different business scenarios.

---

## What Makes This Project Stand Out

Most IEEE-CIS projects stop at "I got 0.95 AUC." Your project will:

1. **Quantify business impact** in dollars, not just metrics
2. **Build a decision tool** for fraud analysts (threshold recommendations)
3. **Account for real-world constraints** (review capacity, customer experience)
4. **Provide actionable recommendations** a company could implement

---

## Dataset Overview

| File | Description |
|------|-------------|
| `train_transaction.csv` | 590K transactions with 394 features |
| `train_identity.csv` | Identity info for subset of transactions |
| `test_transaction.csv` | 500K transactions for prediction |
| `test_identity.csv` | Identity info for test set |

**Key Feature Groups:**
- `TransactionDT`: Seconds from reference time
- `TransactionAMT`: Payment amount (USD)
- `card1-card6`: Card type, issuer, category
- `addr1-addr2`: Billing region/country
- `C1-C14`: Count features (addresses per card, etc.)
- `D1-D15`: Time deltas (days since last transaction)
- `M1-M9`: Match features (name/address matching)
- `V1-V339`: Vesta's proprietary engineered features
- `id_01-id_38`: Device/behavioral fingerprints

**Class Balance:** 3.5% fraud (imbalanced)

---

## Project Phases

### Phase 1: Data Acquisition & Exploration
**Goal:** Understand the data and identify key patterns

- [ ] Download dataset from Kaggle
- [ ] Profile data: missing values, distributions, cardinality
- [ ] Analyze fraud patterns by:
  - Transaction amount
  - Card type (credit vs debit)
  - Time of day/week
  - Product category
- [ ] Identify feature correlations (especially V1-V339)
- [ ] Document data quality issues

**Deliverable:** EDA notebook with 5-7 key insights

---

### Phase 2: Feature Engineering
**Goal:** Create meaningful features that capture fraud signals

- [ ] Time-based features:
  - Hour of day, day of week from TransactionDT
  - Transaction velocity (count in last hour/day)
- [ ] Aggregation features:
  - Average transaction amount per card
  - Fraud rate by card/device/address
- [ ] Interaction features:
  - Amount deviation from card's typical behavior
- [ ] Handle V1-V339:
  - Correlation analysis
  - PCA or feature selection
- [ ] Missing value strategy (missingness as signal)

**Deliverable:** Feature engineering pipeline (reproducible code)

---

### Phase 3: Model Development
**Goal:** Build and compare fraud detection models

- [ ] Baseline model (Logistic Regression)
- [ ] Tree-based models:
  - Random Forest
  - XGBoost / LightGBM
- [ ] Handle class imbalance:
  - SMOTE
  - Class weights
  - Threshold tuning
- [ ] Cross-validation strategy (time-based split)
- [ ] Hyperparameter tuning

**Deliverable:** Model comparison with precision-recall curves

---

### Phase 4: Business Impact Analysis ⭐ (KEY DIFFERENTIATOR)
**Goal:** Translate model performance into business value

#### Cost Matrix Definition
| Outcome | Description | Estimated Cost |
|---------|-------------|----------------|
| True Positive | Fraud caught | $0 (loss prevented) |
| False Negative | Fraud missed | Average fraud amount (~$150) |
| False Positive | Legitimate blocked | Lost sale + customer lifetime value (~$50) |
| True Negative | Good transaction | $0 |

#### Threshold Analysis
- [ ] Calculate total cost at different thresholds
- [ ] Find optimal threshold for different scenarios:
  - Minimize total cost
  - Fixed review capacity (e.g., 1000 reviews/day)
  - Maximum acceptable false positive rate
- [ ] Create threshold recommendation tool

#### ROI Calculation
```
Example calculation:
- Monthly transactions: 500,000
- Fraud rate: 3.5% = 17,500 fraud attempts
- Without model: $2.6M loss (17,500 × $150)
- With model at optimal threshold:
  - Catch 85% of fraud: $2.2M saved
  - False positives (2%): $500K in blocked legitimate sales
  - Net benefit: $1.7M/month
```

**Deliverable:** Interactive cost-benefit analysis

---

### Phase 5: Operationalization Considerations
**Goal:** Show you understand production requirements

- [ ] Feature availability at prediction time
- [ ] Model latency requirements
- [ ] Monitoring for concept drift
- [ ] A/B testing framework proposal
- [ ] Alert queue design for fraud analysts

**Deliverable:** Production deployment recommendations document

---

### Phase 6: Documentation & Presentation
**Goal:** Professional portfolio-ready output

#### GitHub Repository Structure
```
ieee-fraud-detection/
├── README.md                 # Project overview (under 500 words)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_business_analysis.ipynb
├── src/
│   ├── features.py           # Feature engineering functions
│   ├── model.py              # Model training code
│   └── cost_analysis.py      # Business impact calculations
├── reports/
│   └── executive_summary.md  # 1-page business summary
├── requirements.txt
└── .gitignore
```

#### README Sections (per the article)
1. **Problem Statement:** One clear sentence
2. **Business Impact:** Quantified savings/ROI
3. **Methodology:** Brief approach overview
4. **Key Findings:** Bullet-point insights
5. **Recommendations:** Specific thresholds and actions

---

## Technologies to Showcase

| Skill | Tools |
|-------|-------|
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| ML Modeling | scikit-learn, XGBoost/LightGBM |
| Imbalanced Learning | imbalanced-learn (SMOTE) |
| Feature Engineering | Custom pipelines |
| Version Control | Git, GitHub |

---

## Key Differentiators for Your Portfolio

1. **Cost-Sensitive Approach:** Most projects ignore business costs
2. **Threshold Recommendations:** Actionable decision tool
3. **Production Thinking:** Shows you understand deployment
4. **Clear Business Narrative:** Not just "I built a model"
5. **Professional Documentation:** Easy for hiring managers to scan

---

## Potential Extensions (Optional)

- Build a Streamlit dashboard for threshold exploration
- Add explainability (SHAP values for individual predictions)
- Compare with anomaly detection approach (Isolation Forest)
- Time-series analysis of fraud patterns

---

## Kaggle Competition Benchmarks

The IEEE-CIS Fraud Detection competition ran July-October 2019 with 6,381 teams.

### Leaderboard Scores

| Ranking | AUC Score | Notes |
|---------|-----------|-------|
| 1st Place | 0.946 | XGBoost + LightGBM + CatBoost ensemble |
| Top 5% (~300) | ~0.924 | Heavy feature engineering |
| Top 30-40% | ~0.900 | Solid baseline with tuned GBDT |
| Baseline LightGBM | ~0.890 | Default parameters |
| Logistic Regression | ~0.840 | Simple baseline |

### What Top Solutions Did

1. **Massive feature engineering** - 1000+ features from aggregations
2. **Model ensembling** - XGBoost + LightGBM + CatBoost blended
3. **Target encoding** - For high-cardinality categoricals (card1, addr1)
4. **UID creation** - Synthesized user IDs from card/addr combinations
5. **Time-based features** - Transaction velocity, recency patterns

### Techniques to Improve Score

| Technique | Expected Gain |
|-----------|---------------|
| Aggregation features (mean/std per card1) | +0.01-0.02 |
| Target encoding for card1, addr1 | +0.005-0.01 |
| Hyperparameter tuning (Optuna) | +0.005 |
| Ensemble with XGBoost | +0.01 |
| Magic features (UID reconstruction) | +0.01-0.02 |

---

## Data Sources

- **Dataset:** [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/competitions/ieee-fraud-detection)
- **Leaderboard:** [Final Standings](https://www.kaggle.com/c/ieee-fraud-detection/leaderboard)
- **Reference Solutions:**
  - [1st Place Solution (NVIDIA Blog)](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/)
  - [Top 5% Solution Writeup](https://towardsdatascience.com/ieee-cis-fraud-detection-top-5-solution-5488fc66e95f/)
  - [NYC Data Science Blog](https://nycdatascience.com/blog/student-works/ieee-cis-fraud-detection-detecting-fraud-from-customer-transactions/)

---

## Timeline Estimate

| Phase | Tasks |
|-------|-------|
| Phase 1 | EDA and data understanding |
| Phase 2 | Feature engineering |
| Phase 3 | Model development |
| Phase 4 | Business impact analysis |
| Phase 5 | Production considerations |
| Phase 6 | Documentation and polish |

---

## Next Steps

1. Create Kaggle account (if needed) and download dataset
2. Set up project repository structure
3. Begin Phase 1: EDA
