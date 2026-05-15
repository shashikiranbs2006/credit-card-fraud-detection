# Credit Card Fraud Detection System

ML-powered transaction risk scoring using XGBoost, SMOTE, and SHAP explainability.

## Problem
Banks process millions of transactions daily. Only 0.17% are fraud — but missing one costs real money. 
Traditional rule-based systems miss complex patterns. This system learns those patterns from 284,807 
historical labeled transactions.

## Key Challenge
Class imbalance — 99.83% legitimate vs 0.17% fraud. A naive model scores 99.83% accuracy 
while catching zero fraud. Solved using SMOTE oversampling and threshold tuning.

## Model Performance

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.06 | 0.92 | 0.11 | 0.9698 |
| Random Forest | 0.81 | 0.81 | 0.81 | 0.9688 |
| **XGBoost (selected)** | **0.81** | **0.82** | **0.81** | **0.9760** |

Threshold tuned from 0.5 → 0.95, improving Precision from 0.35 to 0.81 with no Recall loss.

## Stack
- Python, XGBoost, scikit-learn, imbalanced-learn
- SHAP for explainability
- Streamlit for UI
- Dataset: [ULB Credit Card Fraud — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features
- 3 model comparison with proper metrics (not accuracy)
- SMOTE for class imbalance handling
- Threshold tuning for Precision-Recall tradeoff
- SHAP waterfall explanation per prediction
- Clean Streamlit UI with dark theme