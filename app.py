import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ──
st.set_page_config(
    page_title="FraudShield — ML Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}

.sidebar-brand {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.4rem;
    letter-spacing: -0.5px;
    color: #ffffff;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 1.5rem;
}
.sidebar-brand span { color: #ff3c5f; }

.main .block-container {
    padding: 2.5rem 3rem;
    max-width: 1200px;
}

.page-header {
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #1e1e2e;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -1px;
    color: #ffffff;
    line-height: 1.1;
    margin: 0 0 0.4rem 0;
}
.page-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #555570;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #1e1e2e;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 2.5rem;
}
.metric-card {
    background: #0f0f1a;
    padding: 1.5rem 1.8rem;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #444460;
    margin-bottom: 0.6rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    color: #ffffff;
    letter-spacing: -1px;
    line-height: 1;
}
.metric-value.accent { color: #ff3c5f; }
.metric-value.dim { color: #888899; font-size: 1.5rem; }

.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #ff3c5f;
    margin: 2rem 0 1rem 0;
}

.info-block {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #ff3c5f;
    border-radius: 0 8px 8px 0;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.92rem;
    line-height: 1.7;
    color: #aaaacc;
}

.model-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    margin-top: 1rem;
}
.model-table th {
    text-align: left;
    padding: 0.7rem 1rem;
    background: #0f0f1a;
    color: #444460;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #1e1e2e;
}
.model-table td {
    padding: 0.85rem 1rem;
    border-bottom: 1px solid #111120;
    color: #aaaacc;
}
.model-table tr.winner td {
    color: #ffffff;
    background: #13131f;
}
.model-table tr.winner td:first-child {
    border-left: 2px solid #ff3c5f;
}
.badge {
    display: inline-block;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.5rem;
    border-radius: 3px;
    background: #ff3c5f22;
    color: #ff3c5f;
    border: 1px solid #ff3c5f44;
    margin-left: 0.5rem;
    vertical-align: middle;
}

.result-fraud {
    background: #1a0808;
    border: 1px solid #ff3c5f44;
    border-left: 4px solid #ff3c5f;
    border-radius: 8px;
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
}
.result-legit {
    background: #081a0e;
    border: 1px solid #00d68f44;
    border-left: 4px solid #00d68f;
    border-radius: 8px;
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
}
.result-label.fraud { color: #ff3c5f; }
.result-label.legit { color: #00d68f; }
.result-confidence {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #666680;
}
.conf-bar-wrap {
    margin: 1rem 0;
    background: #1a1a2e;
    border-radius: 3px;
    height: 4px;
    overflow: hidden;
}
.conf-bar {
    height: 100%;
    border-radius: 3px;
}
.conf-bar.fraud { background: #ff3c5f; }
.conf-bar.legit { background: #00d68f; }

.footer {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1e1e2e;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #333348;
    display: flex;
    justify-content: space-between;
}

div[data-testid="stSlider"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #666680 !important;
}
div[data-testid="stNumberInput"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #666680 !important;
}
.stButton button {
    background: #ff3c5f !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
}
.stButton button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ──
@st.cache_resource
def load_model():
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/model_config.json', 'r') as f:
        config = json.load(f)
    return model, config['threshold']

model, threshold = load_model()


# ── Sidebar ──
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Fraud<span>Shield</span></div>', unsafe_allow_html=True)

    page = st.radio("", ["Overview", "Predict", "Model Insights"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#333348; line-height:2;">
        MODEL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; XGBoost<br>
        THRESHOLD &nbsp; 0.95<br>
        AUC-ROC &nbsp;&nbsp;&nbsp; 0.9760<br>
        DATASET &nbsp;&nbsp;&nbsp;&nbsp; ULB / Kaggle
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>" * 8, unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#222238; line-height:1.8;">
        Built by Shashikiran BS<br>
        github.com/shashikiranbs2006
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════
if page == "Overview":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Credit Card<br>Fraud Detection</div>
        <div class="page-subtitle">ML-powered transaction risk scoring — XGBoost + SHAP explainability</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Total Transactions</div>
            <div class="metric-value">284,807</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Fraud Cases</div>
            <div class="metric-value accent">492</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Fraud Rate</div>
            <div class="metric-value dim">0.17%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Model AUC-ROC</div>
            <div class="metric-value">0.9760</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">The Problem</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block">
        Banks process millions of transactions every day. Fraud is rare — only 0.17% of all transactions —
        but every missed case results in direct financial loss. Traditional rule-based systems fail to catch
        complex fraud patterns that evolve over time. This system uses machine learning to learn those patterns
        from 284,807 historical labeled transactions and score new transactions in real time.
        <br><br>
        The core challenge is class imbalance: with 99.83% legitimate transactions, a naive model that always
        predicts legitimate achieves 99.83% accuracy while catching zero fraud. This system addresses that
        using SMOTE oversampling and threshold tuning, evaluated on Recall and AUC-ROC — not accuracy.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="model-table">
        <thead>
            <tr>
                <th>Model</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>AUC-ROC</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Logistic Regression</td>
                <td>0.06</td><td>0.92</td><td>0.11</td><td>0.9698</td>
            </tr>
            <tr>
                <td>Random Forest</td>
                <td>0.81</td><td>0.81</td><td>0.81</td><td>0.9688</td>
            </tr>
            <tr class="winner">
                <td>XGBoost <span class="badge">Selected</span></td>
                <td>0.81</td><td>0.82</td><td>0.81</td><td>0.9760</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">System Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block">
        <strong style="color:#ffffff;">Data Layer</strong> — 284,807 transactions from ULB Kaggle dataset.
        Amount and Time scaled using StandardScaler. V1–V28 are PCA-transformed anonymized features provided by the issuing bank.
        <br><br>
        <strong style="color:#ffffff;">Preprocessing</strong> — 80/20 train-test split with stratification.
        SMOTE applied exclusively on training data to generate synthetic fraud samples and balance class distribution.
        <br><br>
        <strong style="color:#ffffff;">Model Training</strong> — Three models trained and benchmarked: Logistic Regression (baseline),
        Random Forest (ensemble), XGBoost (gradient boosted trees). XGBoost selected on highest AUC-ROC.
        <br><br>
        <strong style="color:#ffffff;">Threshold Tuning</strong> — Default threshold 0.5 yielded Precision of 0.35.
        Tuning to 0.95 improved Precision to 0.81 with Recall held at 0.82.
        <br><br>
        <strong style="color:#ffffff;">Explainability</strong> — SHAP applied to every prediction.
        Each transaction gets a feature-level breakdown showing which inputs drove the fraud score.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <span>FraudShield — Credit Card Fraud Detection System</span>
        <span>Shashikiran BS &nbsp;|&nbsp; github.com/shashikiranbs2006</span>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════
# PAGE 2 — PREDICT
# ════════════════════════
elif page == "Predict":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Transaction<br>Risk Scoring</div>
        <div class="page-subtitle">Enter transaction parameters — model returns fraud probability with SHAP explanation</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;color:#444460;margin-bottom:0.8rem;">Transaction Info</p>', unsafe_allow_html=True)
        amount  = st.number_input("Amount (USD)", min_value=0.0, max_value=30000.0, value=149.62, step=1.0)
        time_val = st.number_input("Time (seconds elapsed)", min_value=0.0, max_value=200000.0, value=52120.0)
        st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;color:#444460;margin:1.2rem 0 0.8rem 0;">V1 — V10</p>', unsafe_allow_html=True)
        v1  = st.slider("V1",  -5.0, 5.0, -1.36, 0.01)
        v2  = st.slider("V2",  -5.0, 5.0,  0.00, 0.01)
        v3  = st.slider("V3",  -5.0, 5.0,  1.62, 0.01)
        v4  = st.slider("V4",  -5.0, 5.0,  0.00, 0.01)
        v5  = st.slider("V5",  -5.0, 5.0,  0.00, 0.01)
        v6  = st.slider("V6",  -5.0, 5.0,  0.00, 0.01)
        v7  = st.slider("V7",  -5.0, 5.0,  0.00, 0.01)
        v8  = st.slider("V8",  -5.0, 5.0,  0.00, 0.01)
        v9  = st.slider("V9",  -5.0, 5.0,  0.00, 0.01)
        v10 = st.slider("V10", -5.0, 5.0,  0.00, 0.01)

    with col2:
        st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;color:#444460;margin-bottom:0.8rem;">V11 — V20</p>', unsafe_allow_html=True)
        v11 = st.slider("V11", -5.0, 5.0, 0.00, 0.01)
        v12 = st.slider("V12", -5.0, 5.0, 0.00, 0.01)
        v13 = st.slider("V13", -5.0, 5.0, 0.00, 0.01)
        v14 = st.slider("V14", -5.0, 5.0, 0.00, 0.01)
        v15 = st.slider("V15", -5.0, 5.0, 0.00, 0.01)
        v16 = st.slider("V16", -5.0, 5.0, 0.00, 0.01)
        v17 = st.slider("V17", -5.0, 5.0, 0.00, 0.01)
        v18 = st.slider("V18", -5.0, 5.0, 0.00, 0.01)
        v19 = st.slider("V19", -5.0, 5.0, 0.00, 0.01)
        v20 = st.slider("V20", -5.0, 5.0, 0.00, 0.01)

    with col3:
        st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;color:#444460;margin-bottom:0.8rem;">V21 — V28</p>', unsafe_allow_html=True)
        v21 = st.slider("V21", -5.0, 5.0, 0.00, 0.01)
        v22 = st.slider("V22", -5.0, 5.0, 0.00, 0.01)
        v23 = st.slider("V23", -5.0, 5.0, 0.00, 0.01)
        v24 = st.slider("V24", -5.0, 5.0, 0.00, 0.01)
        v25 = st.slider("V25", -5.0, 5.0, 0.00, 0.01)
        v26 = st.slider("V26", -5.0, 5.0, 0.00, 0.01)
        v27 = st.slider("V27", -5.0, 5.0, 0.00, 0.01)
        v28 = st.slider("V28", -5.0, 5.0, 0.00, 0.01)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Run Prediction", use_container_width=True)

    if predict_btn:
        amount_scaled = (amount - 88.35) / 250.12
        time_scaled   = (time_val - 94813.86) / 47488.15

        features = np.array([[
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25, v26, v27, v28,
            amount_scaled, time_scaled
        ]])
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']
        features_df   = pd.DataFrame(features, columns=feature_names)

        proba      = model.predict_proba(features)[0][1]
        prediction = int(proba >= threshold)

        st.markdown("---")
        res_col1, res_col2 = st.columns([1.4, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-fraud">
                    <div class="result-label fraud">Fraud Detected</div>
                    <div class="result-confidence">Model confidence: {proba*100:.2f}% fraud probability</div>
                    <div class="conf-bar-wrap"><div class="conf-bar fraud" style="width:{proba*100:.1f}%"></div></div>
                    <div style="font-size:0.85rem;color:#884455;margin-top:0.8rem;line-height:1.6;">
                        This transaction exceeds the risk threshold of {threshold*100:.0f}%.
                        Flagged for immediate review. Do not process without manual verification.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legit">
                    <div class="result-label legit">Legitimate</div>
                    <div class="result-confidence">Fraud probability: {proba*100:.2f}% — below threshold</div>
                    <div class="conf-bar-wrap"><div class="conf-bar legit" style="width:{(1-proba)*100:.1f}%"></div></div>
                    <div style="font-size:0.85rem;color:#445544;margin-top:0.8rem;line-height:1.6;">
                        Transaction cleared. Risk score is below the {threshold*100:.0f}% threshold. No action required.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("""<div style="background:#0f0f1a;border:1px solid #1e1e2e;border-radius:8px;padding:1.2rem 1.5rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;color:#444460;margin-bottom:1rem;">Score Breakdown</div>
            """, unsafe_allow_html=True)
            st.metric("Fraud Probability", f"{proba*100:.2f}%")
            st.metric("Decision Threshold", f"{threshold*100:.0f}%")
            st.metric("Final Verdict", "FRAUD" if prediction == 1 else "CLEAR")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:2rem;">SHAP Explanation — Why This Score</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#444460;margin-bottom:1rem;">Each bar shows how much that feature pushed the score toward fraud (positive) or away from it (negative).</p>', unsafe_allow_html=True)

        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(features_df)
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(11, 5))
            fig.patch.set_facecolor('#0f0f1a')
            ax.set_facecolor('#0f0f1a')
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[0],
                    base_values=explainer.expected_value,
                    data=features_df.iloc[0],
                    feature_names=feature_names
                ),
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.markdown("""
    <div class="footer">
        <span>FraudShield — Credit Card Fraud Detection System</span>
        <span>Shashikiran BS &nbsp;|&nbsp; github.com/shashikiranbs2006</span>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ════════════════════════
elif page == "Model Insights":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Model<br>Insights</div>
        <div class="page-subtitle">Evaluation metrics, confusion matrices, ROC curves, and SHAP analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Confusion Matrices — All Models</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#444460;margin-bottom:1rem;">Rows = actual class. Columns = predicted class. True Positives (fraud caught) in bottom-right cell.</p>', unsafe_allow_html=True)
    st.image("plot_confusion_matrices.png", use_column_width=True)

    st.markdown('<div class="section-header">ROC Curves</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#444460;margin-bottom:1rem;">Area Under Curve (AUC) measures separation between fraud and legitimate. Closer to 1.0 is better. Random classifier = 0.5.</p>', unsafe_allow_html=True)
    st.image("plot_roc_curves.png", use_column_width=True)

    st.markdown('<div class="section-header">SHAP — Global Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#444460;margin-bottom:1rem;">Average impact of each feature across all test transactions. V14, V12, V10, V4 drive the fraud score most.</p>', unsafe_allow_html=True)
    st.image("plot_shap_global.png", use_column_width=True)

    st.markdown('<div class="section-header">SHAP — Feature Impact Direction</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#444460;margin-bottom:1rem;">Each dot = one transaction. Red = high feature value. Blue = low. X-axis = direction of impact on prediction.</p>', unsafe_allow_html=True)
    st.image("plot_shap_dot.png", use_column_width=True)

    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block">
        <strong style="color:#ffffff;">Feature Importance</strong> — V14, V12, V10, and V4 are the four most influential features.
        Very negative values of V14 strongly indicate fraud — consistent across the entire test set.<br><br>
        <strong style="color:#ffffff;">Threshold Tuning Impact</strong> — Moving from threshold 0.5 to 0.95
        improved Precision from 0.35 to 0.81 while Recall dropped only from 0.87 to 0.82.
        Significantly fewer false alarms in production.<br><br>
        <strong style="color:#ffffff;">SMOTE Contribution</strong> — Without SMOTE, the model trained on raw imbalanced data
        learned to almost always predict legitimate. After SMOTE, balanced 50-50 training gave the model
        sufficient fraud examples to learn meaningful patterns.<br><br>
        <strong style="color:#ffffff;">Model Selection</strong> — XGBoost outperforms both alternatives on AUC-ROC (0.9760 vs 0.9698 and 0.9688).
        Sequential boosting — correcting errors from previous trees — makes it particularly effective
        on imbalanced tabular classification.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <span>FraudShield — Credit Card Fraud Detection System</span>
        <span>Shashikiran BS &nbsp;|&nbsp; github.com/shashikiranbs2006</span>
    </div>
    """, unsafe_allow_html=True)