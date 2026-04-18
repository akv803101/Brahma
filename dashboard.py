import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, json, time
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brahma — Churn Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colour palette ─────────────────────────────────────────────────────────────
PRIMARY   = "#2563EB"
HIGHLIGHT = "#DC2626"
MUTED     = "#D1D5DB"
GREEN     = "#16A34A"

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #2563EB;
    }
    .metric-val  { font-size: 2.2rem; font-weight: 700; color: #2563EB; }
    .metric-label{ font-size: 0.85rem; color: #6B7280; margin-top: 2px; }
    .risk-HIGH   { background:#FEE2E2; color:#DC2626; padding:4px 12px;
                   border-radius:20px; font-weight:700; font-size:0.9rem; }
    .risk-MEDIUM { background:#FEF3C7; color:#D97706; padding:4px 12px;
                   border-radius:20px; font-weight:700; font-size:0.9rem; }
    .risk-LOW    { background:#D1FAE5; color:#16A34A; padding:4px 12px;
                   border-radius:20px; font-weight:700; font-size:0.9rem; }
    .banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563EB 100%);
        color: white; border-radius: 14px; padding: 28px 36px; margin-bottom: 24px;
    }
    .banner h1 { color: white; margin: 0; font-size: 2rem; }
    .banner p  { color: #BFDBFE; margin: 6px 0 0 0; font-size: 1rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #1e3a8a;
        border-bottom: 2px solid #2563EB; padding-bottom: 6px;
        margin: 20px 0 14px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load assets ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    with open("outputs/data/splits.pkl", "rb") as f:
        splits = pickle.load(f)
    with open("outputs/models/xgb_tuned.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("outputs/models/lr_baseline.pkl", "rb") as f:
        lr_model = pickle.load(f)
    with open("outputs/data/drift_config.json") as f:
        drift_cfg = json.load(f)
    lb = pd.read_csv("outputs/data/leaderboard.csv")
    df_features = pd.read_parquet("outputs/data/features_engineered.parquet")
    return splits, xgb_model, lr_model, drift_cfg, lb, df_features

splits, xgb_model, lr_model, drift_cfg, leaderboard_df, df_features = load_assets()

X_test  = splits["X_test"]
y_test  = splits["y_test"]
X_train = splits["X_train"]
y_train = splits["y_train"]
feature_cols = splits["feature_cols"]

# ── predict_brahma ─────────────────────────────────────────────────────────────
MODEL_VERSION = "brahma-churn-v1.0.0"

def predict_brahma(input_dict):
    t0  = time.perf_counter()
    row = np.array([[input_dict.get(f, 0.0) for f in feature_cols]], dtype=float)
    prob       = float(xgb_model.predict_proba(row)[0, 1])
    prediction = int(prob >= 0.5)
    risk_tier  = "HIGH" if prob > 0.70 else ("MEDIUM" if prob > 0.40 else "LOW")
    try:
        imps  = xgb_model.feature_importances_
        vals  = row[0]
        contr = np.abs(imps * vals)
        top3  = contr.argsort()[-3:][::-1]
        top_reasons = [
            {"feature": feature_cols[i], "importance": round(float(imps[i]), 4),
             "value": round(float(vals[i]), 4)}
            for i in top3
        ]
    except Exception:
        top_reasons = []
    latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    return {"prediction": prediction, "probability": round(prob, 4),
            "risk_tier": risk_tier, "top_reasons": top_reasons,
            "model_version": MODEL_VERSION,
            "timestamp": pd.Timestamp.now().isoformat(),
            "latency_ms": latency_ms}

# ── Helper: load chart image ───────────────────────────────────────────────────
def show_chart(path, caption="", use_col_width=True):
    if os.path.exists(path):
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=use_col_width)
    else:
        st.info(f"Chart not found: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="banner">
  <h1>🧠 Brahma — Credit Card Churn Intelligence</h1>
  <p>Model: brahma-churn-v1.0.0 &nbsp;|&nbsp; Pipeline: 13/13 stages complete
     &nbsp;|&nbsp; Last run: April 2026</p>
</div>
""", unsafe_allow_html=True)

# ── Top KPI row ────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    ("0.9931",  "Test ROC-AUC"),
    ("95.4%",   "Recall (Churners Caught)"),
    ("77.6%",   "Precision"),
    ("0.875",   "F1 Score"),
    ("1.6 ms",  "P99 Latency"),
]
for col, (val, label) in zip([k1, k2, k3, k4, k5], kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{val}</div>
          <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 EDA & Data",
    "🏋️ Model Training",
    "📈 Evaluation",
    "✅ Validation",
    "🔀 Ensembling",
    "🚀 Live Predictor",
    "📋 Pipeline Summary"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA & Data
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",   "5,000")
    c2.metric("Features",          "23 (after engineering)")
    c3.metric("Churn Rate",        "14.6%")
    c4.metric("Missing Values",    "0% (after preprocessing)")

    st.markdown('<div class="section-header">Target Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        show_chart("outputs/charts/eda/eda_target_distribution.png")
    with col2:
        show_chart("outputs/charts/eda/eda_feature_correlations.png")

    st.markdown('<div class="section-header">Transaction Behaviour by Churn Status</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        show_chart("outputs/charts/eda/eda_dist_total_trans_ct.png")
    with col2:
        show_chart("outputs/charts/eda/eda_dist_total_trans_amt.png")

    col1, col2 = st.columns(2)
    with col1:
        show_chart("outputs/charts/eda/eda_violin_months_inactive.png")
    with col2:
        show_chart("outputs/charts/eda/eda_correlation_heatmap.png")

    st.markdown('<div class="section-header">Feature Importance (Random Forest)</div>',
                unsafe_allow_html=True)
    show_chart("outputs/charts/eda/feature_importance_top20.png")

    st.markdown('<div class="section-header">Key EDA Findings</div>', unsafe_allow_html=True)
    findings = {
        "Top predictor": "total_trans_ct — r=0.492 with churn (negative)",
        "Inactivity gap": "Churned customers inactive 3× longer than retained",
        "Multicollinearity": "credit_limit ↔ avg_open_to_buy: r=0.96 → dropped avg_open_to_buy",
        "Engineered features": "5 new features created — re_engagement_signal ranks #3 in importance",
        "Class imbalance": "14.6% minority → scale_pos_weight = 5.85 applied to XGBoost",
    }
    for k, v in findings.items():
        st.markdown(f"**{k}:** {v}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Training
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">Model Leaderboard</div>', unsafe_allow_html=True)

    lb_display = leaderboard_df[["model", "auc_train", "auc_val", "f1_val",
                                  "recall_val", "gap"]].copy()
    lb_display.columns = ["Model", "AUC Train", "AUC Val", "F1 Val", "Recall Val", "Gap"]
    lb_display = lb_display.sort_values("AUC Val", ascending=False).reset_index(drop=True)

    def style_lb(row):
        if row["Model"] == "XGBoost_tuned":
            return ["background-color: #EFF6FF; font-weight: bold"] * len(row)
        elif row["Model"] == "DummyClassifier":
            return ["color: #9CA3AF"] * len(row)
        return [""] * len(row)

    st.dataframe(
        lb_display.style.apply(style_lb, axis=1).format({
            "AUC Train": "{:.4f}", "AUC Val": "{:.4f}",
            "F1 Val": "{:.4f}", "Recall Val": "{:.4f}", "Gap": "{:.4f}"
        }),
        use_container_width=True, hide_index=True
    )

    st.markdown('<div class="section-header">Training Charts</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        show_chart("outputs/charts/training/learning_curve_xgb_tuned.png")
    with col2:
        show_chart("outputs/charts/training/optuna_history.png")

    st.markdown('<div class="section-header">Optuna Best Hyperparameters</div>',
                unsafe_allow_html=True)
    params = xgb_model.get_params()
    param_df = pd.DataFrame([
        {"Parameter": k, "Value": str(round(v, 4) if isinstance(v, float) else v)}
        for k, v in params.items()
        if k in ["n_estimators", "max_depth", "learning_rate", "subsample",
                  "colsample_bytree", "min_child_weight", "gamma",
                  "reg_alpha", "reg_lambda", "scale_pos_weight"]
    ])
    st.dataframe(param_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                                  recall_score, accuracy_score,
                                  matthews_corrcoef, cohen_kappa_score,
                                  average_precision_score)

    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    st.markdown('<div class="section-header">Test Set Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC",    f"{roc_auc_score(y_test, y_prob):.4f}")
    m2.metric("F1 Score",   f"{f1_score(y_test, y_pred):.4f}")
    m3.metric("Recall",     f"{recall_score(y_test, y_pred):.4f}")
    m4.metric("Precision",  f"{precision_score(y_test, y_pred):.4f}")
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Accuracy",   f"{accuracy_score(y_test, y_pred):.4f}")
    m6.metric("MCC",        f"{matthews_corrcoef(y_test, y_pred):.4f}")
    m7.metric("Cohen Kappa",f"{cohen_kappa_score(y_test, y_pred):.4f}")
    m8.metric("Avg Precision", f"{average_precision_score(y_test, y_prob):.4f}")

    st.markdown('<div class="section-header">Evaluation Charts</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        show_chart("outputs/charts/evaluation/roc_curve.png")
    with col2:
        show_chart("outputs/charts/evaluation/precision_recall_curve.png")

    col1, col2 = st.columns(2)
    with col1:
        show_chart("outputs/charts/evaluation/confusion_matrix.png")
    with col2:
        show_chart("outputs/charts/evaluation/score_distribution.png")

    col1, col2 = st.columns(2)
    with col1:
        show_chart("outputs/charts/evaluation/calibration_curve.png")
    with col2:
        show_chart("outputs/charts/evaluation/shap_beeswarm.png")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Validation
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">Cross-Validation & Threshold Analysis</div>',
                unsafe_allow_html=True)
    show_chart("outputs/charts/validation/cv_and_threshold.png")

    st.markdown('<div class="section-header">Validation Summary</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**10-Fold Cross-Validation**")
        cv_data = {
            "Metric": ["ROC-AUC", "F1", "Recall", "Precision"],
            "Mean":   [0.9853, 0.8145, 0.9324, 0.7247],
            "Std":    [0.0057, 0.0280, 0.0258, 0.0441],
        }
        st.dataframe(pd.DataFrame(cv_data).style.format(
            {"Mean": "{:.4f}", "Std": "{:.4f}"}),
            use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Overfitting Table**")
        ov_data = {
            "Split":    ["Train (CV mean)", "Val (CV mean)", "Test (holdout)"],
            "AUC":      [0.9880, 0.9853, 0.9931],
            "Status":   ["—", "Gap=0.003", "HEALTHY"],
        }
        st.dataframe(pd.DataFrame(ov_data).style.format({"AUC": "{:.4f}"}),
                     use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Threshold Decision Guide</div>',
                unsafe_allow_html=True)
    thr_data = {
        "Threshold": [0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80],
        "Recall":    [0.982, 0.982, 0.982, 0.954, 0.927, 0.908, 0.862],
        "Precision": [0.652, 0.677, 0.695, 0.776, 0.821, 0.861, 0.913],
        "F1":        [0.784, 0.802, 0.814, 0.856, 0.871, 0.884, 0.887],
        "Flagged":   [164, 158, 154, 134, 123, 115, 103],
    }
    thr_df = pd.DataFrame(thr_data)

    def highlight_recommended(row):
        if row["Threshold"] == 0.35:
            return ["background-color: #EFF6FF; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        thr_df.style.apply(highlight_recommended, axis=1).format({
            "Recall": "{:.3f}", "Precision": "{:.3f}", "F1": "{:.3f}"
        }),
        use_container_width=True, hide_index=True
    )
    st.caption("Highlighted row = recommended threshold for retention campaign")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Ensembling
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">Ensemble Comparison</div>', unsafe_allow_html=True)
    show_chart("outputs/charts/ensembling/ensemble_comparison.png")

    st.markdown('<div class="section-header">Occam\'s Razor Decision</div>',
                unsafe_allow_html=True)
    ens_data = {
        "Model":           ["LogisticRegression", "XGBoost_tuned", "SoftVoting",
                             "WeightedAverage",   "Stacking"],
        "AUC":             [0.9918, 0.9931, 0.9930, 0.9929, 0.9930],
        "F1":              [0.8560, 0.8750, 0.8996, 0.8918, 0.8899],
        "Gain vs Best":    [0.0000, 0.0000, -0.0001, -0.0002, -0.0002],
    }
    ens_df = pd.DataFrame(ens_data)

    def highlight_final(row):
        if row["Model"] == "XGBoost_tuned":
            return ["background-color: #EFF6FF; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        ens_df.style.apply(highlight_final, axis=1).format({
            "AUC": "{:.4f}", "F1": "{:.4f}", "Gain vs Best": "{:+.4f}"
        }),
        use_container_width=True, hide_index=True
    )
    st.info("**Occam's Razor:** Max ensemble gain = −0.0001 (ensembles don't beat XGBoost). "
            "XGBoost_tuned selected as final model.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Live Predictor
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-header">predict_brahma() — Live Customer Scoring</div>',
                unsafe_allow_html=True)

    # Compute training stats for slider defaults
    train_stats = drift_cfg["features"]

    # ── Quick-fill buttons ─────────────────────────────────────────────────────
    st.markdown("**Quick fill from dataset:**")
    qc1, qc2, qc3 = st.columns(3)

    # Find an obvious churner and retainer from test set
    df_test_tmp = pd.DataFrame(X_test, columns=feature_cols)
    df_test_tmp["true_label"] = y_test
    df_test_tmp["pred_prob"]  = xgb_model.predict_proba(X_test)[:, 1]
    high_risk_row  = df_test_tmp[df_test_tmp["true_label"] == 1].sort_values("pred_prob", ascending=False).iloc[0]
    low_risk_row   = df_test_tmp[df_test_tmp["true_label"] == 0].sort_values("pred_prob").iloc[0]
    medium_risk_row= df_test_tmp[(df_test_tmp["pred_prob"] > 0.4) & (df_test_tmp["pred_prob"] < 0.7)].iloc[0] if len(df_test_tmp[(df_test_tmp["pred_prob"] > 0.4) & (df_test_tmp["pred_prob"] < 0.7)]) > 0 else df_test_tmp.iloc[5]

    if qc1.button("Load HIGH Risk Customer", use_container_width=True):
        st.session_state["prefill"] = high_risk_row.to_dict()
    if qc2.button("Load MEDIUM Risk Customer", use_container_width=True):
        st.session_state["prefill"] = medium_risk_row.to_dict()
    if qc3.button("Load LOW Risk Customer", use_container_width=True):
        st.session_state["prefill"] = low_risk_row.to_dict()

    prefill = st.session_state.get("prefill", {})

    def pf(col, default):
        return float(prefill.get(col, default))

    st.markdown("---")
    st.markdown("**Adjust customer features:**")

    # ── Feature inputs ──────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Transaction Behaviour**")
        total_trans_ct = st.slider(
            "Transaction Count (12m)", 0, 150,
            int(pf("total_trans_ct", 0) * train_stats["total_trans_ct"]["std"]
                + train_stats["total_trans_ct"]["mean"]),
            help="Number of transactions in the last 12 months")
        total_trans_amt = st.slider(
            "Transaction Amount ($)", 0, 20000,
            int(pf("total_trans_amt", 0) * train_stats["total_trans_amt"]["std"]
                + train_stats["total_trans_amt"]["mean"]),
            step=100)
        total_amt_chng = st.slider("Amt Change Q4/Q1", 0.0, 4.0,
            round(pf("total_amt_chng_q4_q1", 0) * train_stats["total_amt_chng_q4_q1"]["std"]
                  + train_stats["total_amt_chng_q4_q1"]["mean"], 2), 0.01)
        total_ct_chng  = st.slider("Count Change Q4/Q1", 0.0, 4.0,
            round(pf("total_ct_chng_q4_q1", 0) * train_stats["total_ct_chng_q4_q1"]["std"]
                  + train_stats["total_ct_chng_q4_q1"]["mean"], 2), 0.01)

    with col_b:
        st.markdown("**Engagement & Activity**")
        months_inactive = st.slider("Months Inactive (12m)", 0, 6,
            int(pf("months_inactive_12_mon", 0) * train_stats["months_inactive_12_mon"]["std"]
                + train_stats["months_inactive_12_mon"]["mean"]))
        contacts_count  = st.slider("Contacts Count (12m)", 0, 6,
            int(pf("contacts_count_12_mon", 0) * train_stats["contacts_count_12_mon"]["std"]
                + train_stats["contacts_count_12_mon"]["mean"]))
        total_rel_count = st.slider("Relationship Count", 1, 6,
            int(pf("total_relationship_count", 0) * train_stats["total_relationship_count"]["std"]
                + train_stats["total_relationship_count"]["mean"]))
        months_on_book  = st.slider("Months on Book", 12, 60,
            int(pf("months_on_book", 0) * train_stats["months_on_book"]["std"]
                + train_stats["months_on_book"]["mean"]))

    with col_c:
        st.markdown("**Financial Profile**")
        credit_limit    = st.slider("Credit Limit ($)", 1000, 35000,
            int(pf("credit_limit", 0) * train_stats["credit_limit"]["std"]
                + train_stats["credit_limit"]["mean"]), 500)
        revolving_bal   = st.slider("Revolving Balance ($)", 0, 3000,
            int(pf("total_revolving_bal", 0) * train_stats["total_revolving_bal"]["std"]
                + train_stats["total_revolving_bal"]["mean"]), 100)
        utilization     = st.slider("Utilization Ratio", 0.0, 1.0,
            round(pf("avg_utilization_ratio", 0) * train_stats["avg_utilization_ratio"]["std"]
                  + train_stats["avg_utilization_ratio"]["mean"], 2), 0.01)
        num_complaints  = st.slider("Complaints (12m)", 0, 5,
            int(pf("num_complaints_12_mon", 0) * train_stats["num_complaints_12_mon"]["std"]
                + train_stats["num_complaints_12_mon"]["mean"]))

    # ── Build scaled input ─────────────────────────────────────────────────────
    raw_vals = {
        "total_trans_ct":          total_trans_ct,
        "total_trans_amt":         total_trans_amt,
        "total_amt_chng_q4_q1":    total_amt_chng,
        "total_ct_chng_q4_q1":     total_ct_chng,
        "months_inactive_12_mon":  months_inactive,
        "contacts_count_12_mon":   contacts_count,
        "total_relationship_count":total_rel_count,
        "months_on_book":          months_on_book,
        "credit_limit":            credit_limit,
        "total_revolving_bal":     revolving_bal,
        "avg_utilization_ratio":   utilization,
        "num_complaints_12_mon":   num_complaints,
    }

    # Standardize to match training scale
    def standardize(col, raw):
        mu = train_stats[col]["mean"]
        sd = train_stats[col]["std"] + 1e-9
        return (raw - mu) / sd

    scaled_input = {}
    for f in feature_cols:
        if f in raw_vals:
            scaled_input[f] = standardize(f, raw_vals[f])
        elif f in train_stats:
            scaled_input[f] = train_stats[f]["mean"] / (train_stats[f]["std"] + 1e-9) * 0
        else:
            scaled_input[f] = 0.0

    # Engineered features
    eps = 1e-9
    scaled_input["avg_transaction_value"]  = standardize("avg_transaction_value",
        total_trans_amt / (total_trans_ct + eps))
    scaled_input["engagement_velocity"]    = standardize("engagement_velocity",
        total_amt_chng * total_ct_chng)
    scaled_input["re_engagement_signal"]   = standardize("re_engagement_signal",
        months_inactive * contacts_count)
    scaled_input["transaction_intensity"]  = standardize("transaction_intensity",
        total_trans_ct / (months_on_book + eps))
    scaled_input["complaint_rate"]         = standardize("complaint_rate",
        num_complaints / (total_rel_count + eps))

    # ── Predict ────────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔮  Score This Customer", use_container_width=True, type="primary"):
        result = predict_brahma(scaled_input)
        prob   = result["probability"]
        tier   = result["risk_tier"]
        pred   = result["prediction"]

        # Result banner
        tier_color = {"HIGH": "#DC2626", "MEDIUM": "#D97706", "LOW": "#16A34A"}[tier]
        tier_bg    = {"HIGH": "#FEF2F2", "MEDIUM": "#FFFBEB", "LOW": "#F0FDF4"}[tier]
        pred_label = "CHURN" if pred == 1 else "RETAIN"

        st.markdown(f"""
        <div style="background:{tier_bg}; border-left:6px solid {tier_color};
                    border-radius:10px; padding:20px 24px; margin:12px 0;">
          <div style="font-size:1.6rem; font-weight:800; color:{tier_color};">
            {pred_label} &nbsp;|&nbsp; Risk: {tier}
          </div>
          <div style="font-size:2.4rem; font-weight:900; color:{tier_color}; margin:4px 0;">
            {prob:.1%} churn probability
          </div>
          <div style="color:#6B7280; font-size:0.85rem;">
            Model: {result['model_version']} &nbsp;|&nbsp;
            Latency: {result['latency_ms']} ms &nbsp;|&nbsp;
            {result['timestamp'][:19]}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.barh(["Churn Score"], [prob],      color=tier_color,  height=0.4)
        ax.barh(["Churn Score"], [1 - prob],  color=MUTED,       height=0.4,
                left=prob)
        ax.axvline(0.40, color="#D97706", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(0.70, color="#DC2626", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Churn Probability", fontsize=9)
        ax.text(0.20, 1.05, "LOW", transform=ax.transData, ha="center",
                fontsize=8, color="#16A34A", fontweight="bold")
        ax.text(0.55, 1.05, "MED", transform=ax.transData, ha="center",
                fontsize=8, color="#D97706", fontweight="bold")
        ax.text(0.85, 1.05, "HIGH", transform=ax.transData, ha="center",
                fontsize=8, color="#DC2626", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("white"); fig.set_facecolor("white")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Top reasons
        if result["top_reasons"]:
            st.markdown("**Top 3 reasons driving this score:**")
            for i, r in enumerate(result["top_reasons"], 1):
                st.markdown(f"**{i}.** `{r['feature']}` — "
                            f"importance={r['importance']:.4f}, "
                            f"scaled value={r['value']:.3f}")

        # Recommended action
        actions = {
            "HIGH":   "Immediate personal outreach within 48 hours. Offer: cashback uplift or credit limit increase.",
            "MEDIUM": "Automated personalised email + in-app offer within 7 days.",
            "LOW":    "No urgent action required. Include in standard retention newsletter.",
        }
        st.success(f"**Recommended action:** {actions[tier]}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — Pipeline Summary
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="section-header">Pipeline Execution Summary</div>',
                unsafe_allow_html=True)

    stages = [
        ("1",  "Data Ingestion",          "DONE", "5,000 rows loaded, 4/5 integrity checks PASS"),
        ("2",  "Preprocessing",           "DONE", "Imputed 3% missing, dropped 1 col (>50%), 2 cols winsorized, encoded, scaled"),
        ("3",  "EDA",                     "DONE", "6 charts — transaction count is #1 predictor"),
        ("4",  "Feature Engineering",     "DONE", "+5 features, -1 multicollinear, 23 final features"),
        ("5",  "Algorithm Selection",     "DONE", "XGBoost selected — imbalanced binary classification, N=5K"),
        ("6",  "Model Training",          "DONE", "4 models trained, Optuna 50 trials, best AUC_val=0.9931"),
        ("7",  "Model Evaluation",        "DONE", "AUC=0.9931, F1=0.875, Recall=95.4%, 6 charts"),
        ("8",  "Model Validation",        "DONE", "10-fold CV AUC=0.985±0.006, gap=0.003 HEALTHY"),
        ("9",  "Ensembling",              "DONE", "Ensembles gain < 0.005, Occam: XGBoost_tuned chosen"),
        ("10", "UAT",                     "DONE", "6/6 checks PASS — APPROVED FOR DEPLOYMENT"),
        ("11", "Deployment Testing",      "DONE", "predict_brahma() ready, drift detection armed, 179K pred/sec"),
        ("12", "Dashboard (Streamlit)",   "DONE", "You are here"),
        ("13", "Slide Deck (Gamma)",      "DONE", "gamma.app/docs/1j1jcs5fwzze3mf"),
    ]

    for num, name, status, detail in stages:
        icon = "✅"
        c1, c2, c3 = st.columns([1, 3, 6])
        c1.markdown(f"**Stage {num}**")
        c2.markdown(f"**{name}**")
        c3.markdown(f"{icon} {detail}")

    st.markdown("---")
    st.markdown("""
    **Outputs generated:**
    - `outputs/models/xgb_tuned.pkl` — production model
    - `outputs/models/deployment_package.pkl` — full deployment bundle
    - `outputs/data/drift_config.json` — drift monitoring config
    - `outputs/data/features_engineered.parquet` — final dataset
    - `outputs/charts/` — 15 charts across EDA, training, evaluation, validation, ensembling
    - Slide deck: [gamma.app/docs/1j1jcs5fwzze3mf](https://gamma.app/docs/1j1jcs5fwzze3mf)
    """)

    st.markdown('<div class="section-header">Key Finding & Recommendation</div>',
                unsafe_allow_html=True)
    st.markdown("""
    > **Finding:** Customers who transact less than their historic average and remain inactive
    > for 2+ months are **6× more likely to churn**. Transaction count alone (r=0.49) is the
    > strongest single predictor.

    > **Recommendation:** Deploy `predict_brahma()` as a weekly batch job. Route every customer
    > scoring above **0.35** to the retention team — this catches **98% of churners** while
    > keeping false positives manageable (~164 customers per cycle at the 0.35 threshold).
    """)
