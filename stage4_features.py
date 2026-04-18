import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('outputs/charts/eda', exist_ok=True)
os.makedirs('outputs/data', exist_ok=True)

PRIMARY   = '#2563EB'
HIGHLIGHT = '#DC2626'

def apply_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    ax.figure.set_facecolor('white')

df = pd.read_parquet('outputs/data/preprocessed.parquet')
target_col = 'churn_flag'
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"Starting feature count (excl target): {df.shape[1] - 1}")

# ── Domain feature engineering (financial / churn domain) ─────────────────────
print("\n--- ENGINEERING NEW FEATURES ---")

# avg transaction value: how much each transaction is worth
# (data is standardized, so ratios still capture relative signal)
df['avg_transaction_value'] = np.where(
    df['total_trans_ct'] != 0,
    df['total_trans_amt'] / (df['total_trans_ct'] + 1e-9),
    0
)
print("  + avg_transaction_value  (total_trans_amt / total_trans_ct)")

# engagement velocity: quarter-over-quarter change in both amount AND count
df['engagement_velocity'] = df['total_amt_chng_q4_q1'] * df['total_ct_chng_q4_q1']
print("  + engagement_velocity    (amt_chng * ct_chng — combined momentum)")

# re-engagement signal: inactive * contacts = high inactivity AND high contact = distress
df['re_engagement_signal'] = df['months_inactive_12_mon'] * df['contacts_count_12_mon']
print("  + re_engagement_signal   (months_inactive * contacts_count)")

# transaction intensity: trans count relative to months on book
df['transaction_intensity'] = df['total_trans_ct'] / (df['months_on_book'] + 1e-9)
print("  + transaction_intensity  (trans_ct / months_on_book)")

# complaint rate: complaints per relationship count
df['complaint_rate'] = df['num_complaints_12_mon'] / (df['total_relationship_count'] + 1e-9)
print("  + complaint_rate         (complaints / relationship_count)")

print(f"\nNew feature count (excl target): {df.shape[1] - 1}")

# ── Drop high-correlation redundant feature ────────────────────────────────────
print("\n--- DROPPING REDUNDANT FEATURES ---")
print("  avg_open_to_buy: r=0.960 with credit_limit (avg_open_to_buy = credit_limit - revolving_bal)")
print("  credit_limit already present. Dropping avg_open_to_buy to remove multicollinearity.")
if 'avg_open_to_buy' in df.columns:
    df = df.drop(columns=['avg_open_to_buy'])
    print("  DROPPED: avg_open_to_buy")

# ── Variance filter ───────────────────────────────────────────────────────────
print("\n--- VARIANCE FILTER (threshold < 0.01, numeric only) ---")
feature_cols = [c for c in df.columns if c != target_col]
numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
variances = df[numeric_features].var()
low_var = variances[variances < 0.01].index.tolist()
if low_var:
    print(f"  LOW VARIANCE features: {low_var}")
    df = df.drop(columns=low_var)
    print(f"  DROPPED: {low_var}")
else:
    print("  No low-variance numeric features found.")

# ── Random Forest feature importance ──────────────────────────────────────────
print("\n--- COMPUTING FEATURE IMPORTANCE (RandomForest, 100 trees) ---")
feature_cols = [c for c in df.columns if c != target_col]
# RF requires numeric input — use numeric features only for importance ranking
numeric_for_rf = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"  Using {len(numeric_for_rf)} numeric features for RF importance")
X = df[numeric_for_rf]
y = df[target_col]

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

importance_df = pd.DataFrame({
    'feature': numeric_for_rf,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Feature Importance Rankings:")
print(f"  {'Feature':<40} {'Importance':>10}")
print(f"  {'-'*52}")
for _, row in importance_df.iterrows():
    flag = "  <-- DROP" if row['importance'] < 0.001 else ""
    print(f"  {row['feature']:<40} {row['importance']:>10.4f}{flag}")

# Drop features with importance < 0.001
weak = importance_df[importance_df['importance'] < 0.001]['feature'].tolist()
if weak:
    print(f"\n  DROPPING weak features (importance < 0.001): {weak}")
    df = df.drop(columns=weak)
    importance_df = importance_df[~importance_df['feature'].isin(weak)]
else:
    print("\n  No features below importance threshold.")

# ── Plot feature importance ────────────────────────────────────────────────────
top20 = importance_df.head(20)
fig, ax = plt.subplots(figsize=(10, 8))
bar_colors = [HIGHLIGHT if v >= top20['importance'].quantile(0.7) else PRIMARY
              for v in top20['importance'].values]
ax.barh(top20['feature'][::-1], top20['importance'][::-1],
        color=bar_colors[::-1], edgecolor='white')
ax.set_title(
    'Transaction Behaviour Dominates — Engagement Features Drive Churn Prediction',
    fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Random Forest Feature Importance', fontsize=10)
apply_style(ax)
plt.tight_layout()
plt.savefig('outputs/charts/eda/feature_importance_top20.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nChart saved: feature_importance_top20.png")

# ── Save engineered dataset ────────────────────────────────────────────────────
df.to_parquet('outputs/data/features_engineered.parquet', index=False)

final_features = [c for c in df.columns if c != target_col]
print("\n" + "="*60)
print("STAGE 4 FEATURE ENGINEERING COMPLETE")
print("="*60)
print(f"  Original features : 19")
print(f"  + Engineered      : 5  (avg_txn_value, eng_velocity, re_engage, txn_intensity, complaint_rate)")
print(f"  - Dropped (multicollinearity) : 1  (avg_open_to_buy)")
print(f"  - Dropped (low variance)      : {len(low_var)}")
print(f"  - Dropped (low importance)    : {len(weak)}")
print(f"  Final feature count           : {len(final_features)}")
print(f"\n  Saved: outputs/data/features_engineered.parquet")
print(f"  Shape: {df.shape}")
print("="*60)
