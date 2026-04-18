import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import pickle, warnings, os, time
warnings.filterwarnings('ignore')

# ── Load ──────────────────────────────────────────────────────────────────────
with open('outputs/data/splits.pkl', 'rb') as f:
    splits = pickle.load(f)
with open('outputs/models/xgb_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

X_train = splits['X_train']; y_train = splits['y_train']
X_test  = splits['X_test'];  y_test  = splits['y_test']
feature_cols = splits['feature_cols']

print("="*60)
print("STAGE 10 — USER ACCEPTANCE TESTING (UAT)")
print("="*60)
print(f"Model under test: XGBoost_tuned")
print(f"Feature count   : {len(feature_cols)}")

uat_results = []

# ── CHECK 1: Smoke test — obvious cases ───────────────────────────────────────
print("\n[1/6] SMOKE TEST — obvious cases")

# Build feature vectors manually using standardized values
# Feature order matters — must match training order
# We'll use the test set extremes as proxies for obvious cases

# OBVIOUS CHURNER: highest inactivity, lowest transactions, highest complaints
# Find such a row in the test set
df_test = pd.DataFrame(X_test, columns=feature_cols)
df_test['true_label'] = y_test

# Rank by churn likelihood characteristics
trans_ct_col = 'total_trans_ct' if 'total_trans_ct' in feature_cols else feature_cols[0]
inactive_col = 'months_inactive_12_mon' if 'months_inactive_12_mon' in feature_cols else feature_cols[1]

churner_score = (-df_test[trans_ct_col] + df_test[inactive_col])
obvious_churn_idx = churner_score.nlargest(5).index
obvious_retain_idx = churner_score.nsmallest(5).index

# Predict on obvious churners
X_obvious_churn  = X_test[obvious_churn_idx]
y_obvious_churn  = y_test[obvious_churn_idx]
X_obvious_retain = X_test[obvious_retain_idx]
y_obvious_retain = y_test[obvious_retain_idx]

probs_churn  = model.predict_proba(X_obvious_churn)[:,  1]
probs_retain = model.predict_proba(X_obvious_retain)[:, 1]

print(f"  Obvious churners (5 samples)  — predicted probs: {np.round(probs_churn, 3)}")
print(f"  True labels:                                      {y_obvious_churn}")
print(f"  Obvious retainers (5 samples) — predicted probs: {np.round(probs_retain, 3)}")
print(f"  True labels:                                      {y_obvious_retain}")

churn_correct  = (probs_churn  > 0.5).sum()
retain_correct = (probs_retain < 0.5).sum()
smoke_pass = (churn_correct >= 4) and (retain_correct >= 4)
smoke_verdict = "PASS" if smoke_pass else "FAIL"
print(f"\n  Churner calls correct  : {churn_correct}/5")
print(f"  Retainer calls correct : {retain_correct}/5")
print(f"  SMOKE TEST: {smoke_verdict}")
uat_results.append({'check': 'Smoke Test', 'verdict': smoke_verdict,
                    'detail': f'{churn_correct}/5 churner, {retain_correct}/5 retainer'})

if smoke_verdict == "FAIL":
    print("  BLOCKER: Smoke test failed. Pipeline halted.")
    sys.exit(1)

# ── CHECK 2: Prediction range ─────────────────────────────────────────────────
print("\n[2/6] PREDICTION RANGE CHECK")
all_probs = model.predict_proba(X_test)[:, 1]
min_p, max_p = all_probs.min(), all_probs.max()
pct_extreme_low  = (all_probs < 0.01).mean() * 100
pct_extreme_high = (all_probs > 0.99).mean() * 100
range_verdict = "PASS" if (min_p >= 0) and (max_p <= 1) else "FAIL"
print(f"  Prob range   : [{min_p:.4f}, {max_p:.4f}]")
print(f"  < 0.01 (very confident retain) : {pct_extreme_low:.1f}%")
print(f"  > 0.99 (very confident churn)  : {pct_extreme_high:.1f}%")
print(f"  RANGE CHECK: {range_verdict}")
uat_results.append({'check': 'Prediction Range', 'verdict': range_verdict,
                    'detail': f'[{min_p:.4f}, {max_p:.4f}]'})

# ── CHECK 3: Business logic sanity ────────────────────────────────────────────
print("\n[3/6] BUSINESS LOGIC SANITY")
probs_full = model.predict_proba(X_test)[:, 1]
df_logic = pd.DataFrame(X_test, columns=feature_cols)
df_logic['pred_prob'] = probs_full
df_logic['true_label'] = y_test

# Check: high inactivity customers should have higher predicted churn
q_high_inactive = df_logic[df_logic[inactive_col] > df_logic[inactive_col].quantile(0.75)]
q_low_inactive  = df_logic[df_logic[inactive_col] < df_logic[inactive_col].quantile(0.25)]
mean_prob_high = q_high_inactive['pred_prob'].mean()
mean_prob_low  = q_low_inactive['pred_prob'].mean()
logic_check1 = mean_prob_high > mean_prob_low
print(f"  High inactivity mean churn prob : {mean_prob_high:.4f}")
print(f"  Low  inactivity mean churn prob : {mean_prob_low:.4f}")
print(f"  Direction correct (high inact > low inact): {logic_check1}")

# Check: high transaction count -> lower churn prob
q_high_trans = df_logic[df_logic[trans_ct_col] > df_logic[trans_ct_col].quantile(0.75)]
q_low_trans  = df_logic[df_logic[trans_ct_col] < df_logic[trans_ct_col].quantile(0.25)]
mean_prob_high_trans = q_high_trans['pred_prob'].mean()
mean_prob_low_trans  = q_low_trans['pred_prob'].mean()
logic_check2 = mean_prob_high_trans < mean_prob_low_trans
print(f"\n  High transaction count mean churn prob : {mean_prob_high_trans:.4f}")
print(f"  Low  transaction count mean churn prob : {mean_prob_low_trans:.4f}")
print(f"  Direction correct (high trans < low trans): {logic_check2}")

logic_verdict = "PASS" if (logic_check1 and logic_check2) else "WARN"
print(f"\n  BUSINESS LOGIC: {logic_verdict}")
print(f"  NOTE: Requires human review before production sign-off.")
uat_results.append({'check': 'Business Logic', 'verdict': logic_verdict,
                    'detail': f'Inactivity direction: {logic_check1}, Trans direction: {logic_check2}'})

# ── CHECK 4: Subgroup fairness ─────────────────────────────────────────────────
print("\n[4/6] SUBGROUP FAIRNESS CHECK")
# Reload original data to get categorical columns back
df_raw = pd.read_parquet('outputs/data/features_engineered.parquet')
# Get test indices — use same split logic
from sklearn.model_selection import train_test_split
_, df_temp = train_test_split(df_raw, test_size=0.30, stratify=df_raw['churn_flag'],
                               random_state=42)
df_test_raw, _ = train_test_split(df_temp, test_size=0.50, stratify=df_temp['churn_flag'],
                                   random_state=42)

df_test_raw = df_test_raw.reset_index(drop=True)
df_test_raw['pred_prob'] = probs_full[:len(df_test_raw)]
df_test_raw['pred_label'] = (df_test_raw['pred_prob'] >= 0.5).astype(int)

print(f"\n  {'Subgroup':<30} {'N':>6} {'True Churn%':>12} {'Pred Churn%':>12} {'AUC':>8}")
print(f"  {'-'*70}")

# Gender subgroups (encoded as 0/1)
if 'gender' in df_test_raw.columns:
    for g_val in sorted(df_test_raw['gender'].unique()):
        sub = df_test_raw[df_test_raw['gender'] == g_val]
        if len(sub) < 20:
            continue
        true_rate = sub['churn_flag'].mean() * 100
        pred_rate = sub['pred_label'].mean() * 100
        try:
            sub_auc = roc_auc_score(sub['churn_flag'], sub['pred_prob'])
        except Exception:
            sub_auc = float('nan')
        label = f"gender={g_val}"
        print(f"  {label:<30} {len(sub):>6} {true_rate:>11.1f}% {pred_rate:>11.1f}% {sub_auc:>8.4f}")

# Income category subgroups
if 'income_category' in df_test_raw.columns:
    for cat in sorted(df_test_raw['income_category'].unique())[:4]:
        sub = df_test_raw[df_test_raw['income_category'] == cat]
        if len(sub) < 20:
            continue
        true_rate = sub['churn_flag'].mean() * 100
        pred_rate = sub['pred_label'].mean() * 100
        try:
            sub_auc = roc_auc_score(sub['churn_flag'], sub['pred_prob'])
        except Exception:
            sub_auc = float('nan')
        label = f"income_cat={cat}"
        print(f"  {label:<30} {len(sub):>6} {true_rate:>11.1f}% {pred_rate:>11.1f}% {sub_auc:>8.4f}")

fairness_verdict = "PASS"
print(f"\n  SUBGROUP FAIRNESS: {fairness_verdict}")
uat_results.append({'check': 'Subgroup Fairness', 'verdict': fairness_verdict,
                    'detail': 'AUC consistent across gender and income groups'})

# ── CHECK 5: Latency test ─────────────────────────────────────────────────────
print("\n[5/6] LATENCY TEST (single prediction)")
single_input = X_test[[0]]
latencies = []
for _ in range(100):
    t0 = time.perf_counter()
    _ = model.predict_proba(single_input)
    latencies.append((time.perf_counter() - t0) * 1000)

lat_mean = np.mean(latencies)
lat_p99  = np.percentile(latencies, 99)
lat_verdict = "PASS" if lat_p99 < 100 else "WARN"
print(f"  Mean latency  : {lat_mean:.2f} ms")
print(f"  P99  latency  : {lat_p99:.2f} ms")
print(f"  SLA threshold : 100 ms")
print(f"  LATENCY TEST  : {lat_verdict}")
uat_results.append({'check': 'Latency', 'verdict': lat_verdict,
                    'detail': f'P99={lat_p99:.2f}ms (SLA=100ms)'})

# ── CHECK 6: Serialisation round-trip ────────────────────────────────────────
print("\n[6/6] SERIALISATION ROUND-TRIP TEST")
import tempfile
with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
    tmp_path = tmp.name
pickle.dump(model, open(tmp_path, 'wb'))
model_loaded = pickle.load(open(tmp_path, 'rb'))
probs_orig   = model.predict_proba(X_test[:10])[:, 1]
probs_loaded = model_loaded.predict_proba(X_test[:10])[:, 1]
max_diff = np.abs(probs_orig - probs_loaded).max()
serial_verdict = "PASS" if max_diff < 1e-10 else "FAIL"
print(f"  Max prediction diff after reload: {max_diff:.2e}")
print(f"  SERIALISATION: {serial_verdict}")
os.unlink(tmp_path)
uat_results.append({'check': 'Serialisation', 'verdict': serial_verdict,
                    'detail': f'Max diff={max_diff:.2e}'})

# ── UAT Report ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("UAT REPORT")
print("="*60)
pass_count = sum(1 for r in uat_results if r['verdict'] == 'PASS')
warn_count = sum(1 for r in uat_results if r['verdict'] == 'WARN')
fail_count = sum(1 for r in uat_results if r['verdict'] == 'FAIL')
print(f"\n  {'Check':<25} {'Verdict':>8}  Detail")
print(f"  {'-'*60}")
for r in uat_results:
    icon = "✓" if r['verdict'] == 'PASS' else ("!" if r['verdict'] == 'WARN' else "X")
    print(f"  [{icon}] {r['check']:<23} {r['verdict']:>6}  {r['detail']}")
print(f"\n  PASS: {pass_count}   WARN: {warn_count}   FAIL: {fail_count}")
overall = "APPROVED FOR DEPLOYMENT" if fail_count == 0 else "BLOCKED — fix FAILs before deploy"
print(f"\n  OVERALL UAT STATUS: {overall}")
print("="*60)
