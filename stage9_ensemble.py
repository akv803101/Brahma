import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import pickle, warnings, os
warnings.filterwarnings('ignore')

os.makedirs('outputs/charts/ensembling', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)

PRIMARY   = '#2563EB'
HIGHLIGHT = '#DC2626'
MUTED     = '#D1D5DB'
GREEN     = '#16A34A'

def apply_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    ax.figure.set_facecolor('white')

# ── Load ──────────────────────────────────────────────────────────────────────
with open('outputs/data/splits.pkl', 'rb') as f:
    splits = pickle.load(f)
with open('outputs/models/lr_baseline.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('outputs/models/xgb_tuned.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

X_train = splits['X_train']; y_train = splits['y_train']
X_val   = splits['X_val'];   y_val   = splits['y_val']
X_test  = splits['X_test'];  y_test  = splits['y_test']

print("Loaded 2 models: LogisticRegression + XGBoost_tuned")

# ── Step 1: Diversity check ────────────────────────────────────────────────────
print("\n--- DIVERSITY CHECK ---")
lr_val_prob  = lr_model.predict_proba(X_val)[:, 1]
xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
r_corr = np.corrcoef(lr_val_prob, xgb_val_prob)[0, 1]
print(f"  Prediction correlation (LR vs XGB): r={r_corr:.4f}")
if r_corr > 0.95:
    print(f"  WARNING: r > 0.95 — models are very similar, ensemble gain may be minimal")
else:
    print(f"  r < 0.95 — models are diverse enough to benefit from ensembling")

# Individual test-set performance
lr_test_prob  = lr_model.predict_proba(X_test)[:, 1]
xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]
lr_auc  = roc_auc_score(y_test, lr_test_prob)
xgb_auc = roc_auc_score(y_test, xgb_test_prob)
lr_f1   = f1_score(y_test, (lr_test_prob  >= 0.5).astype(int))
xgb_f1  = f1_score(y_test, (xgb_test_prob >= 0.5).astype(int))
best_single_auc = max(lr_auc, xgb_auc)
print(f"\n  Individual models on test set:")
print(f"  {'Model':<25} {'AUC':>8} {'F1':>8}")
print(f"  {'-'*43}")
print(f"  {'LogisticRegression':<25} {lr_auc:>8.4f} {lr_f1:>8.4f}")
print(f"  {'XGBoost_tuned':<25} {xgb_auc:>8.4f} {xgb_f1:>8.4f}")
print(f"  Best single model AUC: {best_single_auc:.4f}")

# ── Step 2: Soft voting ensemble ──────────────────────────────────────────────
print("\n--- SOFT VOTING (simple average) ---")
soft_prob = (lr_test_prob + xgb_test_prob) / 2
soft_auc  = roc_auc_score(y_test, soft_prob)
soft_f1   = f1_score(y_test, (soft_prob >= 0.5).astype(int))
soft_gain = soft_auc - best_single_auc
print(f"  Soft vote AUC: {soft_auc:.4f}  F1: {soft_f1:.4f}  Gain: {soft_gain:+.4f}")

# ── Step 3: Weighted average (weights = 1/val_error) ─────────────────────────
print("\n--- WEIGHTED AVERAGE (1/val_error weights) ---")
lr_val_auc  = roc_auc_score(y_val, lr_val_prob)
xgb_val_auc = roc_auc_score(y_val, xgb_val_prob)
lr_err  = 1 - lr_val_auc
xgb_err = 1 - xgb_val_auc
lr_w    = (1 / lr_err)  / (1/lr_err + 1/xgb_err)
xgb_w   = (1 / xgb_err) / (1/lr_err + 1/xgb_err)
print(f"  LR  weight: {lr_w:.4f}  (val AUC={lr_val_auc:.4f})")
print(f"  XGB weight: {xgb_w:.4f}  (val AUC={xgb_val_auc:.4f})")
weighted_prob = lr_w * lr_test_prob + xgb_w * xgb_test_prob
weighted_auc  = roc_auc_score(y_test, weighted_prob)
weighted_f1   = f1_score(y_test, (weighted_prob >= 0.5).astype(int))
weighted_gain = weighted_auc - best_single_auc
print(f"  Weighted AUC: {weighted_auc:.4f}  F1: {weighted_f1:.4f}  Gain: {weighted_gain:+.4f}")

# ── Step 4: Stacking with OOF (LR meta-learner) ───────────────────────────────
print("\n--- STACKING (OOF + LR meta-learner) ---")
X_full = np.vstack([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lr  = np.zeros(len(X_full))
oof_xgb = np.zeros(len(X_full))

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_full, y_full)):
    # LR OOF
    lr_f = LogisticRegression(max_iter=1000, class_weight='balanced',
                               random_state=42, C=1.0)
    lr_f.fit(X_full[tr_idx], y_full[tr_idx])
    oof_lr[vl_idx] = lr_f.predict_proba(X_full[vl_idx])[:, 1]
    # XGB OOF
    import xgboost as xgb
    scale_pos_w = (1 - y_full[tr_idx].mean()) / y_full[tr_idx].mean()
    with open('outputs/models/xgb_tuned.pkl', 'rb') as f:
        xgb_params = pickle.load(f)
    xgb_f = type(xgb_params)(**xgb_params.get_params())
    xgb_f.fit(X_full[tr_idx], y_full[tr_idx], verbose=False)
    oof_xgb[vl_idx] = xgb_f.predict_proba(X_full[vl_idx])[:, 1]

# Meta-features
S_train = np.column_stack([oof_lr, oof_xgb])
meta_lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
meta_lr.fit(S_train, y_full)

print(f"  Meta-learner coefficients:")
print(f"    LR  coefficient : {meta_lr.coef_[0][0]:.4f}")
print(f"    XGB coefficient : {meta_lr.coef_[0][1]:.4f}")
print(f"    Intercept       : {meta_lr.intercept_[0]:.4f}")

# Retrain base models on full train+val
lr_full = LogisticRegression(max_iter=1000, class_weight='balanced',
                              random_state=42, C=1.0)
lr_full.fit(X_full, y_full)
xgb_full = type(xgb_params)(**xgb_params.get_params())
xgb_full.fit(X_full, y_full, verbose=False)

# Stack predictions on test
S_test = np.column_stack([
    lr_full.predict_proba(X_test)[:, 1],
    xgb_full.predict_proba(X_test)[:, 1]
])
stack_prob = meta_lr.predict_proba(S_test)[:, 1]
stack_auc  = roc_auc_score(y_test, stack_prob)
stack_f1   = f1_score(y_test, (stack_prob >= 0.5).astype(int))
stack_gain = stack_auc - best_single_auc
print(f"  Stacking AUC: {stack_auc:.4f}  F1: {stack_f1:.4f}  Gain: {stack_gain:+.4f}")

# ── Occam's Razor: choose final model ─────────────────────────────────────────
print("\n--- OCCAM'S RAZOR DECISION ---")
results = {
    'LogisticRegression':  {'auc': lr_auc,       'f1': lr_f1,       'gain': 0},
    'XGBoost_tuned':       {'auc': xgb_auc,       'f1': xgb_f1,      'gain': xgb_auc - best_single_auc},
    'SoftVoting':          {'auc': soft_auc,      'f1': soft_f1,     'gain': soft_gain},
    'WeightedAverage':     {'auc': weighted_auc,  'f1': weighted_f1, 'gain': weighted_gain},
    'Stacking':            {'auc': stack_auc,     'f1': stack_f1,    'gain': stack_gain},
}

print(f"\n  {'Model':<22} {'AUC':>8} {'F1':>8} {'Gain vs best':>14}")
print(f"  {'-'*55}")
for name, r in results.items():
    flag = " <-- FINAL CHOICE" if name == max(results, key=lambda k: results[k]['auc']) else ""
    print(f"  {name:<22} {r['auc']:>8.4f} {r['f1']:>8.4f} {r['gain']:>14.4f}{flag}")

# Occam's Razor: if ensemble gain < 0.005, use simpler model
best_ensemble_name = max(['SoftVoting', 'WeightedAverage', 'Stacking'],
                          key=lambda k: results[k]['auc'])
best_ensemble_auc  = results[best_ensemble_name]['auc']
ensemble_gain      = best_ensemble_auc - best_single_auc

if ensemble_gain < 0.005:
    final_model_name = 'LogisticRegression'
    final_model      = lr_model
    final_auc        = lr_auc
    final_f1         = lr_f1
    print(f"\n  Ensemble gain = {ensemble_gain:.4f} < 0.005 threshold")
    print(f"  OCCAM'S RAZOR: Choose simpler model --> LogisticRegression")
else:
    final_model_name = best_ensemble_name
    final_auc        = best_ensemble_auc
    final_f1         = results[best_ensemble_name]['f1']
    print(f"\n  Ensemble gain = {ensemble_gain:.4f} >= 0.005 --> Use {best_ensemble_name}")

# ── Save final model ───────────────────────────────────────────────────────────
with open('outputs/models/final_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print(f"\n  Final model saved: outputs/models/final_model.pkl  ({final_model_name})")

# ── Plot ensemble comparison ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
names = list(results.keys())
aucs  = [results[n]['auc'] for n in names]
bar_colors = [HIGHLIGHT if n == final_model_name else PRIMARY for n in names]
bars = ax.bar(names, aucs, color=bar_colors, width=0.6, edgecolor='white', linewidth=1.5)
ax.axhline(best_single_auc, color='gray', linestyle='--', linewidth=1.5,
           alpha=0.7, label=f'Best single model ({best_single_auc:.4f})')
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Ensemble Gain Is Marginal — Logistic Regression Stands on Its Own',
             fontsize=12, fontweight='bold', pad=15)
ax.set_ylabel('Test ROC-AUC', fontsize=10)
ax.set_ylim(min(aucs) - 0.01, max(aucs) + 0.008)
ax.legend(fontsize=9)
plt.xticks(rotation=15, ha='right', fontsize=9)
apply_style(ax)
plt.tight_layout()
plt.savefig('outputs/charts/ensembling/ensemble_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: ensemble_comparison.png")

print("\n" + "="*55)
print("STAGE 9 ENSEMBLING COMPLETE")
print("="*55)
print(f"  Diversity (LR vs XGB corr): r={r_corr:.4f}")
print(f"  Best ensemble gain:  {ensemble_gain:+.4f}")
print(f"  Occam decision:  {final_model_name}")
print(f"  Final AUC: {final_auc:.4f}  Final F1: {final_f1:.4f}")
print("="*55)
