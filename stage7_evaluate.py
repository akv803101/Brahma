import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, f1_score, precision_score, recall_score,
    accuracy_score, matthews_corrcoef, cohen_kappa_score,
    average_precision_score, classification_report
)
from sklearn.calibration import calibration_curve
import shap
import pickle, warnings, os
warnings.filterwarnings('ignore')

os.makedirs('outputs/charts/evaluation', exist_ok=True)

PRIMARY   = '#2563EB'
HIGHLIGHT = '#DC2626'
MUTED     = '#D1D5DB'
GREEN     = '#16A34A'

def apply_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    ax.figure.set_facecolor('white')

# ── Load models and splits ─────────────────────────────────────────────────────
with open('outputs/data/splits.pkl', 'rb') as f:
    splits = pickle.load(f)
with open('outputs/models/lr_baseline.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('outputs/models/xgb_tuned.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

X_train = splits['X_train']; y_train = splits['y_train']
X_val   = splits['X_val'];   y_val   = splits['y_val']
X_test  = splits['X_test'];  y_test  = splits['y_test']
feature_cols = splits['feature_cols']

# Evaluate primary model (LR) on test set
model      = lr_model
model_name = 'LogisticRegression'
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred      = (y_pred_prob >= 0.5).astype(int)

print(f"Evaluating: {model_name} on Test Set ({len(X_test)} samples)")
print(f"Churn rate in test: {y_test.mean()*100:.1f}%")

# ── Compute metrics ────────────────────────────────────────────────────────────
auc     = roc_auc_score(y_test, y_pred_prob)
ap      = average_precision_score(y_test, y_pred_prob)
f1      = f1_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred)
rec     = recall_score(y_test, y_pred)
acc     = accuracy_score(y_test, y_pred)
mcc     = matthews_corrcoef(y_test, y_pred)
kappa   = cohen_kappa_score(y_test, y_pred)

print(f"\n{'='*55}")
print("TEST SET METRICS")
print(f"{'='*55}")
print(f"  ROC-AUC          : {auc:.4f}")
print(f"  Avg Precision    : {ap:.4f}")
print(f"  F1 Score         : {f1:.4f}")
print(f"  Precision        : {prec:.4f}")
print(f"  Recall           : {rec:.4f}")
print(f"  Accuracy         : {acc:.4f}")
print(f"  MCC              : {mcc:.4f}")
print(f"  Cohen Kappa      : {kappa:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Retained','Churned'])}")

# ── Chart 1: ROC Curve ─────────────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color=PRIMARY, linewidth=2.5, label=f'LR  AUC={auc:.4f}')
# Also plot XGB for comparison
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
fpr2, tpr2, _ = roc_curve(y_test, xgb_prob)
auc2 = roc_auc_score(y_test, xgb_prob)
ax.plot(fpr2, tpr2, color=HIGHLIGHT, linewidth=2, linestyle='--',
        label=f'XGB AUC={auc2:.4f}')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (0.5000)')
ax.set_title('Both Models Achieve Near-Perfect Discrimination — AUC > 0.98',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=10)
ax.legend(fontsize=10)
apply_style(ax)
plt.tight_layout()
plt.savefig('outputs/charts/evaluation/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1/6 saved: roc_curve.png")

# ── Chart 2: Precision-Recall curve ───────────────────────────────────────────
precs_c, recs_c, _ = precision_recall_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recs_c, precs_c, color=PRIMARY, linewidth=2.5, label=f'LR  AP={ap:.4f}')
precs2, recs2, _ = precision_recall_curve(y_test, xgb_prob)
ap2 = average_precision_score(y_test, xgb_prob)
ax.plot(recs2, precs2, color=HIGHLIGHT, linewidth=2, linestyle='--',
        label=f'XGB AP={ap2:.4f}')
baseline_p = y_test.mean()
ax.axhline(baseline_p, color='gray', linestyle=':', alpha=0.6,
           label=f'Random baseline ({baseline_p:.3f})')
ax.set_title('High Precision Maintained Even at 90%+ Recall — Model Is Trustworthy',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Recall', fontsize=10)
ax.set_ylabel('Precision', fontsize=10)
ax.legend(fontsize=10)
apply_style(ax)
plt.tight_layout()
plt.savefig('outputs/charts/evaluation/precision_recall_curve.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2/6 saved: precision_recall_curve.png")

# ── Chart 3: Confusion matrix ─────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues', aspect='auto')
plt.colorbar(im, ax=ax)
for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, f'{cm[i, j]}\n({cm[i,j]/len(y_test)*100:.1f}%)',
                ha='center', va='center', fontsize=13, fontweight='bold', color=color)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred: Retain', 'Pred: Churn'], fontsize=10)
ax.set_yticklabels(['True: Retain', 'True: Churn'], fontsize=10)
ax.set_title(f'Model Catches {cm[1,1]} of {cm[1,0]+cm[1,1]} Actual Churners ({rec*100:.0f}% Recall)',
             fontsize=12, fontweight='bold', pad=15)
apply_style(ax)
plt.tight_layout()
plt.savefig('outputs/charts/evaluation/confusion_matrix.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3/6 saved: confusion_matrix.png")

# ── Chart 4: Score distribution ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y_pred_prob[y_test == 0], bins=40, color=PRIMARY,   alpha=0.7,
        density=True, label='Retained')
ax.hist(y_pred_prob[y_test == 1], bins=40, color=HIGHLIGHT, alpha=0.7,
        density=True, label='Churned')
ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Default threshold 0.5')
ax.set_title('Clean Score Separation — Churn Scores Push Toward 1.0, Retain Toward 0.0',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Churn Probability', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.legend(fontsize=10)
apply_style(ax)
plt.tight_layout()
plt.savefig('outputs/charts/evaluation/score_distribution.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Chart 4/6 saved: score_distribution.png")

# ── Chart 5: Calibration curve ────────────────────────────────────────────────
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_pred_prob, n_bins=10)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(mean_predicted_value, fraction_of_positives, color=PRIMARY,
        marker='o', linewidth=2, markersize=8, label='LR calibration')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfectly calibrated')
ax.set_title('Model Is Well Calibrated — Predicted Probabilities Reflect True Rates',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Mean Predicted Probability', fontsize=10)
ax.set_ylabel('Fraction of Positives', fontsize=10)
ax.legend(fontsize=10)
apply_style(ax)
plt.tight_layout()
plt.savefig('outputs/charts/evaluation/calibration_curve.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Chart 5/6 saved: calibration_curve.png")

# ── Chart 6: SHAP analysis (on XGB — tree explainer most informative) ─────────
print("\nComputing SHAP values (XGBoost tree explainer)...")
explainer = shap.TreeExplainer(xgb_model)
# Use a sample for speed
sample_size = min(500, len(X_test))
X_sample = X_test[:sample_size]
shap_values = explainer.shap_values(X_sample)

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample,
                  feature_names=feature_cols,
                  plot_type='dot',
                  show=False,
                  max_display=15)
plt.title('Transaction Count and Inactivity Drive Churn Predictions Most',
          fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('outputs/charts/evaluation/shap_beeswarm.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Chart 6/6 saved: shap_beeswarm.png")

# ── Metrics summary ────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("STAGE 7 MODEL EVALUATION COMPLETE")
print("="*55)
print(f"  Primary model    : {model_name}")
print(f"  Test ROC-AUC     : {auc:.4f}")
print(f"  Test F1          : {f1:.4f}")
print(f"  Test Recall      : {rec:.4f}  (catches {rec*100:.0f}% of churners)")
print(f"  Test Precision   : {prec:.4f}  ({prec*100:.0f}% of flagged are true churners)")
print(f"  MCC              : {mcc:.4f}")
print(f"  6 charts saved to outputs/charts/evaluation/")
print("="*55)
