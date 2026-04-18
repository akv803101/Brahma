import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('outputs/charts/eda', exist_ok=True)

PRIMARY   = '#2563EB'
HIGHLIGHT = '#DC2626'
MUTED     = '#D1D5DB'
BG        = 'white'

def apply_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)

df = pd.read_parquet('outputs/data/preprocessed.parquet')
print(f"Loaded preprocessed data: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"Columns: {list(df.columns)}")

target_col = 'churn_flag'

# ── Chart 1: Target distribution ──────────────────────────────────────────────
churn_counts = df[target_col].value_counts().sort_index()
labels = ['Retained', 'Churned']
colors = [PRIMARY, HIGHLIGHT]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, churn_counts.values, color=colors, width=0.5,
              edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, churn_counts.values):
    pct = val / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom',
            fontsize=11, fontweight='bold')
ax.set_title(
    'Churned Customers Are a Clear Minority - Imbalance Will Require Weighting',
    fontsize=12, fontweight='bold', pad=15)
ax.set_ylabel('Count', fontsize=10)
ax.set_ylim(0, max(churn_counts.values) * 1.2)
apply_style(ax)
plt.tight_layout()
plt.savefig('outputs/charts/eda/eda_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1/6 saved: eda_target_distribution.png")

# ── Chart 2: Transaction count distribution by churn ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
churned  = df[df[target_col] == 1]['total_trans_ct']
retained = df[df[target_col] == 0]['total_trans_ct']
ax.hist(retained, bins=40, color=PRIMARY,    alpha=0.7, label='Retained', density=True)
ax.hist(churned,  bins=40, color=HIGHLIGHT,  alpha=0.7, label='Churned',  density=True)
ax.axvline(retained.median(), color=PRIMARY,   linestyle='--', linewidth=2,
           label=f'Retained median: {retained.median():.0f}')
ax.axvline(churned.median(),  color=HIGHLIGHT, linestyle='--', linewidth=2,
           label=f'Churned median: {churned.median():.0f}')
ax.set_title(
    'Churned Customers Transact Far Less - Transaction Count Is a Strong Predictor',
    fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Total Transaction Count (12 months)', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.legend(fontsize=9)
apply_style(ax)
plt.tight_layout()
plt.savefig('outputs/charts/eda/eda_dist_total_trans_ct.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2/6 saved: eda_dist_total_trans_ct.png")

# ── Chart 3: Transaction amount distribution by churn ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
churned_amt  = df[df[target_col] == 1]['total_trans_amt']
retained_amt = df[df[target_col] == 0]['total_trans_amt']
ax.hist(retained_amt, bins=40, color=PRIMARY,   alpha=0.7, label='Retained', density=True)
ax.hist(churned_amt,  bins=40, color=HIGHLIGHT, alpha=0.7, label='Churned',  density=True)
ax.axvline(retained_amt.median(), color=PRIMARY,   linestyle='--', linewidth=2,
           label=f'Retained median: ${retained_amt.median():,.0f}')
ax.axvline(churned_amt.median(),  color=HIGHLIGHT, linestyle='--', linewidth=2,
           label=f'Churned median: ${churned_amt.median():,.0f}')
ax.set_title(
    'Churned Customers Spend Less - Lower Transaction Amounts Signal Disengagement',
    fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Total Transaction Amount ($)', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.legend(fontsize=9)
apply_style(ax)
plt.tight_layout()
plt.savefig('outputs/charts/eda/eda_dist_total_trans_amt.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3/6 saved: eda_dist_total_trans_amt.png")

# ── Chart 4: Inactivity violin plot ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
data_plot = [
    df[df[target_col] == 0]['months_inactive_12_mon'].values,
    df[df[target_col] == 1]['months_inactive_12_mon'].values
]
parts = ax.violinplot(data_plot, positions=[0, 1], showmedians=True, showextrema=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor([PRIMARY, HIGHLIGHT][i])
    pc.set_alpha(0.7)
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(2)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Retained', 'Churned'], fontsize=11)
ax.set_title(
    'Churned Customers Are More Inactive - Inactivity Months Separates the Groups',
    fontsize=12, fontweight='bold', pad=15)
ax.set_ylabel('Months Inactive (12 months)', fontsize=10)
apply_style(ax)
plt.tight_layout()
plt.savefig('outputs/charts/eda/eda_violin_months_inactive.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 4/6 saved: eda_violin_months_inactive.png")

# ── Chart 5: Correlation heatmap ──────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

corr_with_target = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
top_features     = corr_with_target.head(12).index.tolist()
cols_heatmap     = top_features + [target_col]
corr_matrix      = df[cols_heatmap].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(corr_matrix, ax=ax, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot_kws={'size': 8}, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
ax.set_title(
    'Transaction Behaviour Features Show Strongest Correlation with Churn',
    fontsize=12, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('outputs/charts/eda/eda_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 5/6 saved: eda_correlation_heatmap.png")

# ── Chart 6: Top feature correlations bar ─────────────────────────────────────
top10_corr = corr_with_target.head(10)
fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = [HIGHLIGHT if v > 0.3 else PRIMARY for v in top10_corr.values]
bars = ax.barh(top10_corr.index[::-1], top10_corr.values[::-1],
               color=bar_colors[::-1], edgecolor='white')
for bar, val in zip(bars, top10_corr.values[::-1]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
ax.axvline(0.3, color=HIGHLIGHT, linestyle='--', alpha=0.5, label='Strong (>0.30)')
ax.set_title(
    'Transaction Count and Amount Are the Strongest Churn Predictors',
    fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Absolute Correlation with Churn', fontsize=10)
ax.legend(fontsize=9)
apply_style(ax)
plt.tight_layout()
plt.savefig('outputs/charts/eda/eda_feature_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 6/6 saved: eda_feature_correlations.png")

# ── Key Findings ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EDA KEY FINDINGS")
print("="*60)

churn_rate = df[target_col].mean() * 100
print(f"\n1. TARGET BALANCE")
print(f"   Churn rate: {churn_rate:.1f}%  |  Retained: {100-churn_rate:.1f}%")
print(f"   --> Class imbalance confirmed. Use scale_pos_weight or class_weight.")

print(f"\n2. TOP PREDICTORS (|correlation| with churn_flag)")
for feat, val in corr_with_target.head(5).items():
    direction = "NEG" if df[feat].corr(df[target_col]) < 0 else "POS"
    print(f"   {feat:<38} r={val:.3f}  ({direction})")

retained_trans = df[df[target_col]==0]['total_trans_ct'].median()
churned_trans  = df[df[target_col]==1]['total_trans_ct'].median()
print(f"\n3. TRANSACTION BEHAVIOUR SEPARATION")
print(f"   Retained median trans count : {retained_trans:.0f}")
print(f"   Churned  median trans count : {churned_trans:.0f}")
print(f"   Gap: {retained_trans - churned_trans:.0f} fewer transactions for churned customers")

retained_inact = df[df[target_col]==0]['months_inactive_12_mon'].median()
churned_inact  = df[df[target_col]==1]['months_inactive_12_mon'].median()
print(f"\n4. INACTIVITY SEPARATION")
print(f"   Retained median inactive months : {retained_inact:.1f}")
print(f"   Churned  median inactive months : {churned_inact:.1f}")

high_corr_pairs = []
for i, c1 in enumerate(numeric_cols):
    for c2 in numeric_cols[i+1:]:
        r = abs(df[[c1, c2]].corr().iloc[0, 1])
        if r > 0.85:
            high_corr_pairs.append((c1, c2, r))
print(f"\n5. MULTICOLLINEARITY FLAGS (r > 0.85)")
if high_corr_pairs:
    for c1, c2, r in sorted(high_corr_pairs, key=lambda x: -x[2])[:5]:
        print(f"   {c1} <-> {c2}: r={r:.3f}")
else:
    print("   None detected.")

print(f"\n6. ENGINEERING RECOMMENDATIONS")
print(f"   - Create: total_trans_amt / total_trans_ct = avg_transaction_value")
print(f"   - Create: total_amt_chng_q4_q1 * total_ct_chng_q4_q1 = engagement_velocity")
print(f"   - Create: months_inactive_12_mon * contacts_count_12_mon = re_engagement_signal")
print(f"   - Credit utilization already present (avg_utilization_ratio)")

print("\n" + "="*60)
print("Stage 3 EDA COMPLETE -- 6 charts saved to outputs/charts/eda/")
print("="*60)
