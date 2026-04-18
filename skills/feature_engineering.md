# Skill: Feature Engineering

## Purpose
Encode domain knowledge into the feature set. Every engineered feature must answer:
"What would a domain expert compute manually — and why does it predict the target?"
Generic polynomial expansion is not feature engineering. Domain reasoning is.

**Depends on:** `skills/visualization_style.md`

---

## Core Rule — Domain First, Math Second

Before writing code, ask:
1. What does a senior analyst in this domain track manually?
2. What ratios, rates, trends, or thresholds signal risk/opportunity?
3. Does the raw column tell the story, or does a derived version?

A good feature is one a domain expert would nod at.
A bad feature is one that only a laptop understands.

---

## Standard Import Block

```python
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "skills")

from visualization_style import (
    apply_brahma_style, new_figure, annotate_chart,
    save_chart, BRAHMA_COLORS, BRAHMA_PALETTE,
)
apply_brahma_style()

OUTPUT_DIR_DATA   = "outputs/data"
OUTPUT_DIR_CHARTS = "outputs/charts/eda"
os.makedirs(OUTPUT_DIR_DATA,   exist_ok=True)
os.makedirs(OUTPUT_DIR_CHARTS, exist_ok=True)
```

---

## Domain Pattern Library

### FINANCIAL / BANKING

```python
def engineer_financial_features(df: pd.DataFrame,
                                  col_map: dict) -> pd.DataFrame:
    """
    col_map keys (all optional — only computed if columns exist):
        transactions_30d  : count of transactions in last 30 days
        tenure_months     : months as customer
        monthly_balances  : list column or prefix of monthly balance cols
        last_complaint_date: date of last complaint
        credit_limit      : credit limit
        current_balance   : current outstanding balance
        reference_date    : as-of date for recency calculations (default: today)
    """
    ref_date = col_map.get("reference_date", pd.Timestamp.today())
    print("\n  [FEATURE ENGINEERING — FINANCIAL/BANKING]")

    # transaction_velocity: how active is the customer relative to their tenure?
    if col_map.get("transactions_30d") and col_map.get("tenure_months"):
        t30   = col_map["transactions_30d"]
        ten   = col_map["tenure_months"]
        if t30 in df.columns and ten in df.columns:
            df["transaction_velocity"] = (
                df[t30] / df[ten].replace(0, np.nan)
            )
            print(f"    + transaction_velocity = {t30} / {ten}")
            print(f"      Why: Active new customers behave differently from active long-tenure ones.")
            print(f"           Velocity normalises raw transaction count by relationship length.")

    # balance_volatility: coefficient of variation of monthly balance
    balance_cols = col_map.get("monthly_balance_cols", [])
    if len(balance_cols) >= 2:
        bal_df = df[balance_cols]
        df["balance_volatility"] = bal_df.std(axis=1) / bal_df.mean(axis=1).replace(0, np.nan)
        print(f"    + balance_volatility = std / mean across {len(balance_cols)} balance columns")
        print(f"      Why: High volatility signals financial instability — a strong default predictor.")

    # days_since_last_complaint
    if col_map.get("last_complaint_date") and col_map["last_complaint_date"] in df.columns:
        lc = col_map["last_complaint_date"]
        df[lc] = pd.to_datetime(df[lc], errors="coerce")
        df["days_since_last_complaint"] = (ref_date - df[lc]).dt.days
        print(f"    + days_since_last_complaint = today - {lc}")
        print(f"      Why: Recency of complaints is a leading indicator of churn and default.")

    # credit_utilization_trend (simple: current / limit)
    if col_map.get("current_balance") and col_map.get("credit_limit"):
        cb = col_map["current_balance"]
        cl = col_map["credit_limit"]
        if cb in df.columns and cl in df.columns:
            df["credit_utilization"] = df[cb] / df[cl].replace(0, np.nan)
            df["credit_utilization"] = df["credit_utilization"].clip(0, 1)
            high_util_flag = (df["credit_utilization"] > 0.80).astype(int)
            df["high_credit_utilization_flag"] = high_util_flag
            print(f"    + credit_utilization = {cb} / {cl}  (clipped 0–1)")
            print(f"    + high_credit_utilization_flag = 1 if utilization > 80%")
            print(f"      Why: Utilization >80% is the industry threshold for credit stress.")

    return df
```

---

### RETAIL / E-COMMERCE

```python
def engineer_retail_features(df: pd.DataFrame,
                               col_map: dict) -> pd.DataFrame:
    """
    col_map keys:
        order_value_col   : column with per-order revenue
        customer_id_col   : customer identifier
        order_date_col    : date of order
        reference_date    : as-of date (default: max order date)
    """
    print("\n  [FEATURE ENGINEERING — RETAIL]")
    cid  = col_map.get("customer_id_col")
    ov   = col_map.get("order_value_col")
    od   = col_map.get("order_date_col")

    if not all(c in df.columns for c in [cid, ov, od] if c):
        print("    WARNING: Required columns missing. Skipping retail features.")
        return df

    df[od] = pd.to_datetime(df[od], errors="coerce")
    ref_date = col_map.get("reference_date", df[od].max())

    rfm = df.groupby(cid).agg(
        recency          = (od,  lambda x: (ref_date - x.max()).days),
        purchase_frequency = (od, "count"),
        avg_order_value  = (ov,  "mean"),
        total_revenue    = (ov,  "sum"),
    ).reset_index()

    # CLV proxy: frequency × avg_order_value
    rfm["customer_lifetime_value"] = rfm["purchase_frequency"] * rfm["avg_order_value"]

    # RFM segments (simple quintile scoring 1–5)
    for col_r, ascending in [("recency", True), ("purchase_frequency", False), ("avg_order_value", False)]:
        label = col_r[0].upper()  # R, P, A
        rfm[f"rfm_{label}"] = pd.qcut(rfm[col_r], q=5, labels=[1,2,3,4,5],
                                       duplicates="drop").astype(float)
    rfm["rfm_score"] = rfm[["rfm_R","rfm_P","rfm_A"]].sum(axis=1)

    df = df.merge(rfm[[cid, "recency", "purchase_frequency",
                        "avg_order_value", "customer_lifetime_value",
                        "rfm_score"]], on=cid, how="left")

    print(f"    + recency               = days since last order")
    print(f"    + purchase_frequency    = total orders per customer")
    print(f"    + avg_order_value       = mean revenue per order")
    print(f"    + customer_lifetime_value = frequency × avg_order_value")
    print(f"    + rfm_score             = composite RFM quintile score (3–15)")
    print(f"      Why: RFM is the gold standard for retail segmentation and churn prediction.")

    return df
```

---

### HR / PEOPLE ANALYTICS

```python
def engineer_hr_features(df: pd.DataFrame,
                          col_map: dict) -> pd.DataFrame:
    """
    col_map keys:
        hire_date_col       : date of hire
        last_promotion_col  : date of last promotion
        salary_col          : current salary
        salary_band_min_col : salary band minimum
        salary_band_max_col : salary band maximum
        reference_date      : as-of date (default: today)
    """
    print("\n  [FEATURE ENGINEERING — HR/PEOPLE ANALYTICS]")
    ref_date = col_map.get("reference_date", pd.Timestamp.today())

    # tenure_months and tenure_bucket
    if col_map.get("hire_date_col") and col_map["hire_date_col"] in df.columns:
        hd = col_map["hire_date_col"]
        df[hd] = pd.to_datetime(df[hd], errors="coerce")
        df["tenure_months"] = ((ref_date - df[hd]).dt.days / 30.44).round(1)
        df["tenure_bucket"] = pd.cut(
            df["tenure_months"],
            bins=[0, 6, 12, 24, 60, float("inf")],
            labels=["0-6m", "6-12m", "1-2yr", "2-5yr", "5yr+"],
            right=True,
        )
        print(f"    + tenure_months   = months from {hd} to reference date")
        print(f"    + tenure_bucket   = [0-6m, 6-12m, 1-2yr, 2-5yr, 5yr+]")
        print(f"      Why: Attrition risk is highest at 6-18 months and after 5 years.")

    # months_since_last_promotion
    if col_map.get("last_promotion_col") and col_map["last_promotion_col"] in df.columns:
        lp = col_map["last_promotion_col"]
        df[lp] = pd.to_datetime(df[lp], errors="coerce")
        df["months_since_promotion"] = ((ref_date - df[lp]).dt.days / 30.44).round(1)
        if "tenure_months" in df.columns:
            df["promotion_rate"] = (
                df["months_since_promotion"] / df["tenure_months"].replace(0, np.nan)
            )
        print(f"    + months_since_promotion = recency of last promotion")
        print(f"    + promotion_rate         = months_since_promotion / tenure_months")
        print(f"      Why: Stagnation (high months_since_promotion relative to tenure) drives attrition.")

    # salary_to_band_ratio
    sc  = col_map.get("salary_col")
    sbl = col_map.get("salary_band_min_col")
    sbh = col_map.get("salary_band_max_col")
    if sc and sbl and sbh and all(c in df.columns for c in [sc, sbl, sbh]):
        band_range = df[sbh] - df[sbl]
        df["salary_to_band_ratio"] = (
            (df[sc] - df[sbl]) / band_range.replace(0, np.nan)
        ).clip(0, 1)
        print(f"    + salary_to_band_ratio = (salary - band_min) / (band_max - band_min)")
        print(f"      Why: Employees near the top of their band (ratio > 0.8) have less room to grow.")
        print(f"           Those near the bottom (< 0.2) may be underpaid — both signal flight risk.")

    return df
```

---

### HEALTHCARE

```python
def engineer_health_features(df: pd.DataFrame,
                               col_map: dict) -> pd.DataFrame:
    """
    col_map keys:
        weight_kg_col     : weight in kilograms
        height_m_col      : height in metres
        age_col           : age in years
        diagnosis_cols    : list of binary diagnosis columns (0/1)
    """
    print("\n  [FEATURE ENGINEERING — HEALTHCARE]")

    # BMI and category
    wc = col_map.get("weight_kg_col")
    hc = col_map.get("height_m_col")
    if wc and hc and wc in df.columns and hc in df.columns:
        df["bmi"] = df[wc] / (df[hc] ** 2)
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, float("inf")],
            labels=["Underweight", "Normal", "Overweight", "Obese"],
        )
        print(f"    + bmi          = {wc} / {hc}²")
        print(f"    + bmi_category = WHO classification [Underweight/Normal/Overweight/Obese]")
        print(f"      Why: BMI category is a standard clinical risk stratifier.")

    # Age group
    ac = col_map.get("age_col")
    if ac and ac in df.columns:
        df["age_group"] = pd.cut(
            df[ac],
            bins=[0, 17, 34, 49, 64, float("inf")],
            labels=["0-17", "18-34", "35-49", "50-64", "65+"],
        )
        print(f"    + age_group = [0-17, 18-34, 35-49, 50-64, 65+]")
        print(f"      Why: Disease risk and treatment response vary by age band, not linearly by year.")

    # Comorbidity count
    diag_cols = col_map.get("diagnosis_cols", [])
    valid_diag = [c for c in diag_cols if c in df.columns]
    if valid_diag:
        df["comorbidity_count"] = df[valid_diag].sum(axis=1)
        print(f"    + comorbidity_count = sum of {len(valid_diag)} diagnosis flags")
        print(f"      Why: Comorbidity count is a stronger predictor than any single diagnosis.")

    return df
```

---

## Universal Feature Patterns (All Domains)

### Date/Time Features with Cyclic Encoding

```python
def engineer_datetime_features(df: pd.DataFrame,
                                 datetime_cols: list) -> pd.DataFrame:
    """
    For each datetime column, extract components and apply cyclic sin/cos
    encoding for month, day_of_week, and hour to preserve circular continuity.

    Cyclic encoding: sin(2π × value / max_value) and cos(2π × value / max_value)
    Why: Month 12 and Month 1 are adjacent — one-hot treats them as unrelated.
         Cyclic encoding preserves this circular relationship.
    """
    print("\n  [DATE/TIME FEATURES + CYCLIC ENCODING]")

    for col in datetime_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_datetime(df[col], errors="coerce")

        # Raw components
        df[f"{col}_year"]        = df[col].dt.year
        df[f"{col}_month"]       = df[col].dt.month
        df[f"{col}_day"]         = df[col].dt.day
        df[f"{col}_day_of_week"] = df[col].dt.dayofweek    # 0=Mon, 6=Sun
        df[f"{col}_is_weekend"]  = (df[col].dt.dayofweek >= 5).astype(int)
        df[f"{col}_quarter"]     = df[col].dt.quarter

        if df[col].dt.hour.nunique() > 1:
            df[f"{col}_hour"] = df[col].dt.hour

        # Cyclic encoding for month (1–12)
        df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[f"{col}_month"] / 12)
        df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[f"{col}_month"] / 12)

        # Cyclic encoding for day_of_week (0–6)
        df[f"{col}_dow_sin"] = np.sin(2 * np.pi * df[f"{col}_day_of_week"] / 7)
        df[f"{col}_dow_cos"] = np.cos(2 * np.pi * df[f"{col}_day_of_week"] / 7)

        # Cyclic encoding for hour (0–23) if available
        if f"{col}_hour" in df.columns:
            df[f"{col}_hour_sin"] = np.sin(2 * np.pi * df[f"{col}_hour"] / 24)
            df[f"{col}_hour_cos"] = np.cos(2 * np.pi * df[f"{col}_hour"] / 24)

        print(f"    + {col}: year, month, day, day_of_week, is_weekend, quarter")
        print(f"    + {col}: month_sin/cos, dow_sin/cos  (cyclic)")
        print(f"      Why: month=12 and month=1 are adjacent; cyclic encoding")
        print(f"           preserves this. One-hot and ordinal both break it.")

    return df
```

---

### Top 3 Interaction Features (Domain-Justified Only)

```python
def engineer_interaction_features(df: pd.DataFrame,
                                    interactions: list[tuple]) -> pd.DataFrame:
    """
    Create interaction features from a list of (col_a, col_b, operation) tuples.
    Operations: 'multiply', 'divide', 'add', 'subtract', 'ratio'

    Only domain-justified interactions are created.
    Never generate all pairwise interactions — that is brute force, not engineering.

    Example:
        interactions = [
            ("tenure_months",   "transaction_velocity", "multiply"),
            ("credit_utilization", "balance_volatility", "multiply"),
            ("avg_order_value",    "purchase_frequency",  "multiply"),
        ]
    """
    print("\n  [INTERACTION FEATURES]")

    for item in interactions:
        if len(item) == 3:
            col_a, col_b, op = item
        else:
            print(f"    SKIP: malformed tuple {item}")
            continue

        if col_a not in df.columns or col_b not in df.columns:
            print(f"    SKIP: {col_a} or {col_b} not in DataFrame")
            continue

        if op == "multiply":
            new_col = f"{col_a}_x_{col_b}"
            df[new_col] = df[col_a] * df[col_b]
        elif op in ("divide", "ratio"):
            new_col = f"{col_a}_over_{col_b}"
            df[new_col] = df[col_a] / df[col_b].replace(0, np.nan)
        elif op == "add":
            new_col = f"{col_a}_plus_{col_b}"
            df[new_col] = df[col_a] + df[col_b]
        elif op == "subtract":
            new_col = f"{col_a}_minus_{col_b}"
            df[new_col] = df[col_a] - df[col_b]
        else:
            print(f"    SKIP: unknown operation '{op}'")
            continue

        print(f"    + {new_col}  [{op}]")

    print(f"      Rule: Only domain-justified interactions. No blind pairwise expansion.")

    return df
```

---

## Feature Selection

### Step 1 — Random Forest Importance (Pre-Model Signal Scan)

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.ticker as ticker

def compute_feature_importance(df: pd.DataFrame,
                                 target_col: str,
                                 problem_type: str = "classification",
                                 dataset_name: str = "dataset") -> pd.Series:
    """
    Quick Random Forest to rank features by predictive importance.
    Returns a Series of importance scores indexed by feature name.
    Drops: importances < 0.001 (noise), correlation > 0.90 (redundant),
           variance < 0.01 (near-constant).

    problem_type: 'classification' or 'regression'
    """
    print("\n" + "=" * 60)
    print("  FEATURE SELECTION")
    print("=" * 60)

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != target_col]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target_col]

    # ── Drop near-zero variance ───────────────────────────────────────────────
    from sklearn.feature_selection import VarianceThreshold
    vt      = VarianceThreshold(threshold=0.01)
    vt.fit(X)
    low_var = [c for c, keep in zip(feature_cols, vt.get_support()) if not keep]
    X       = X[vt.get_feature_names_out()]
    feature_cols = list(vt.get_feature_names_out())
    if low_var:
        print(f"\n  [DROP — LOW VARIANCE < 0.01]")
        for c in low_var:
            print(f"    ✗ {c}  — near-constant, no predictive signal")

    # ── Drop high-correlation pairs (keep higher-importance one, flag here) ──
    corr_matrix = X.corr().abs()
    upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    high_corr_pairs = [(c, r)
                       for c in upper.columns
                       for r in upper.index
                       if upper.loc[r, c] > 0.90]
    # We will drop after importance is known (keep the more important one)

    # ── Fit quick Random Forest ───────────────────────────────────────────────
    print(f"\n  Fitting Random Forest for importance ranking...")
    print(f"    Problem type : {problem_type}")
    print(f"    Features     : {len(feature_cols)}")
    print(f"    Rows         : {len(X):,}")

    if problem_type == "classification":
        model = RandomForestClassifier(n_estimators=100, max_depth=8,
                                       random_state=42, n_jobs=-1,
                                       class_weight="balanced")
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=8,
                                      random_state=42, n_jobs=-1)

    if problem_type == "classification" and y.dtype == object:
        le = LabelEncoder()
        y  = le.fit_transform(y)

    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=feature_cols) \
                   .sort_values(ascending=False)

    # ── Drop importance < 0.001 ──────────────────────────────────────────────
    noise_feats = importance[importance < 0.001].index.tolist()
    if noise_feats:
        print(f"\n  [DROP — IMPORTANCE < 0.001]")
        for c in noise_feats:
            print(f"    ✗ {c:<40}  importance={importance[c]:.5f}  — noise level")
        importance = importance[importance >= 0.001]

    # ── Drop from high-corr pairs (keep higher importance) ───────────────────
    drop_corr = set()
    for col_a, col_b in high_corr_pairs:
        if col_a in importance.index and col_b in importance.index:
            drop = col_b if importance[col_a] >= importance[col_b] else col_a
            keep = col_a if drop == col_b else col_b
            drop_corr.add(drop)
    if drop_corr:
        print(f"\n  [DROP — CORRELATION > 0.90]")
        for c in drop_corr:
            print(f"    ✗ {c:<40}  — highly correlated with a more important feature")
        importance = importance.drop(index=list(drop_corr), errors="ignore")

    print(f"\n  Features retained: {len(importance)}")
    print(f"  Features dropped : {len(low_var) + len(noise_feats) + len(drop_corr)}")

    return importance
```

---

### Step 2 — Plot Top 20 Features

```python
def plot_feature_importance(importance: pd.Series,
                              dataset_name: str = "dataset",
                              top_n: int = 20) -> str:
    """
    Horizontal bar chart of top N features by Random Forest importance.
    Top feature highlighted in primary blue; rest in muted grey.
    """
    top = importance.head(top_n).sort_values(ascending=True)

    colors = [
        BRAHMA_COLORS["primary"] if i == len(top) - 1
        else BRAHMA_COLORS["muted"]
        for i in range(len(top))
    ]

    fig, ax = new_figure(figsize=(14, max(8, len(top) * 0.55 + 2)))

    bars = ax.barh(top.index, top.values, color=colors,
                   edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + top.max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="left", fontsize=9,
                color=BRAHMA_COLORS["neutral"])

    ax.set_xlim(0, top.max() * 1.18)

    top_feat = importance.index[0]
    top_val  = importance.iloc[0]

    annotate_chart(ax,
        title=f"Top {top_n} Features by Predictive Importance (Pre-Model) — '{top_feat}' Leads",
        subtitle=(f"Random Forest importance  |  {len(importance)} features retained after selection  "
                  f"|  Source: {dataset_name}"),
        xlabel="Feature Importance (Gini)",
        ylabel="Feature",
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR_CHARTS, "feature_importance_top20.png")
    save_chart(fig, path)
    return path
```

---

## Save Engineered Data

```python
def save_engineered(df: pd.DataFrame,
                     output_path: str = "outputs/data/engineered.parquet"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n  Saved engineered data → {output_path}  ({size_kb:.1f} KB)")
```

---

## Master Orchestrator

```python
def run_feature_engineering(
    df: pd.DataFrame,
    target_col: str,
    domain: str = None,                # 'financial', 'retail', 'hr', 'health', or None
    col_map: dict = None,              # domain-specific column mapping
    datetime_cols: list = None,        # list of datetime column names
    interactions: list = None,         # list of (col_a, col_b, op) tuples
    problem_type: str = "classification",
    dataset_name: str = "dataset",
    output_path: str = "outputs/data/engineered.parquet",
) -> tuple[pd.DataFrame, pd.Series]:

    col_map       = col_map or {}
    datetime_cols = datetime_cols or []
    interactions  = interactions or []

    print("\n" + "=" * 60)
    print("  FEATURE ENGINEERING")
    print(f"  Domain  : {domain or 'generic'}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("=" * 60)

    # Domain-specific features
    if domain == "financial":
        df = engineer_financial_features(df, col_map)
    elif domain == "retail":
        df = engineer_retail_features(df, col_map)
    elif domain == "hr":
        df = engineer_hr_features(df, col_map)
    elif domain == "health":
        df = engineer_health_features(df, col_map)

    # Universal: date/time
    if datetime_cols:
        df = engineer_datetime_features(df, datetime_cols)

    # Universal: interactions
    if interactions:
        df = engineer_interaction_features(df, interactions)

    # Feature selection: importance + drop rules
    importance = compute_feature_importance(df, target_col, problem_type, dataset_name)

    # Plot top 20
    plot_feature_importance(importance, dataset_name)

    # Drop low-importance and high-correlation features from df
    keep_cols = list(importance.index) + [target_col]
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if target_col in non_numeric:
        non_numeric.remove(target_col)
    keep_cols = keep_cols + non_numeric
    df = df[[c for c in keep_cols if c in df.columns]]

    # Save
    save_engineered(df, output_path)

    print(f"\n  Final feature count : {df.shape[1] - 1} features + 1 target")
    print(f"  Brahma is ready for algorithm selection.\n")

    return df, importance
```

---

## Usage Examples

```python
# Financial domain
df, importance = run_feature_engineering(
    df,
    target_col="default",
    domain="financial",
    col_map={
        "transactions_30d": "txn_count_30d",
        "tenure_months":    "customer_tenure_months",
        "current_balance":  "outstanding_balance",
        "credit_limit":     "approved_limit",
    },
    datetime_cols=["account_open_date", "last_transaction_date"],
    interactions=[
        ("transaction_velocity", "credit_utilization", "multiply"),
        ("tenure_months", "balance_volatility", "multiply"),
    ],
    problem_type="classification",
    dataset_name="bank_customers.csv",
)

# Retail domain
df, importance = run_feature_engineering(
    df,
    target_col="churned",
    domain="retail",
    col_map={
        "customer_id_col":  "customer_id",
        "order_value_col":  "order_total",
        "order_date_col":   "order_date",
    },
    datetime_cols=["order_date", "signup_date"],
    problem_type="classification",
    dataset_name="retail_orders.parquet",
)
```
