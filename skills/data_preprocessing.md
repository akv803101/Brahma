# Skill: Data Preprocessing

## Purpose
Transform raw loaded data into a clean, encoded, scaled dataset ready for model training.
Follows a strict 8-step sequence. Every action is logged. Output saved to parquet.

---

## Core Rule — Never Silently Impute

**Every transformation that modifies data values must print:**
1. **What** — which column, how many values, what they were changed to
2. **Why** — which threshold triggered the decision, why that method was chosen
3. **Risk** — what model impact to watch for

This rule applies to imputation, winsorization, type casting, encoding, and scaling.
Silent transformations corrupt models silently. Brahma never operates in the dark.

---

## Step 1 — Duplicate Removal

```python
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after

    print("=" * 60)
    print("  STEP 1: DUPLICATE REMOVAL")
    print("=" * 60)
    print(f"  Rows before : {before:,}")
    print(f"  Rows after  : {after:,}")
    print(f"  Removed     : {removed:,} duplicate rows")
    print("  Status      : " + ("No duplicates found." if removed == 0 else f"REMOVED {removed:,} rows."))

    return df
```

---

## Step 2 — Missing Value Treatment

Rules applied per column based on null percentage:

| Null %    | Action                                              |
|-----------|-----------------------------------------------------|
| 0%        | No action                                           |
| 1–5%      | Impute (median for numeric, mode for categorical)   |
| 5–20%     | Impute + add binary `{col}_was_missing` indicator   |
| 20–50%    | Flag + Impute + add binary indicator                |
| >50%      | DROP column, log reason                             |

```python
import pandas as pd
import numpy as np

def _explain_fill(col: str, fill_val, is_numeric: bool, null_count: int, null_pct: float) -> str:
    """Build a human-readable explanation of why this fill value was chosen."""
    method = "median" if is_numeric else "mode"
    reason = (
        "median chosen — robust to outliers; mean would be skewed by extremes"
        if is_numeric else
        f"mode chosen — preserves most frequent category ('{fill_val}'); "
        "mean is undefined for categoricals"
    )
    return (
        f"    What  : {null_count:,} of {null_count + (null_count // max(null_pct, 1e-9) - null_count):.0f} "
        f"values were NaN ({null_pct*100:.1f}%)\n"
        f"    Method: {method} → fill value = {fill_val}\n"
        f"    Why   : {reason}"
    )


def treat_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    CORE RULE: Never silently impute.
    Every decision prints: column name, null %, threshold triggered,
    fill method, fill value, reason for that method, and model risk.
    """
    log = []

    print("\n" + "=" * 60)
    print("  STEP 2: MISSING VALUE TREATMENT")
    print("  Rule: Every imputation is explained. Nothing is silent.")
    print("=" * 60)

    cols_to_drop = []
    total_rows = len(df)

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct   = null_count / total_rows

        if null_pct == 0:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        # ── Tier 5: >50% → DROP ─────────────────────────────────────────────
        if null_pct > 0.50:
            cols_to_drop.append(col)
            log.append({
                'column': col, 'null_pct': round(null_pct * 100, 1),
                'action': 'DROPPED',
                'reason': f"{null_pct*100:.1f}% missing — imputing fabricates more data than exists"
            })
            print(f"\n  [DROP] {col}")
            print(f"    What  : {null_count:,} / {total_rows:,} values missing ({null_pct*100:.1f}%)")
            print(f"    Rule  : >50% threshold triggered")
            print(f"    Why   : Imputing over half a column invents data — the column tells us almost")
            print(f"             nothing real. Keeping it would inject fabricated signal into the model.")
            print(f"    Action: Column REMOVED entirely.")
            print(f"    Risk  : If this column was important, reconsider collection strategy upstream.")

        # ── Tier 4: 20–50% → Flag + Impute + Indicator ──────────────────────
        elif null_pct > 0.20:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)
            fill_val = df[col].median() if is_numeric else df[col].mode()[0]
            df[col]  = df[col].fillna(fill_val)
            log.append({
                'column': col, 'null_pct': round(null_pct * 100, 1),
                'action': 'FLAGGED + IMPUTED + INDICATOR',
                'fill_value': str(fill_val),
                'reason': '20–50% threshold — high missingness, imputation carries bias risk'
            })
            print(f"\n  [FLAG + IMPUTE + INDICATOR] {col}")
            print(f"    What  : {null_count:,} / {total_rows:,} values missing ({null_pct*100:.1f}%)")
            print(f"    Rule  : 20–50% threshold triggered")
            print(f"    Why   : Missingness is high enough that the pattern of *who* is missing")
            print(f"             may itself be predictive (e.g. income not reported → high risk).")
            print(f"             Imputing alone would destroy this signal.")
            print(f"    Action: (1) Created '{col}_was_missing' binary indicator (1=was null, 0=observed)")
            method = "median" if is_numeric else "mode"
            reason = "robust to outliers" if is_numeric else "most frequent category"
            print(f"             (2) Imputed with {method} = {fill_val}  [{reason}]")
            print(f"    Risk  : {null_pct*100:.1f}% imputation. Treat this column's coefficients with caution.")

        # ── Tier 3: 5–20% → Impute + Indicator ──────────────────────────────
        elif null_pct > 0.05:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)
            fill_val = df[col].median() if is_numeric else df[col].mode()[0]
            df[col]  = df[col].fillna(fill_val)
            log.append({
                'column': col, 'null_pct': round(null_pct * 100, 1),
                'action': 'IMPUTED + INDICATOR',
                'fill_value': str(fill_val),
                'reason': '5–20% threshold — missingness may be informative, indicator preserves signal'
            })
            print(f"\n  [IMPUTE + INDICATOR] {col}")
            print(f"    What  : {null_count:,} / {total_rows:,} values missing ({null_pct*100:.1f}%)")
            print(f"    Rule  : 5–20% threshold triggered")
            print(f"    Why   : Enough missingness that the absence pattern could carry predictive signal.")
            print(f"             Indicator column lets the model learn whether 'was missing' matters.")
            print(f"    Action: (1) Created '{col}_was_missing' binary indicator")
            method = "median" if is_numeric else "mode"
            reason = "robust to outliers" if is_numeric else "most frequent category"
            print(f"             (2) Imputed with {method} = {fill_val}  [{reason}]")
            print(f"    Risk  : Low-to-moderate. Monitor feature importance of '{col}_was_missing'.")

        # ── Tier 2: 1–5% → Impute only ───────────────────────────────────────
        elif null_pct > 0.01:
            fill_val = df[col].median() if is_numeric else df[col].mode()[0]
            df[col]  = df[col].fillna(fill_val)
            log.append({
                'column': col, 'null_pct': round(null_pct * 100, 1),
                'action': 'IMPUTED',
                'fill_value': str(fill_val),
                'reason': '1–5% threshold — missingness too sparse to carry signal, safe to impute'
            })
            print(f"\n  [IMPUTE] {col}")
            print(f"    What  : {null_count:,} / {total_rows:,} values missing ({null_pct*100:.1f}%)")
            print(f"    Rule  : 1–5% threshold triggered")
            print(f"    Why   : Sparse missingness — pattern is unlikely to be systematic.")
            print(f"             Indicator column would add noise, not signal, at this rate.")
            method = "median" if is_numeric else "mode"
            reason = "robust to outliers" if is_numeric else "most frequent category"
            print(f"    Action: Imputed with {method} = {fill_val}  [{reason}]")
            print(f"    Risk  : Minimal. {null_count:,} values changed.")

        # ── Tier 1: <1% → Trace impute ───────────────────────────────────────
        else:
            fill_val = df[col].median() if is_numeric else df[col].mode()[0]
            df[col]  = df[col].fillna(fill_val)
            log.append({
                'column': col, 'null_pct': round(null_pct * 100, 2),
                'action': 'IMPUTED (trace)',
                'fill_value': str(fill_val),
                'reason': '<1% threshold — trace nulls, negligible impact'
            })
            print(f"\n  [IMPUTE — TRACE] {col}")
            print(f"    What  : {null_count:,} / {total_rows:,} values missing ({null_pct*100:.2f}%)")
            print(f"    Rule  : <1% threshold — trace nulls")
            method = "median" if is_numeric else "mode"
            print(f"    Action: Imputed with {method} = {fill_val}")
            print(f"    Risk  : Negligible — {null_count:,} row(s) affected.")

    df = df.drop(columns=cols_to_drop)

    dropped_count  = len(cols_to_drop)
    imputed_count  = len(log) - dropped_count
    print("\n" + "-" * 60)
    print(f"  STEP 2 SUMMARY")
    print(f"    Columns dropped : {dropped_count}  {cols_to_drop if cols_to_drop else ''}")
    print(f"    Columns imputed : {imputed_count}")
    print(f"    Columns clean   : {sum(1 for c in df.columns if df[c].isna().sum() == 0)}")
    print("-" * 60)

    return df, log
```

---

## Step 3 — Outlier Detection + Winsorization (IQR × 3)

Winsorize numeric columns: clip values outside `[Q1 - 3×IQR, Q3 + 3×IQR]`.
IQR × 3 is intentionally wide — catches only extreme outliers, preserves real variance.

```python
def winsorize_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    CORE RULE: Never silently clip.
    Every winsorized column prints: bounds, rows affected, and why IQR×3 was chosen.
    """
    log = []

    print("\n" + "=" * 60)
    print("  STEP 3: OUTLIER DETECTION + WINSORIZATION (IQR × 3)")
    print("  Rule: IQR×3 is intentionally wide — only catches extreme outliers,")
    print("        preserves real variance. Values are clipped, not deleted.")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            print(f"\n  [SKIP] {col}")
            print(f"    Why   : IQR = 0 — column has no spread (constant or near-constant).")
            print(f"             Winsorization would be meaningless. No change made.")
            continue

        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR

        n_low  = int((df[col] < lower).sum())
        n_high = int((df[col] > upper).sum())
        total  = n_low + n_high

        df[col] = df[col].clip(lower=lower, upper=upper)

        if total > 0:
            log.append({
                'column': col, 'lower_bound': round(lower, 4),
                'upper_bound': round(upper, 4),
                'clipped_low': n_low, 'clipped_high': n_high
            })
            print(f"\n  [CLIP] {col}")
            print(f"    What  : {total:,} extreme value(s) clipped  "
                  f"({n_low} below lower bound, {n_high} above upper bound)")
            print(f"    Bounds: [{lower:.4f},  {upper:.4f}]  (Q1±3×IQR)")
            print(f"    Why   : IQR×3 threshold — values this far from the bulk are")
            print(f"             almost certainly data errors or extreme edge cases.")
            print(f"             Clipped to boundary (not removed) to preserve row count.")
            print(f"    Risk  : If these extremes are real (e.g. fraud amounts), review manually.")
        else:
            print(f"\n  [OK]   {col}")
            print(f"    What  : No values outside [{lower:.4f}, {upper:.4f}]")
            print(f"    Why   : All values within IQR×3 — no winsorization needed.")

    print("\n" + "-" * 60)
    print(f"  STEP 3 SUMMARY: {len(log)} column(s) winsorized.")
    print("-" * 60)

    return df, log
```

---

## Step 4 — Data Type Correction

Detects and converts:
- Date-like object columns → `datetime64`
- Boolean-like object columns (`yes/no`, `true/false`, `1/0`) → `bool`
- Numeric strings stored as object → `float64` or `int64`

```python
import re

def correct_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    log = []

    print("\n" + "=" * 60)
    print("  STEP 4: DATA TYPE CORRECTION")
    print("=" * 60)

    bool_map = {
        'yes': True, 'no': False,
        'true': True, 'false': False,
        '1': True, '0': False,
        'y': True, 'n': False,
        't': True, 'f': False,
    }
    date_pattern = re.compile(r'date|time|dt|year|month|created|updated|timestamp', re.IGNORECASE)

    for col in df.select_dtypes(include='object').columns:
        sample = df[col].dropna().astype(str).str.strip().str.lower()

        # Boolean check
        unique_vals = set(sample.unique())
        if unique_vals <= set(bool_map.keys()):
            df[col] = df[col].astype(str).str.strip().str.lower().map(bool_map)
            log.append({'column': col, 'from': 'object', 'to': 'bool'})
            print(f"  BOOL   {col:<35} object → bool")
            continue

        # Date check (name hint + parseable)
        if date_pattern.search(col):
            try:
                df[col] = pd.to_datetime(df[col], infer_format=True, errors='raise')
                log.append({'column': col, 'from': 'object', 'to': 'datetime64'})
                print(f"  DATE   {col:<35} object → datetime64")
                continue
            except Exception:
                pass

        # Numeric string check
        try:
            converted = pd.to_numeric(df[col], errors='raise')
            if converted.dtype == float and (converted % 1 == 0).all():
                df[col] = converted.astype('Int64')
                log.append({'column': col, 'from': 'object', 'to': 'Int64'})
                print(f"  INT    {col:<35} object → Int64")
            else:
                df[col] = converted
                log.append({'column': col, 'from': 'object', 'to': 'float64'})
                print(f"  FLOAT  {col:<35} object → float64")
        except (ValueError, TypeError):
            print(f"  OK     {col:<35} kept as object (categorical)")

    if not log:
        print("  No type corrections needed.")
    else:
        print(f"\n  Summary: {len(log)} column(s) recast.")

    return df, log
```

---

## Step 5 — Encoding

| Cardinality / Type | Strategy          |
|--------------------|-------------------|
| Binary (2 values)  | Label encode (0/1)|
| Nominal ≤ 10 cats  | One-hot encode    |
| Nominal > 10 cats  | Frequency encode  |
| Ordinal            | Ordinal encode    |

Pass `ordinal_cols` dict: `{col_name: [ordered_categories]}` to specify ordinal columns.

```python
def encode_categoricals(
    df: pd.DataFrame,
    ordinal_cols: dict = None,       # e.g. {"size": ["S","M","L","XL"]}
    target_col: str = None           # exclude target from encoding
) -> tuple[pd.DataFrame, dict]:

    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    ordinal_cols = ordinal_cols or {}
    encoding_map = {}

    print("\n" + "=" * 60)
    print("  STEP 5: CATEGORICAL ENCODING")
    print("=" * 60)

    cat_cols = [
        c for c in df.select_dtypes(include=['object', 'category']).columns
        if c != target_col
    ]

    for col in cat_cols:
        n_unique = df[col].nunique()

        # Ordinal
        if col in ordinal_cols:
            categories = ordinal_cols[col]
            oe = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1)
            df[col] = oe.fit_transform(df[[col]])
            encoding_map[col] = {'strategy': 'ordinal', 'categories': categories}
            print(f"\n  [ORDINAL] {col}")
            print(f"    What  : {n_unique} categories → integer ranks")
            print(f"    Order : {categories}")
            print(f"    Why   : Column has a known natural order. Ordinal encoding preserves")
            print(f"             rank distance so the model sees Low < Medium < High correctly.")
            print(f"             One-hot would lose the ordering; label encode would assign arbitrary ranks.")

        # Binary
        elif n_unique == 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoding_map[col] = {'strategy': 'label', 'classes': list(le.classes_)}
            print(f"\n  [LABEL / BINARY] {col}")
            print(f"    What  : 2 categories → 0 / 1    mapping: {list(le.classes_)}")
            print(f"    Why   : Exactly 2 values — label encode is lossless and adds no extra columns.")
            print(f"             One-hot on a binary column creates a redundant mirror column.")

        # Nominal ≤ 10 → One-hot
        elif n_unique <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            encoding_map[col] = {'strategy': 'one_hot', 'columns': list(dummies.columns)}
            print(f"\n  [ONE-HOT] {col}")
            print(f"    What  : {n_unique} categories → {len(dummies.columns)} binary columns")
            print(f"    Cols  : {list(dummies.columns)}")
            print(f"    Why   : ≤10 categories — one-hot is safe. No ordinal relationship assumed.")
            print(f"             Each category gets its own column; model learns independently per category.")
            print(f"    Risk  : +{len(dummies.columns)} columns added to feature space.")

        # Nominal > 10 → Frequency encode
        else:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
            encoding_map[col] = {'strategy': 'frequency', 'n_unique': n_unique}
            print(f"\n  [FREQUENCY ENCODE] {col}")
            print(f"    What  : {n_unique} categories → frequency ratio (0.0–1.0)")
            print(f"    Why   : >10 categories — one-hot would add {n_unique} columns (dimensionality explosion).")
            print(f"             Frequency encoding compresses cardinality into one column while")
            print(f"             preserving relative importance (rare vs. common categories).")
            print(f"    Risk  : Two different categories with the same frequency get the same value.")
            print(f"             Acceptable trade-off at high cardinality.")

    print("\n" + "-" * 60)
    print(f"  STEP 5 SUMMARY: {len(cat_cols)} categorical column(s) encoded.")
    print("-" * 60)

    return df, encoding_map
```

---

## Step 6 — Scaling (StandardScaler default)

Scales all numeric columns except binary (0/1) indicator columns and the target.

```python
from sklearn.preprocessing import StandardScaler
import joblib
import os

def scale_features(
    df: pd.DataFrame,
    target_col: str = None,
    scaler_path: str = "outputs/data/scaler.pkl"
) -> tuple[pd.DataFrame, object]:

    print("\n" + "=" * 60)
    print("  STEP 6: FEATURE SCALING (StandardScaler)")
    print("=" * 60)

    # Identify columns to scale: numeric, not binary indicator, not target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skip_cols = []
    if target_col and target_col in numeric_cols:
        skip_cols.append(target_col)

    # Skip binary (0/1 only) indicator columns
    binary_cols = [
        c for c in numeric_cols
        if set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})
    ]
    skip_cols += binary_cols

    cols_to_scale = [c for c in numeric_cols if c not in skip_cols]

    if not cols_to_scale:
        print("  No numeric columns to scale.")
        return df, None

    # Capture pre-scale stats for the audit log
    pre_stats = df[cols_to_scale].agg(['mean', 'std']).T

    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    print(f"\n  Why StandardScaler:")
    print(f"    StandardScaler subtracts mean and divides by std → each feature has mean=0, std=1.")
    print(f"    Without this, features with large ranges (e.g. income $0–$500k) dominate features")
    print(f"    with small ranges (e.g. age 18–90), causing gradient-based models to learn poorly.")
    print(f"    Binary (0/1) columns and the target are excluded — scaling them would corrupt meaning.")
    print(f"\n  Columns scaled ({len(cols_to_scale)}):")
    for c in cols_to_scale:
        print(f"    → {c:<35}  was: mean={pre_stats.loc[c,'mean']:.3f}, std={pre_stats.loc[c,'std']:.3f}  →  now: mean≈0, std≈1")
    if skip_cols:
        print(f"\n  Columns skipped:")
        for c in skip_cols:
            reason = "target column" if c == target_col else "binary indicator (0/1)"
            print(f"    ✗ {c:<35}  [{reason}]")
    print(f"\n  Scaler saved → {scaler_path}  (required to inverse-transform predictions)")

    return df, scaler
```

---

## Step 7 — Class Imbalance Check

Flags if any class in the target column represents < 20% of total samples.

```python
def check_class_imbalance(df: pd.DataFrame, target_col: str) -> dict:
    print("\n" + "=" * 60)
    print("  STEP 7: CLASS IMBALANCE CHECK")
    print("=" * 60)

    if target_col not in df.columns:
        print(f"  WARNING: Target column '{target_col}' not found. Skipping.")
        return {}

    counts = df[target_col].value_counts()
    proportions = df[target_col].value_counts(normalize=True)
    imbalanced = {}

    for cls, prop in proportions.items():
        flag = prop < 0.20
        status = "⚠ IMBALANCED" if flag else "OK"
        print(f"  Class {str(cls):<20} {counts[cls]:>6,} rows  {prop*100:.1f}%  {status}")
        if flag:
            imbalanced[cls] = round(prop * 100, 2)

    if imbalanced:
        print(f"\n  ⚠ IMBALANCE DETECTED: {list(imbalanced.keys())} below 20% threshold.")
        print("  Recommendation: consider SMOTE, class_weight='balanced', or stratified sampling.")
    else:
        print("\n  Classes are balanced (all ≥ 20%).")

    return imbalanced
```

---

## Step 8 — Save Preprocessed Data

```python
def save_preprocessed(df: pd.DataFrame, output_path: str = "outputs/data/preprocessed.parquet"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine='pyarrow')
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n  Saved preprocessed data → {output_path}  ({size_kb:.1f} KB)")
```

---

## PREPROCESSING COMPLETE Report

```python
import datetime

def print_preprocessing_report(
    df_raw: pd.DataFrame,
    df_processed: pd.DataFrame,
    missing_log: list,
    outlier_log: list,
    dtype_log: list,
    encoding_map: dict,
    imbalanced: dict,
    output_path: str
):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 60)
    print("  ✅  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Timestamp          : {ts}")
    print(f"  Rows  (raw)        : {len(df_raw):,}")
    print(f"  Rows  (clean)      : {len(df_processed):,}")
    print(f"  Cols  (raw)        : {df_raw.shape[1]:,}")
    print(f"  Cols  (clean)      : {df_processed.shape[1]:,}")
    print(f"  Duplicates removed : {len(df_raw) - len(df_processed) + (df_processed.shape[1] - df_raw.shape[1]):,}")
    dropped = [e for e in missing_log if e['action'] == 'DROPPED']
    print(f"  Columns dropped    : {len(dropped)}")
    print(f"  Columns imputed    : {len(missing_log) - len(dropped)}")
    print(f"  Outlier cols clipped: {len(outlier_log)}")
    print(f"  Type corrections   : {len(dtype_log)}")
    print(f"  Encodings applied  : {len(encoding_map)}")
    print(f"  Imbalanced classes : {list(imbalanced.keys()) if imbalanced else 'None'}")
    print(f"  Output path        : {output_path}")
    print("=" * 60)
    print("  Brahma is ready for EDA.\n")
```

---

## Master Orchestrator

```python
import pandas as pd
import numpy as np

def run_preprocessing(
    df: pd.DataFrame,
    target_col: str = None,
    ordinal_cols: dict = None,
    output_path: str = "outputs/data/preprocessed.parquet",
    scaler_path: str = "outputs/data/scaler.pkl"
) -> pd.DataFrame:

    df_raw = df.copy()

    # Step 1: Duplicate removal
    df = remove_duplicates(df)

    # Step 2: Missing value treatment
    df, missing_log = treat_missing_values(df)

    # Step 3: Outlier detection + Winsorization
    df, outlier_log = winsorize_outliers(df)

    # Step 4: Data type correction
    df, dtype_log = correct_dtypes(df)

    # Step 5: Encoding
    df, encoding_map = encode_categoricals(df, ordinal_cols=ordinal_cols, target_col=target_col)

    # Step 6: Scaling
    df, scaler = scale_features(df, target_col=target_col, scaler_path=scaler_path)

    # Step 7: Class imbalance check
    imbalanced = {}
    if target_col:
        imbalanced = check_class_imbalance(df, target_col)

    # Step 8: Save
    save_preprocessed(df, output_path)

    # Final report
    print_preprocessing_report(
        df_raw, df, missing_log, outlier_log,
        dtype_log, encoding_map, imbalanced, output_path
    )

    return df
```

---

## Usage Examples

```python
# Minimal — no target column
df_clean = run_preprocessing(df)

# Classification task with target
df_clean = run_preprocessing(
    df,
    target_col="churn",
    output_path="outputs/data/preprocessed.parquet"
)

# With ordinal columns
df_clean = run_preprocessing(
    df,
    target_col="loan_default",
    ordinal_cols={
        "education": ["High School", "Bachelor", "Master", "PhD"],
        "risk_rating": ["Low", "Medium", "High", "Very High"]
    }
)
```
