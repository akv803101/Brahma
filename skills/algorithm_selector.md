# Skill: Algorithm Selector

## Purpose
Brahma's decision-making mind. Classifies the problem, profiles the data,
applies the selection matrix, and explains every choice in plain English.
No algorithm is selected silently. Every decision gets a 3-sentence justification.

---

## Step 1 — Classify Problem Type

```python
import pandas as pd
import numpy as np

def classify_problem_type(df: pd.DataFrame, target_col: str) -> dict:
    """
    Infer problem type from the target column.
    Returns a dict with: problem_type, n_classes, target_dtype, is_ordinal_hint
    """
    print("\n" + "=" * 60)
    print("  STEP 1: PROBLEM TYPE CLASSIFICATION")
    print("=" * 60)

    y = df[target_col].dropna()
    n_unique  = y.nunique()
    dtype     = y.dtype
    is_numeric = pd.api.types.is_numeric_dtype(y)

    # Detect ordinal hint from column name
    ordinal_keywords = ["rating", "grade", "level", "rank", "score", "tier", "stage"]
    is_ordinal_hint  = any(kw in target_col.lower() for kw in ordinal_keywords)

    if not is_numeric:
        if n_unique == 2:
            problem_type = "binary_classification"
        else:
            problem_type = "multiclass_classification"
    else:
        if n_unique == 2:
            problem_type = "binary_classification"
        elif n_unique <= 20 and y.apply(float.is_integer).all():
            if is_ordinal_hint:
                problem_type = "ordinal_classification"
            else:
                problem_type = "multiclass_classification"
        else:
            problem_type = "regression"

    result = {
        "problem_type":  problem_type,
        "n_classes":     n_unique if "classification" in problem_type else None,
        "target_dtype":  str(dtype),
        "is_ordinal":    is_ordinal_hint,
    }

    print(f"  Target column   : {target_col}")
    print(f"  Target dtype    : {dtype}")
    print(f"  Unique values   : {n_unique}")
    print(f"  Problem type    : {problem_type.upper().replace('_', ' ')}")
    if is_ordinal_hint:
        print(f"  Ordinal hint    : Column name suggests ordered categories")

    return result
```

---

## Step 2 — Profile the Data

```python
def profile_data(df: pd.DataFrame, target_col: str, problem_info: dict) -> dict:
    """
    Compute dataset characteristics that drive algorithm selection:
    N rows, P features, class balance, feature types, sparsity.
    """
    print("\n" + "=" * 60)
    print("  STEP 2: DATA PROFILE")
    print("=" * 60)

    N = len(df)
    P = df.shape[1] - 1   # exclude target

    numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Class balance (classification only)
    imbalance_ratio = None
    minority_pct    = None
    if "classification" in problem_info["problem_type"]:
        counts       = df[target_col].value_counts()
        imbalance_ratio = counts.max() / counts.min()
        minority_pct    = counts.min() / len(df) * 100

    # Sparsity: % of zeros in numeric features
    sparsity = (df[numeric_cols] == 0).sum().sum() / max(df[numeric_cols].size, 1) * 100

    # Missing % overall
    missing_pct = df.drop(columns=[target_col]).isna().mean().mean() * 100

    profile = {
        "N":                N,
        "P":                P,
        "numeric_features": len(numeric_cols),
        "cat_features":     len(cat_cols),
        "imbalance_ratio":  imbalance_ratio,
        "minority_pct":     minority_pct,
        "sparsity_pct":     round(sparsity, 1),
        "missing_pct":      round(missing_pct, 1),
    }

    print(f"  Rows (N)            : {N:,}")
    print(f"  Features (P)        : {P}")
    print(f"  Numeric features    : {len(numeric_cols)}")
    print(f"  Categorical features: {len(cat_cols)}")
    if imbalance_ratio is not None:
        print(f"  Imbalance ratio     : {imbalance_ratio:.1f}:1  "
              f"(minority = {minority_pct:.1f}%)")
        if minority_pct < 20:
            print(f"  ⚠ HIGH IMBALANCE — class_weight or resampling required")
    print(f"  Sparsity            : {sparsity:.1f}% zeros in numeric features")
    print(f"  Missing overall     : {missing_pct:.1f}%")

    return profile
```

---

## Step 3 — Selection Matrix

```python
SELECTION_MATRIX = {

    # ── BINARY CLASSIFICATION ────────────────────────────────────────────────
    "binary_classification": [
        {
            "condition": lambda p: p["N"] < 10_000 and p["minority_pct"] >= 20,
            "algorithm": "LogisticRegression",
            "label":     "Logistic Regression",
            "justification": (
                "Dataset is small (N < 10K) and classes are balanced. "
                "Logistic Regression trains fast, is fully interpretable, "
                "and serves as the essential linear baseline all nonlinear models must beat."
            ),
            "params": {"max_iter": 1000, "class_weight": "balanced", "random_state": 42},
        },
        {
            "condition": lambda p: p["N"] < 10_000 and p["minority_pct"] < 20,
            "algorithm": "LogisticRegression",
            "label":     "Logistic Regression (balanced)",
            "justification": (
                "Dataset is small with class imbalance. "
                "Logistic Regression with class_weight='balanced' adjusts the decision "
                "boundary toward the minority class without requiring resampling."
            ),
            "params": {"max_iter": 1000, "class_weight": "balanced", "random_state": 42},
        },
        {
            "condition": lambda p: 10_000 <= p["N"] <= 100_000 and p["minority_pct"] >= 20,
            "algorithm": "RandomForestClassifier",
            "label":     "Random Forest",
            "justification": (
                "Mid-size dataset with mixed feature types and balanced classes. "
                "Random Forest handles nonlinearity and mixed types natively, "
                "requires minimal preprocessing, and is robust to noisy features."
            ),
            "params": {"n_estimators": 200, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"},
        },
        {
            "condition": lambda p: p["N"] > 10_000 and p["minority_pct"] >= 20,
            "algorithm": "XGBClassifier",
            "label":     "XGBoost",
            "justification": (
                "Large dataset with balanced classes. "
                "XGBoost consistently achieves state-of-the-art accuracy on tabular data, "
                "handles missing values natively, and trains efficiently at scale."
            ),
            "params": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
                       "use_label_encoder": False, "eval_metric": "logloss",
                       "random_state": 42, "n_jobs": -1},
        },
        {
            "condition": lambda p: p["N"] > 10_000 and p["minority_pct"] < 20,
            "algorithm": "XGBClassifier",
            "label":     "XGBoost (imbalanced)",
            "justification": (
                "Large dataset with class imbalance. "
                "XGBoost with scale_pos_weight penalises minority class misclassification, "
                "making it the strongest choice when both scale and imbalance are present."
            ),
            "params": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
                       "use_label_encoder": False, "eval_metric": "aucpr",
                       "random_state": 42, "n_jobs": -1},
            "scale_pos_weight": True,   # calculated at train time
        },
    ],

    # ── MULTICLASS CLASSIFICATION ────────────────────────────────────────────
    "multiclass_classification": [
        {
            "condition": lambda p: p["N"] < 10_000,
            "algorithm": "LogisticRegression",
            "label":     "Logistic Regression (multinomial)",
            "justification": (
                "Small dataset with multiple classes. "
                "Multinomial logistic regression is interpretable and memory-efficient, "
                "appropriate when N < 10K and a linear decision boundary is plausible."
            ),
            "params": {"multi_class": "multinomial", "max_iter": 1000,
                       "class_weight": "balanced", "random_state": 42},
        },
        {
            "condition": lambda p: p["N"] >= 10_000,
            "algorithm": "LGBMClassifier",
            "label":     "LightGBM",
            "justification": (
                "Large multiclass dataset. "
                "LightGBM is faster than XGBoost at large N, uses leaf-wise tree growth "
                "for higher accuracy, and handles multiclass natively with softmax objective."
            ),
            "params": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63,
                       "class_weight": "balanced", "random_state": 42, "n_jobs": -1},
        },
    ],

    # ── ORDINAL CLASSIFICATION ───────────────────────────────────────────────
    "ordinal_classification": [
        {
            "condition": lambda p: True,
            "algorithm": "XGBClassifier",
            "label":     "XGBoost (ordinal via label encoding)",
            "justification": (
                "Target has ordered categories. "
                "XGBoost with label-encoded ordinal target captures rank relationships "
                "while handling nonlinearity — more appropriate than treating this as nominal multiclass."
            ),
            "params": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5,
                       "objective": "multi:softmax", "random_state": 42, "n_jobs": -1},
        },
    ],

    # ── REGRESSION ───────────────────────────────────────────────────────────
    "regression": [
        {
            "condition": lambda p: p["N"] < 50_000 and p["sparsity_pct"] < 50,
            "algorithm": "RandomForestRegressor",
            "label":     "Random Forest Regressor",
            "justification": (
                "Mid-size dataset with nonlinear relationships and low sparsity. "
                "Random Forest Regressor handles feature interactions and mixed types "
                "without assuming linearity or requiring feature scaling."
            ),
            "params": {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
        },
        {
            "condition": lambda p: p["N"] >= 50_000,
            "algorithm": "XGBRegressor",
            "label":     "XGBoost Regressor",
            "justification": (
                "Large regression dataset. "
                "XGBoost Regressor achieves best-in-class accuracy on tabular regression, "
                "handles missing values natively, and scales to millions of rows."
            ),
            "params": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
                       "random_state": 42, "n_jobs": -1},
        },
        {
            "condition": lambda p: p["sparsity_pct"] >= 50,
            "algorithm": "HuberRegressor",
            "label":     "Huber Regression",
            "justification": (
                "Dataset has heavy sparsity or outlier-prone target. "
                "Huber Regression is robust to outliers by down-weighting large residuals, "
                "avoiding the distortion that OLS and RMSE-minimising models suffer."
            ),
            "params": {"epsilon": 1.35, "max_iter": 300},
        },
    ],
}
```

---

## Step 4 — Always Train These Baselines

```python
BASELINES = {
    "binary_classification":    ["DummyClassifier", "LogisticRegression"],
    "multiclass_classification": ["DummyClassifier", "LogisticRegression"],
    "ordinal_classification":   ["DummyClassifier", "LogisticRegression"],
    "regression":               ["DummyRegressor",  "Ridge"],
}

BASELINE_PARAMS = {
    "DummyClassifier":  {"strategy": "most_frequent"},
    "DummyRegressor":   {"strategy": "mean"},
    "LogisticRegression": {"max_iter": 1000, "class_weight": "balanced", "random_state": 42},
    "Ridge":            {"alpha": 1.0},
}

def get_baselines(problem_type: str) -> list[str]:
    return BASELINES.get(problem_type, ["DummyClassifier", "LogisticRegression"])
```

---

## Step 5 — Select Algorithm + Print Justification

```python
def select_algorithm(profile: dict, problem_info: dict) -> dict:
    """
    Apply selection matrix to profile. Returns selected algorithm config.
    Prints 3-sentence justification for every decision.
    """
    print("\n" + "=" * 60)
    print("  STEP 3–5: ALGORITHM SELECTION")
    print("=" * 60)

    problem_type = problem_info["problem_type"]
    candidates   = SELECTION_MATRIX.get(problem_type, [])
    baselines    = get_baselines(problem_type)

    selected = None
    for candidate in candidates:
        if candidate["condition"](profile):
            selected = candidate
            break

    if selected is None:
        # Fallback
        selected = candidates[-1] if candidates else {
            "algorithm": "RandomForestClassifier",
            "label":     "Random Forest (fallback)",
            "justification": "No condition matched — defaulting to Random Forest as a robust general-purpose choice.",
            "params": {"n_estimators": 100, "random_state": 42},
        }

    print(f"\n  Problem Type  : {problem_type.upper().replace('_', ' ')}")
    print(f"  N             : {profile['N']:,} rows")
    print(f"  P             : {profile['P']} features")
    if profile["minority_pct"] is not None:
        print(f"  Imbalance     : minority = {profile['minority_pct']:.1f}%")

    print(f"\n  ─── PRIMARY ALGORITHM ───")
    print(f"  Selected      : {selected['label']}")
    print(f"\n  Justification:")
    for line in selected["justification"].split(". "):
        if line.strip():
            print(f"    • {line.strip()}.")

    print(f"\n  ─── BASELINES (always trained) ───")
    for b in baselines:
        print(f"    → {b}  {BASELINE_PARAMS.get(b, {})}")
    print(f"\n    Why baselines matter:")
    print(f"    • DummyClassifier/Regressor sets the floor — the minimum score any model must beat.")
    print(f"    • Logistic/Ridge provides the linear baseline — the score attributable to linear signal alone.")
    print(f"    • A complex model that barely beats a linear baseline is not worth the complexity cost.")

    return {
        "problem_type":  problem_type,
        "primary":       selected,
        "baselines":     baselines,
        "baseline_params": BASELINE_PARAMS,
        "profile":       profile,
    }
```

---

## Master Orchestrator

```python
def run_algorithm_selection(df: pd.DataFrame, target_col: str) -> dict:
    """
    Full pipeline: classify problem → profile data → select algorithm → print justification.
    Returns selection dict for use by model_trainer.
    """
    problem_info = classify_problem_type(df, target_col)
    profile      = profile_data(df, target_col, problem_info)
    selection    = select_algorithm(profile, problem_info)

    print("\n" + "=" * 60)
    print("  ALGORITHM SELECTION COMPLETE")
    print(f"  Primary model  : {selection['primary']['label']}")
    print(f"  Baselines      : {selection['baselines']}")
    print(f"  Brahma is ready for model training.")
    print("=" * 60 + "\n")

    return selection
```

---

## Usage Example

```python
selection = run_algorithm_selection(df_engineered, target_col="churn")

# Inspect selection
print(selection["primary"]["algorithm"])   # e.g. 'XGBClassifier'
print(selection["primary"]["params"])      # hyperparameter starting point
print(selection["problem_type"])           # e.g. 'binary_classification'
```
