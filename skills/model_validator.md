# Skill: Model Validator

## Purpose
Brahma does not ship memorization. This skill proves a model generalizes.
Four checks — cross-validation, overfitting table, learning curve, feature stability —
and threshold analysis for classification. Every check prints a verdict.

**Depends on:** `skills/visualization_style.md`

---

## Standard Import Block

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

OUTPUT_DIR = "outputs/charts/validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

---

## Step 1 — 10-Fold Stratified Cross-Validation

```python
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_validate,
)
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, r2_score

def run_cross_validation(model,
                          X: pd.DataFrame,
                          y: pd.Series,
                          problem_type: str,
                          n_folds: int = 10) -> dict:
    """
    10-fold stratified cross-validation.
    Reports mean ± std for all primary metrics.
    Flags unstable models (std > 0.05).
    """
    print("\n" + "=" * 60)
    print(f"  STEP 1: {n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 60)

    is_clf = "classification" in problem_type

    cv = (StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
          if is_clf else
          KFold(n_splits=n_folds, shuffle=True, random_state=42))

    if "binary" in problem_type:
        scoring = {
            "accuracy":    "accuracy",
            "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
            "roc_auc":     "roc_auc",
        }
    elif "classification" in problem_type:
        scoring = {
            "accuracy":    "accuracy",
            "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
        }
    else:
        scoring = {
            "r2":   "r2",
            "neg_rmse": "neg_root_mean_squared_error",
        }

    print(f"  Folds     : {n_folds}")
    print(f"  Strategy  : {'Stratified' if is_clf else 'KFold'}")
    print(f"  Scoring   : {list(scoring.keys())}")
    print(f"  Running cross-validation...")

    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                 return_train_score=True, n_jobs=-1)

    results = {}
    print(f"\n  {'Metric':<25} {'Train Mean':>12} {'Val Mean':>12} {'Val Std':>10} {'Verdict':>20}")
    print("  " + "-" * 82)

    for metric in scoring:
        val_key   = f"test_{metric}"
        train_key = f"train_{metric}"
        val_mean  = cv_results[val_key].mean()
        val_std   = cv_results[val_key].std()
        train_mean = cv_results[train_key].mean()

        # Handle neg metrics
        if "neg" in metric:
            val_mean   = -val_mean
            train_mean = -train_mean
            display    = metric.replace("neg_", "")
        else:
            display = metric

        verdict = "STABLE" if val_std <= 0.05 else "UNSTABLE ⚠"

        results[display] = {
            "val_mean":   round(val_mean,   4),
            "val_std":    round(val_std,    4),
            "train_mean": round(train_mean, 4),
        }

        print(f"  {display:<25} {train_mean:>12.4f} {val_mean:>12.4f} {val_std:>10.4f} {verdict:>20}")

        if val_std > 0.05:
            print(f"  ⚠ WARNING: {display} std = {val_std:.4f} > 0.05 — model is UNSTABLE across folds.")
            print(f"    This means performance varies significantly by data sample.")
            print(f"    Recommendations: increase regularization, reduce features, or collect more data.")

    print("\n" + "-" * 60)
    stable_count = sum(1 for v in results.values() if v["val_std"] <= 0.05)
    print(f"  {stable_count}/{len(results)} metrics stable (std ≤ 0.05)")

    return results
```

---

## Step 2 — Overfitting Check

```python
def check_overfitting(cv_results: dict,
                       train_score: float,
                       test_score: float,
                       metric_name: str = "primary metric") -> dict:
    """
    Apply Brahma's overfitting classification table:
      gap < 0.03   → Well-generalized
      0.03–0.08    → Slight overfit (acceptable)
      > 0.08       → Overfit (recommend regularization)
      test > train → SUSPECTED DATA LEAKAGE — STOP

    Returns verdict dict.
    """
    print("\n" + "=" * 60)
    print("  STEP 2: OVERFITTING CHECK")
    print("=" * 60)

    gap = train_score - test_score

    print(f"  Train {metric_name:<20} {train_score:.4f}")
    print(f"  Test  {metric_name:<20} {test_score:.4f}")
    print(f"  Gap   (train - test)        {gap:.4f}")
    print()

    if gap < 0 and abs(gap) > 0.01:
        verdict = "SUSPECTED DATA LEAKAGE"
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  🚨 SUSPECTED DATA LEAKAGE — STOP                   │")
        print("  │  Test score > Train score by a meaningful margin.   │")
        print("  │  This is statistically impossible under fair splits. │")
        print("  │                                                      │")
        print("  │  Investigate immediately:                            │")
        print("  │  1. Is the target derived from any feature?          │")
        print("  │  2. Was preprocessing fit on the full dataset        │")
        print("  │     before splitting? (imputer, scaler, encoder)     │")
        print("  │  3. Are there timestamp/ID columns that leak future? │")
        print("  └─────────────────────────────────────────────────────┘")
    elif gap < 0.03:
        verdict = "WELL-GENERALIZED"
        print(f"  VERDICT: WELL-GENERALIZED ✓")
        print(f"  Gap < 0.03 — model performance on unseen data matches training.")
        print(f"  Safe to proceed to production evaluation.")
    elif gap <= 0.08:
        verdict = "SLIGHT OVERFIT (ACCEPTABLE)"
        print(f"  VERDICT: SLIGHT OVERFIT — ACCEPTABLE")
        print(f"  Gap = {gap:.4f} (0.03–0.08 range).")
        print(f"  Minor overfitting — model has learned some noise but generalizes adequately.")
        print(f"  Consider: mild regularization increase, or proceed with monitoring.")
    else:
        verdict = "OVERFIT"
        print(f"  VERDICT: OVERFIT ⚠")
        print(f"  Gap = {gap:.4f} > 0.08 — model is memorizing training data.")
        print(f"  Recommendations:")
        print(f"    1. Increase regularization (lower C, higher alpha, max_depth reduction)")
        print(f"    2. Reduce feature count (apply stricter importance threshold)")
        print(f"    3. Add more training data")
        print(f"    4. Try ensemble with cross-val predictions (stacking)")

    return {"verdict": verdict, "gap": round(gap, 4),
            "train_score": train_score, "test_score": test_score}
```

---

## Step 3 — Bias-Variance Tradeoff Learning Curve

```python
from sklearn.model_selection import learning_curve

def plot_validation_learning_curve(model,
                                    X: pd.DataFrame,
                                    y: pd.Series,
                                    problem_type: str,
                                    dataset_name: str = "dataset") -> str:
    """
    Train score (solid) vs CV score (dashed) vs training set size.
    Diagnoses whether model needs more data or less complexity.
    """
    print("\n" + "=" * 60)
    print("  STEP 3: BIAS-VARIANCE LEARNING CURVE")
    print("=" * 60)

    from sklearn.model_selection import StratifiedKFold, KFold

    is_clf = "classification" in problem_type
    cv = (StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
          if is_clf else
          KFold(n_splits=5, shuffle=True, random_state=42))

    scoring = ("roc_auc"     if "binary"         in problem_type else
               "f1_weighted" if "classification" in problem_type else "r2")

    train_sizes, train_scores, cv_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.10, 1.0, 12),
        n_jobs=-1,
    )

    tr_mean = train_scores.mean(axis=1)
    tr_std  = train_scores.std(axis=1)
    cv_mean = cv_scores.mean(axis=1)
    cv_std  = cv_scores.std(axis=1)

    fig, ax = new_figure(figsize=(14, 9))

    ax.plot(train_sizes, tr_mean,
            color=BRAHMA_COLORS["primary"], linewidth=2.5,
            linestyle="-", marker="o", markersize=6, label="Train score")
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                    alpha=0.15, color=BRAHMA_COLORS["primary"])

    ax.plot(train_sizes, cv_mean,
            color=BRAHMA_COLORS["highlight"], linewidth=2.5,
            linestyle="--", marker="s", markersize=6, label="CV score (5-fold)")
    ax.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std,
                    alpha=0.15, color=BRAHMA_COLORS["highlight"])

    ax.legend(fontsize=11, loc="lower right")

    # Diagnosis
    final_gap     = tr_mean[-1] - cv_mean[-1]
    cv_improving  = (cv_mean[-1] - cv_mean[len(cv_mean)//2]) > 0.01
    high_bias     = cv_mean[-1] < 0.70 and not cv_improving
    high_variance = final_gap > 0.08

    if high_bias:
        finding  = "High Bias — More Data Won't Help. Try a More Complex Model"
        subtitle = (f"CV score plateau ({cv_mean[-1]:.3f}) — model underfits. "
                    f"Consider more features, deeper trees, or more estimators.")
    elif high_variance:
        finding  = "High Variance — Model Is Overfitting. Reduce Complexity or Add Data"
        subtitle = (f"Train-CV gap = {final_gap:.3f}. "
                    f"More data may close this gap if the curve is still rising.")
    elif cv_improving:
        finding  = "Model Benefits from More Data — Performance Still Improving"
        subtitle = (f"CV score still rising at full training size. "
                    f"Collecting more data is likely to improve performance further.")
    else:
        finding  = "Model Has Converged — Performance Is Stable"
        subtitle = (f"Train-CV gap = {final_gap:.3f}  |  CV score = {cv_mean[-1]:.3f}  "
                    f"|  Further data unlikely to help significantly.")

    annotate_chart(ax,
        title=f"Does the Model Need More Data or Less Complexity? {finding}",
        subtitle=subtitle + f"  |  Scoring: {scoring}",
        xlabel="Training Set Size (rows)",
        ylabel=scoring.upper().replace("_", " "),
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, "validation_learning_curve.png")
    save_chart(fig, path)
    print(f"  Diagnosis: {finding}")
    return path
```

---

## Step 4 — Feature Stability Check (10 Runs, Different Seeds)

```python
def check_feature_stability(model_class,
                              model_params: dict,
                              X: pd.DataFrame,
                              y: pd.Series,
                              top_n: int = 10,
                              n_runs: int = 10) -> dict:
    """
    Fit the model 10 times with different random seeds.
    Track which features appear in the top-N for each run.
    Flag features that appear in top-N for < 50% of runs (unstable).
    """
    print("\n" + "=" * 60)
    print(f"  STEP 4: FEATURE STABILITY CHECK ({n_runs} runs, different seeds)")
    print("=" * 60)

    appearance_count = {col: 0 for col in X.columns}

    for seed in range(n_runs):
        params = {**model_params, "random_state": seed}

        # Suppress params not accepted by this model
        import inspect
        valid = inspect.signature(model_class.__init__).parameters
        filtered = {k: v for k, v in params.items() if k in valid}

        m = model_class(**filtered)
        m.fit(X, y)

        if hasattr(m, "feature_importances_"):
            imp = pd.Series(m.feature_importances_, index=X.columns)
            top_feats = imp.nlargest(top_n).index.tolist()
            for f in top_feats:
                appearance_count[f] += 1
        else:
            print("  Model does not expose feature_importances_. Skipping stability check.")
            return {}

    stable   = {f: c for f, c in appearance_count.items() if c >= n_runs * 0.5 and c > 0}
    unstable = {f: c for f, c in appearance_count.items()
                if 0 < c < n_runs * 0.5 and c in [appearance_count[f2]
                for f2 in list(appearance_count)[:top_n]]}

    print(f"\n  {'Feature':<40} {'Appearances':>12} {'Stability':>15}")
    print("  " + "-" * 70)

    sorted_by_count = sorted(
        [(f, c) for f, c in appearance_count.items() if c > 0],
        key=lambda x: -x[1]
    )
    for feat, count in sorted_by_count[:top_n + 5]:
        rate    = count / n_runs
        verdict = "STABLE" if rate >= 0.50 else "UNSTABLE ⚠"
        print(f"  {feat:<40} {count:>12}/{n_runs}  {verdict:>15}")

    unstable_feats = [f for f, c in sorted_by_count if c / n_runs < 0.50 and c > 0]
    if unstable_feats:
        print(f"\n  ⚠ Unstable features (appear in top {top_n} for < 50% of runs):")
        for f in unstable_feats[:10]:
            print(f"    {f} — consider dropping, may be noise-driven")
    else:
        print(f"\n  All top-{top_n} features are stable (appear ≥ 50% of runs).")

    return {"stable": list(stable.keys()), "unstable": unstable_feats,
            "appearance_counts": appearance_count}
```

---

## Step 5 — Threshold Analysis (Classification Only)

```python
from sklearn.metrics import precision_score, recall_score, f1_score as sk_f1

def plot_threshold_analysis(y_true, y_prob,
                              model_name: str,
                              dataset_name: str = "dataset") -> dict:
    """
    Plot Precision, Recall, F1 vs decision threshold (0.1 to 0.9).
    Mark the threshold that maximises F1.
    Tells decision-makers: at what confidence should we act?
    """
    print("\n" + "=" * 60)
    print("  STEP 5: THRESHOLD ANALYSIS")
    print("=" * 60)

    thresholds  = np.arange(0.10, 0.91, 0.02)
    precisions  = []
    recalls     = []
    f1_scores   = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_t, zero_division=0))
        f1_scores.append(sk_f1(y_true, y_pred_t, zero_division=0))

    # Optimal F1 threshold
    opt_idx       = int(np.argmax(f1_scores))
    opt_threshold = thresholds[opt_idx]
    opt_f1        = f1_scores[opt_idx]
    opt_prec      = precisions[opt_idx]
    opt_rec       = recalls[opt_idx]

    fig, ax = new_figure(figsize=(14, 9))

    ax.plot(thresholds, precisions, color=BRAHMA_COLORS["primary"],
            linewidth=2.5, label="Precision", marker=".")
    ax.plot(thresholds, recalls,    color=BRAHMA_COLORS["highlight"],
            linewidth=2.5, label="Recall",    marker=".")
    ax.plot(thresholds, f1_scores,  color=BRAHMA_COLORS["success"],
            linewidth=2.5, label="F1 Score",  marker=".", linestyle="--")

    ax.axvline(opt_threshold, color=BRAHMA_COLORS["warning"],
               linewidth=2, linestyle=":",
               label=f"Optimal F1 threshold = {opt_threshold:.2f}")

    ax.annotate(
        f"Optimal threshold\n"
        f"t={opt_threshold:.2f}\n"
        f"F1={opt_f1:.3f}\n"
        f"P={opt_prec:.3f} / R={opt_rec:.3f}",
        xy=(opt_threshold, opt_f1),
        xytext=(opt_threshold + 0.05, opt_f1 - 0.12),
        fontsize=9, color=BRAHMA_COLORS["dark"],
        arrowprops=dict(arrowstyle="->", color=BRAHMA_COLORS["warning"]),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFBEB", edgecolor=BRAHMA_COLORS["warning"]),
    )

    ax.set_xlim([0.08, 0.92])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=11, loc="upper right")

    # Business framing for title
    if opt_prec > opt_rec + 0.10:
        business = "Conservative Setting — Few False Alarms, May Miss Some Cases"
    elif opt_rec > opt_prec + 0.10:
        business = "Aggressive Setting — Catches More Cases, More False Alarms"
    else:
        business = "Balanced Precision and Recall"

    annotate_chart(ax,
        title=f"At What Confidence Should We Act? Optimal Threshold = {opt_threshold:.2f} ({business})",
        subtitle=(f"F1={opt_f1:.3f}  Precision={opt_prec:.3f}  Recall={opt_rec:.3f}  "
                  f"at t={opt_threshold:.2f}  |  Default threshold = 0.50"),
        xlabel="Decision Threshold",
        ylabel="Score",
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, "validation_threshold_analysis.png")
    save_chart(fig, path)

    print(f"\n  Optimal F1 threshold : {opt_threshold:.2f}")
    print(f"  At this threshold    : Precision={opt_prec:.3f}  Recall={opt_rec:.3f}  F1={opt_f1:.3f}")
    print(f"  Default threshold    : 0.50")
    print(f"  Recommendation       : Use threshold={opt_threshold:.2f} in production if F1 is primary metric.")

    return {
        "optimal_threshold": round(opt_threshold, 2),
        "optimal_f1":        round(opt_f1, 4),
        "optimal_precision": round(opt_prec, 4),
        "optimal_recall":    round(opt_rec, 4),
    }
```

---

## Validation Summary Report

```python
import datetime

def print_validation_report(cv_results: dict,
                              overfit_check: dict,
                              stability: dict,
                              threshold: dict,
                              model_name: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 60)
    print("  ✅  MODEL VALIDATION COMPLETE")
    print("=" * 60)
    print(f"  Model     : {model_name}")
    print(f"  Timestamp : {ts}")
    print("-" * 60)

    print(f"  Cross-Validation (10-fold):")
    for metric, vals in cv_results.items():
        stable = "✓" if vals["val_std"] <= 0.05 else "⚠"
        print(f"    {metric:<25} {vals['val_mean']:.4f} ± {vals['val_std']:.4f}  {stable}")

    print(f"\n  Overfitting:")
    print(f"    Verdict  : {overfit_check.get('verdict', 'N/A')}")
    print(f"    Gap      : {overfit_check.get('gap', 'N/A')}")

    if stability:
        print(f"\n  Feature Stability:")
        print(f"    Stable   : {len(stability.get('stable', []))} features")
        print(f"    Unstable : {len(stability.get('unstable', []))} features")

    if threshold:
        print(f"\n  Optimal Threshold  : {threshold.get('optimal_threshold', 0.5)}")
        print(f"  At this threshold  : F1={threshold.get('optimal_f1')}  "
              f"P={threshold.get('optimal_precision')}  R={threshold.get('optimal_recall')}")

    print("-" * 60)
    verdict = overfit_check.get("verdict", "")
    if "LEAKAGE" in verdict:
        print("  FINAL VERDICT: 🚨 STOP — INVESTIGATE DATA LEAKAGE BEFORE PROCEEDING")
    elif "OVERFIT" in verdict and "SLIGHT" not in verdict:
        print("  FINAL VERDICT: ⚠ OVERFIT — Regularize before deploying")
    else:
        print("  FINAL VERDICT: ✓ Model validates — safe to proceed to ensembling or deployment")
    print("=" * 60 + "\n")
```

---

## Master Orchestrator

```python
def run_model_validation(
    model,
    model_class,
    model_params: dict,
    splits: dict,
    problem_type: str,
    model_name: str = "Model",
    dataset_name: str = "dataset",
    primary_metric_train: float = None,
    primary_metric_test: float = None,
    primary_metric_name: str = "primary metric",
) -> dict:

    X_full = pd.concat([splits["X_train"], splits["X_val"], splits["X_test"]])
    y_full = pd.concat([splits["y_train"], splits["y_val"], splits["y_test"]])

    print("\n" + "=" * 60)
    print("  BRAHMA MODEL VALIDATION")
    print(f"  Model   : {model_name}")
    print(f"  Problem : {problem_type}")
    print("=" * 60)

    # Step 1: Cross-validation
    cv_results = run_cross_validation(model, X_full, y_full, problem_type)

    # Step 2: Overfitting check
    overfit_check = {}
    if primary_metric_train and primary_metric_test:
        overfit_check = check_overfitting(
            cv_results, primary_metric_train,
            primary_metric_test, primary_metric_name
        )

    # Step 3: Learning curve
    plot_validation_learning_curve(model, splits["X_train"], splits["y_train"],
                                    problem_type, dataset_name)

    # Step 4: Feature stability
    stability = {}
    if model_class and hasattr(model_class, "__init__"):
        stability = check_feature_stability(
            model_class, model_params,
            splits["X_train"], splits["y_train"]
        )

    # Step 5: Threshold analysis (classification only)
    threshold = {}
    if "binary" in problem_type and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(splits["X_test"])[:, 1]
        threshold = plot_threshold_analysis(
            splits["y_test"], y_prob, model_name, dataset_name
        )

    print_validation_report(cv_results, overfit_check,
                              stability, threshold, model_name)

    return {
        "cv_results":    cv_results,
        "overfit_check": overfit_check,
        "stability":     stability,
        "threshold":     threshold,
    }
```

---

## Usage Example

```python
from sklearn.ensemble import RandomForestClassifier

val_results = run_model_validation(
    model          = training["best_model"],
    model_class    = RandomForestClassifier,
    model_params   = selection["primary"]["params"],
    splits         = training["splits"],
    problem_type   = selection["problem_type"],
    model_name     = "XGBoost (tuned)",
    dataset_name   = "telco_churn.csv",
    primary_metric_train = 0.94,
    primary_metric_test  = 0.87,
    primary_metric_name  = "roc_auc",
)

# Use optimal threshold in production
optimal_threshold = val_results["threshold"]["optimal_threshold"]
```
