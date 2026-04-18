# Skill: UAT Checklist

## Purpose
A model with 95% accuracy can fail UAT if it's wrong on obvious cases.
Six structured checks before any model goes to production.
FAIL on Check 1 means stop — do not proceed.

**Depends on:** `skills/visualization_style.md`

---

## Core Rule — Trust But Verify

Every check produces a PASS, WARN, or FAIL verdict.
- FAIL: Do not proceed. Fix root cause.
- WARN: Proceed with monitoring. Document the finding.
- PASS: Move to next check.

A model that passes all 6 checks is UAT-certified.

---

## Standard Import Block

```python
import pandas as pd
import numpy as np
import time
import os
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "skills")

from visualization_style import (
    apply_brahma_style, new_figure, annotate_chart,
    save_chart, BRAHMA_COLORS,
)
apply_brahma_style()
```

---

## Check 1 — Smoke Test (5 Hand-Picked Cases)

```python
def smoke_test(model,
               feature_cols: list,
               obvious_positives: list[dict],
               obvious_negatives: list[dict],
               borderline_case: dict,
               problem_type: str = "binary_classification",
               positive_threshold: float = 0.60,
               negative_threshold: float = 0.40) -> dict:
    """
    Run the model on 5 hand-crafted cases:
        - 2 obvious positives (should predict high risk / positive class)
        - 2 obvious negatives (should predict low risk / negative class)
        - 1 borderline case (document what the model says — no strict pass/fail)

    FAIL = any obvious positive predicts below positive_threshold, OR
           any obvious negative predicts above negative_threshold.

    Why hand-crafted cases?
    Aggregate metrics can mask catastrophic failures on obvious inputs.
    If a fraud model classifies a $1M cash withdrawal as low risk, AUC=0.91 means nothing.
    """
    print("\n" + "=" * 60)
    print("  CHECK 1: SMOKE TEST")
    print("  5 hand-picked cases — obvious positives, negatives, borderline.")
    print("  FAIL here = do not proceed to production.")
    print("=" * 60)

    all_cases = (
        [(c, "OBVIOUS_POSITIVE") for c in obvious_positives] +
        [(c, "OBVIOUS_NEGATIVE") for c in obvious_negatives] +
        [(borderline_case, "BORDERLINE")]
    )

    failures = []
    results  = []

    for case_dict, case_type in all_cases:
        row = pd.DataFrame([case_dict])
        # Fill any missing columns with 0
        for col in feature_cols:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_cols]

        if hasattr(model, "predict_proba"):
            prob  = model.predict_proba(row)[0]
            score = prob[1] if len(prob) == 2 else prob.max()
            pred  = int(np.argmax(prob))
        else:
            pred  = int(model.predict(row)[0])
            score = float(pred)

        # Verdict
        if case_type == "OBVIOUS_POSITIVE":
            if score < positive_threshold:
                verdict = f"FAIL — predicted {score:.3f}, expected ≥ {positive_threshold}"
                failures.append({"case": case_dict, "type": case_type,
                                  "score": score, "issue": verdict})
            else:
                verdict = f"PASS — predicted {score:.3f} ✓"
        elif case_type == "OBVIOUS_NEGATIVE":
            if score > negative_threshold:
                verdict = f"FAIL — predicted {score:.3f}, expected ≤ {negative_threshold}"
                failures.append({"case": case_dict, "type": case_type,
                                  "score": score, "issue": verdict})
            else:
                verdict = f"PASS — predicted {score:.3f} ✓"
        else:
            verdict = f"INFO — borderline predicted {score:.3f} (document, no pass/fail)"

        results.append({"case_type": case_type, "score": score, "pred": pred, "verdict": verdict})
        print(f"\n  [{case_type}]")
        print(f"    Input   : {case_dict}")
        print(f"    Score   : {score:.4f}  |  Prediction: {pred}")
        print(f"    Verdict : {verdict}")

    overall = "FAIL" if failures else "PASS"
    print(f"\n  {'─'*50}")
    print(f"  SMOKE TEST: {overall}")
    if failures:
        print(f"  {len(failures)} case(s) failed. DO NOT PROCEED TO PRODUCTION.")
        print(f"  Investigate model calibration or feature pipeline before re-testing.")
    else:
        print(f"  All obvious cases correctly classified. ✓")

    return {"verdict": overall, "failures": failures, "results": results}
```

---

## Check 2 — Edge Cases

```python
def edge_case_test(model,
                    feature_cols: list,
                    training_max: pd.Series = None) -> dict:
    """
    5 edge cases — all must handle gracefully (no crash, no NaN output):
        1. All-null row          → should not crash; imputed values used
        2. Values 10× max        → extreme input; should produce valid prediction
        3. All-zero row          → zero vector; should produce valid prediction
        4. Unseen category       → only relevant if model received raw categoricals
        5. Single-row input      → batch pipeline should handle n=1

    FAIL = any case crashes or returns NaN/None.
    WARN = any case produces extreme probability (0.0 or 1.0 exactly).
    """
    print("\n" + "=" * 60)
    print("  CHECK 2: EDGE CASES")
    print("  All cases must return a valid prediction without crashing.")
    print("=" * 60)

    numeric_cols = [c for c in feature_cols]
    verdicts     = {}

    def run_case(name: str, row: pd.DataFrame) -> str:
        try:
            for col in feature_cols:
                if col not in row.columns:
                    row[col] = 0
            row = row[feature_cols].fillna(0)
            if hasattr(model, "predict_proba"):
                result = model.predict_proba(row)
                val    = result[0, 1] if result.shape[1] == 2 else result[0].max()
            else:
                result = model.predict(row)
                val    = float(result[0])

            if np.isnan(val) or val is None:
                return f"FAIL — returned NaN"
            if val in (0.0, 1.0):
                return f"WARN — extreme prediction ({val:.4f}), check calibration"
            return f"PASS — prediction={val:.4f}"
        except Exception as e:
            return f"FAIL — CRASHED: {type(e).__name__}: {e}"

    # Case 1: All-null row
    null_row = pd.DataFrame([{col: np.nan for col in numeric_cols}])
    v = run_case("All-null row", null_row)
    verdicts["all_null"] = v
    print(f"\n  [1] All-null row          → {v}")

    # Case 2: Values 10× max
    if training_max is not None:
        extreme_vals = {col: training_max.get(col, 1.0) * 10 for col in numeric_cols}
    else:
        extreme_vals = {col: 999_999 for col in numeric_cols}
    extreme_row = pd.DataFrame([extreme_vals])
    v = run_case("10× max values", extreme_row)
    verdicts["extreme_values"] = v
    print(f"  [2] Values 10× max        → {v}")

    # Case 3: All-zero row
    zero_row = pd.DataFrame([{col: 0 for col in numeric_cols}])
    v = run_case("All-zero row", zero_row)
    verdicts["all_zero"] = v
    print(f"  [3] All-zero row          → {v}")

    # Case 4: Unseen category (numeric proxy: negative values)
    unseen_row = pd.DataFrame([{col: -9999 for col in numeric_cols}])
    v = run_case("Unseen values (-9999)", unseen_row)
    verdicts["unseen_category"] = v
    print(f"  [4] Unseen/OOV values     → {v}")

    # Case 5: Single row
    single_row = pd.DataFrame([{col: 0 for col in numeric_cols}])
    v = run_case("Single-row input", single_row)
    verdicts["single_row"] = v
    print(f"  [5] Single-row input      → {v}")

    fails = [k for k, v in verdicts.items() if "FAIL" in v]
    warns = [k for k, v in verdicts.items() if "WARN" in v]

    overall = "FAIL" if fails else ("WARN" if warns else "PASS")
    print(f"\n  EDGE CASE TEST: {overall}")
    if fails:
        print(f"  {len(fails)} case(s) crashed. Fix before deployment.")
    if warns:
        print(f"  {len(warns)} case(s) produced extreme predictions — review calibration.")

    return {"verdict": overall, "verdicts": verdicts}
```

---

## Check 3 — Prediction Distribution Audit

```python
def prediction_distribution_audit(model,
                                    X_test: pd.DataFrame,
                                    y_test: pd.Series,
                                    problem_type: str = "binary_classification",
                                    dataset_name: str = "dataset") -> dict:
    """
    Audit the distribution of predictions on the test set.

    Flags:
    - > 90% of predictions cluster near 0 or 1 (overconfident model)
    - Predicted positive rate differs > 2× from actual positive rate
      (systematic bias in who gets predicted positive)
    """
    print("\n" + "=" * 60)
    print("  CHECK 3: PREDICTION DISTRIBUTION AUDIT")
    print("=" * 60)

    issues  = []
    verdict = "PASS"

    if "classification" in problem_type and hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        if probs.shape[1] == 2:
            probs = probs[:, 1]
        else:
            probs = probs.max(axis=1)

        # Check 1: Overconfidence — >90% near 0 or 1
        near_extremes = np.mean((probs < 0.05) | (probs > 0.95))
        print(f"\n  Probability distribution summary:")
        print(f"    Mean prob         : {probs.mean():.4f}")
        print(f"    Std prob          : {probs.std():.4f}")
        print(f"    % near 0 (<0.05)  : {np.mean(probs < 0.05)*100:.1f}%")
        print(f"    % near 1 (>0.95)  : {np.mean(probs > 0.95)*100:.1f}%")
        print(f"    % near extremes   : {near_extremes*100:.1f}%")

        if near_extremes > 0.90:
            msg = (f"WARN — {near_extremes*100:.1f}% of predictions near 0 or 1. "
                   f"Model is overconfident. Consider Platt scaling or isotonic regression.")
            issues.append(msg)
            verdict = "WARN"
            print(f"\n  ⚠ {msg}")
        else:
            print(f"\n  Overconfidence check: PASS ({near_extremes*100:.1f}% near extremes)")

        # Check 2: Predicted vs actual positive rate
        preds         = model.predict(X_test)
        pred_pos_rate = np.mean(preds)
        actual_pos_rate = y_test.mean() if hasattr(y_test, "mean") else np.mean(y_test)
        ratio = pred_pos_rate / max(actual_pos_rate, 1e-9)

        print(f"\n  Positive rate audit:")
        print(f"    Actual positive rate    : {actual_pos_rate:.4f} ({actual_pos_rate*100:.1f}%)")
        print(f"    Predicted positive rate : {pred_pos_rate:.4f} ({pred_pos_rate*100:.1f}%)")
        print(f"    Ratio (pred/actual)     : {ratio:.2f}×")

        if ratio > 2.0 or ratio < 0.5:
            msg = (f"WARN — predicted positive rate ({pred_pos_rate*100:.1f}%) differs "
                   f"{ratio:.1f}× from actual ({actual_pos_rate*100:.1f}%). "
                   f"Model may be systematically over- or under-predicting positives.")
            issues.append(msg)
            verdict = "WARN" if verdict == "PASS" else verdict
            print(f"\n  ⚠ {msg}")
        else:
            print(f"\n  Positive rate check: PASS (ratio={ratio:.2f}×, within 0.5–2.0× range)")

    else:
        preds = model.predict(X_test)
        print(f"  Prediction range: [{preds.min():.4f}, {preds.max():.4f}]")
        print(f"  Mean: {preds.mean():.4f}  Std: {preds.std():.4f}")
        print(f"  No probability audit for non-probabilistic or regression models.")

    print(f"\n  PREDICTION DISTRIBUTION: {verdict}")
    return {"verdict": verdict, "issues": issues}
```

---

## Check 4 — Subgroup Fairness Check

```python
from sklearn.metrics import f1_score as sk_f1

def subgroup_fairness_check(model,
                              X_test: pd.DataFrame,
                              y_test: pd.Series,
                              df_test: pd.DataFrame,
                              segment_cols: list,
                              problem_type: str = "binary_classification") -> dict:
    """
    Split predictions by demographic/segment columns.
    Flag any subgroup where F1 is > 10% below overall F1.

    Report even if not blocking — fairness always matters.
    A model that is accurate overall but catastrophically wrong
    on a subgroup is not fit for production.
    """
    print("\n" + "=" * 60)
    print("  CHECK 4: SUBGROUP FAIRNESS CHECK")
    print("  Flagging subgroups where F1 is > 10% below overall.")
    print("  This check is ALWAYS reported — fairness matters.")
    print("=" * 60)

    if "classification" not in problem_type:
        print("  Subgroup fairness check is for classification tasks only.")
        return {"verdict": "SKIP"}

    preds        = model.predict(X_test)
    overall_f1   = sk_f1(y_test, preds, average="weighted", zero_division=0)
    flagged      = []

    print(f"\n  Overall F1 (weighted): {overall_f1:.4f}")
    print(f"\n  {'Segment':<40} {'N':>8} {'F1':>10} {'vs Overall':>12}  Verdict")
    print("  " + "-" * 78)

    for col in segment_cols:
        if col not in df_test.columns:
            continue
        for val in df_test[col].dropna().unique():
            mask       = df_test[col] == val
            n_sub      = mask.sum()
            if n_sub < 30:
                continue   # too small for reliable F1
            y_sub      = y_test[mask]
            p_sub      = preds[mask]
            sub_f1     = sk_f1(y_sub, p_sub, average="weighted", zero_division=0)
            delta      = sub_f1 - overall_f1
            delta_pct  = delta / max(overall_f1, 1e-9) * 100

            if delta_pct < -10:
                verdict_str = f"⚠ FLAGGED  ({delta_pct:.1f}%)"
                flagged.append({
                    "segment_col": col, "value": val,
                    "n": n_sub, "f1": sub_f1, "delta_pct": round(delta_pct, 1)
                })
            else:
                verdict_str = f"OK"

            print(f"  {col}={val:<35} {n_sub:>8} {sub_f1:>10.4f} {delta_pct:>+11.1f}%  {verdict_str}")

    if flagged:
        print(f"\n  ⚠ {len(flagged)} subgroup(s) flagged:")
        for f in flagged:
            print(f"    {f['segment_col']}={f['value']}  F1={f['f1']:.4f}  "
                  f"({f['delta_pct']:.1f}% below overall)")
        print(f"\n  These subgroups may be underserved by the model.")
        print(f"  Recommendations:")
        print(f"    1. Investigate whether subgroup is underrepresented in training data.")
        print(f"    2. Consider subgroup-specific threshold adjustment.")
        print(f"    3. Document fairness gaps in the model card before deployment.")
        print(f"  NOTE: This is a WARNING, not a FAIL — decision is domain-dependent.")
        verdict = "WARN"
    else:
        print(f"\n  No subgroups with F1 > 10% below overall. Fairness check passed.")
        verdict = "PASS"

    return {"verdict": verdict, "flagged": flagged, "overall_f1": round(overall_f1, 4)}
```

---

## Check 5 — Business Logic Sanity

```python
def business_logic_sanity(model,
                            X_test: pd.DataFrame,
                            feature_cols: list) -> dict:
    """
    Present the top 5 SHAP features to the user and ask:
    'Do these make business sense?'

    This check cannot be automated — it requires a domain expert
    to confirm that the model learned real signal, not noise.
    Brahma presents the evidence; the human makes the call.
    """
    print("\n" + "=" * 60)
    print("  CHECK 5: BUSINESS LOGIC SANITY")
    print("  Brahma presents the evidence. You make the call.")
    print("=" * 60)

    top_features = []

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_test)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]
        mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_test.columns)
        top5     = mean_abs.nlargest(5)
        top_features = list(top5.index)

        print(f"\n  Top 5 features driving predictions (mean |SHAP|):\n")
        for i, (feat, imp) in enumerate(top5.items(), 1):
            print(f"  #{i}  {feat:<40}  mean |SHAP| = {imp:.4f}")

    except Exception:
        # Fallback to model feature_importances_
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=X_test.columns).nlargest(5)
            top_features = list(imp.index)
            print(f"\n  Top 5 features by importance (SHAP unavailable):\n")
            for i, (feat, val) in enumerate(imp.items(), 1):
                print(f"  #{i}  {feat:<40}  importance = {val:.4f}")
        else:
            print("  Feature importance unavailable for this model type.")

    print(f"\n  ─── HUMAN REVIEW REQUIRED ───")
    print(f"  Ask a domain expert:")
    print(f"    1. Do these top features make causal or business sense?")
    print(f"    2. Are any features that should NOT predict the outcome present?")
    print(f"       (e.g. customer ID, timestamps, proxy variables for protected classes)")
    print(f"    3. Is anything critically important missing from the top 5?")
    print(f"\n  Record the domain expert's verdict below:")
    print(f"  → If YES to all: mark PASS")
    print(f"  → If any feature is suspicious: mark WARN and document")
    print(f"  → If proxy/protected class detected: mark FAIL and retrain")

    return {
        "verdict":        "MANUAL_REVIEW_REQUIRED",
        "top_5_features": top_features,
    }
```

---

## Check 6 — Latency Test

```python
def latency_test(model,
                  X_test: pd.DataFrame,
                  feature_cols: list,
                  real_time_threshold_ms: float = 100.0) -> dict:
    """
    Time single prediction and batch prediction of 1,000 rows.

    Flag if single prediction > 100ms (not suitable for real-time use).
    Report batch latency for capacity planning.
    """
    print("\n" + "=" * 60)
    print("  CHECK 6: LATENCY TEST")
    print(f"  Real-time threshold: {real_time_threshold_ms}ms per prediction")
    print("=" * 60)

    sample_row = X_test[feature_cols].head(1)
    batch_rows = X_test[feature_cols].head(1000)
    if len(batch_rows) < 1000:
        batch_rows = pd.concat([batch_rows] * (1000 // len(batch_rows) + 1)).head(1000)

    N_WARMUP = 3
    N_SINGLE = 20

    # Warmup
    for _ in range(N_WARMUP):
        _ = model.predict(sample_row)

    # Single prediction timing
    single_times = []
    for _ in range(N_SINGLE):
        t0 = time.perf_counter()
        _  = model.predict(sample_row)
        single_times.append((time.perf_counter() - t0) * 1000)

    single_mean = np.mean(single_times)
    single_p95  = np.percentile(single_times, 95)

    # Batch timing
    t0        = time.perf_counter()
    _         = model.predict(batch_rows)
    batch_ms  = (time.perf_counter() - t0) * 1000
    per_row   = batch_ms / 1000

    print(f"\n  Single prediction  (n={N_SINGLE} runs):")
    print(f"    Mean latency : {single_mean:.2f} ms")
    print(f"    P95 latency  : {single_p95:.2f} ms")
    print(f"\n  Batch prediction   (n=1,000 rows):")
    print(f"    Total time   : {batch_ms:.2f} ms")
    print(f"    Per-row time : {per_row:.3f} ms")

    single_verdict = "PASS" if single_mean <= real_time_threshold_ms else "WARN"
    if single_mean > real_time_threshold_ms:
        print(f"\n  ⚠ Single prediction ({single_mean:.1f}ms) exceeds "
              f"real-time threshold ({real_time_threshold_ms}ms).")
        print(f"    Not suitable for synchronous real-time APIs.")
        print(f"    Consider: async prediction, pre-scoring batch, or model compression.")
    else:
        print(f"\n  Latency within real-time threshold ({single_mean:.1f}ms ≤ {real_time_threshold_ms}ms). ✓")

    return {
        "verdict":         single_verdict,
        "single_mean_ms":  round(single_mean, 2),
        "single_p95_ms":   round(single_p95, 2),
        "batch_total_ms":  round(batch_ms, 2),
        "per_row_ms":      round(per_row, 3),
    }
```

---

## UAT Summary Report

```python
import datetime

def print_uat_report(check_results: dict, model_name: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 60)
    print("  UAT SUMMARY REPORT")
    print("=" * 60)
    print(f"  Model     : {model_name}")
    print(f"  Timestamp : {ts}")
    print("-" * 60)

    verdict_order = {"FAIL": 0, "WARN": 1, "MANUAL_REVIEW_REQUIRED": 2, "PASS": 3, "SKIP": 4}
    overall_verdict = "PASS"

    rows = [
        ("Check 1: Smoke Test",          check_results.get("smoke_test", {}).get("verdict", "SKIP")),
        ("Check 2: Edge Cases",           check_results.get("edge_cases", {}).get("verdict", "SKIP")),
        ("Check 3: Prediction Audit",     check_results.get("pred_audit", {}).get("verdict", "SKIP")),
        ("Check 4: Subgroup Fairness",    check_results.get("fairness",   {}).get("verdict", "SKIP")),
        ("Check 5: Business Logic",       check_results.get("business",   {}).get("verdict", "MANUAL_REVIEW_REQUIRED")),
        ("Check 6: Latency",              check_results.get("latency",    {}).get("verdict", "SKIP")),
    ]

    for check_name, verdict in rows:
        icon = {"FAIL": "✗", "WARN": "⚠", "PASS": "✓",
                "MANUAL_REVIEW_REQUIRED": "?", "SKIP": "-"}.get(verdict, "?")
        print(f"  {icon}  {check_name:<35} {verdict}")
        if verdict_order.get(verdict, 5) < verdict_order.get(overall_verdict, 5):
            overall_verdict = verdict

    print("-" * 60)
    if overall_verdict == "FAIL":
        print("  OVERALL: ✗ FAIL — DO NOT DEPLOY. Fix failures and re-run UAT.")
    elif overall_verdict == "WARN":
        print("  OVERALL: ⚠ WARN — Deploy with monitoring. Document all warnings.")
    elif overall_verdict == "MANUAL_REVIEW_REQUIRED":
        print("  OVERALL: ? PENDING — Awaiting domain expert sign-off on Check 5.")
    else:
        print("  OVERALL: ✓ PASS — Model is UAT-certified. Safe to proceed to deployment.")
    print("=" * 60 + "\n")
```

---

## Master Orchestrator

```python
def run_uat(
    model,
    splits: dict,
    df_test: pd.DataFrame,
    problem_type: str,
    feature_cols: list,
    obvious_positives: list[dict],
    obvious_negatives: list[dict],
    borderline_case: dict,
    segment_cols: list = None,
    model_name: str = "Model",
    training_max: pd.Series = None,
) -> dict:

    X_test = splits["X_test"]
    y_test = splits["y_test"]

    print("\n" + "=" * 70)
    print("  BRAHMA UAT CHECKLIST")
    print(f"  Model   : {model_name}")
    print(f"  Problem : {problem_type}")
    print("=" * 70)

    results = {}

    # Check 1
    results["smoke_test"] = smoke_test(
        model, feature_cols,
        obvious_positives, obvious_negatives, borderline_case,
        problem_type,
    )

    # Check 2
    results["edge_cases"] = edge_case_test(model, feature_cols, training_max)

    # Check 3
    results["pred_audit"] = prediction_distribution_audit(
        model, X_test, y_test, problem_type
    )

    # Check 4
    if segment_cols:
        results["fairness"] = subgroup_fairness_check(
            model, X_test, y_test, df_test, segment_cols, problem_type
        )

    # Check 5
    results["business"] = business_logic_sanity(model, X_test, feature_cols)

    # Check 6
    results["latency"] = latency_test(model, X_test, feature_cols)

    print_uat_report(results, model_name)

    return results
```

---

## Usage Example

```python
uat_results = run_uat(
    model         = ensemble_results["final"]["model"],
    splits        = training["splits"],
    df_test       = df.loc[training["splits"]["X_test"].index],
    problem_type  = selection["problem_type"],
    feature_cols  = training["splits"]["feature_cols"],
    model_name    = "XGBoost (tuned)",

    # Hand-crafted smoke test cases — domain expert provides these
    obvious_positives=[
        {"tenure_months": 2,  "monthly_charges": 95,  "contract": 0, "num_complaints": 5},
        {"tenure_months": 1,  "monthly_charges": 100, "contract": 0, "num_complaints": 4},
    ],
    obvious_negatives=[
        {"tenure_months": 60, "monthly_charges": 30,  "contract": 1, "num_complaints": 0},
        {"tenure_months": 48, "monthly_charges": 25,  "contract": 1, "num_complaints": 0},
    ],
    borderline_case={"tenure_months": 12, "monthly_charges": 60, "contract": 0, "num_complaints": 1},

    segment_cols=["gender", "senior_citizen", "partner"],
    training_max=df[training["splits"]["feature_cols"]].max(),
)
```
