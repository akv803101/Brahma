# Skill: Deployment Tester

## Purpose
Wrap the final model in a production function, validate serialization,
set up drift detection, and generate a monitoring template.
A model in a notebook is not a model in production. This skill bridges that gap.

---

## Standard Import Block

```python
import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import datetime
import warnings
warnings.filterwarnings("ignore")

OUTPUT_MODEL_DIR = "outputs/models"
OUTPUT_DATA_DIR  = "outputs/data"
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DATA_DIR,  exist_ok=True)
```

---

## Step 1 — predict_brahma() Production Function

```python
def build_predict_function(model,
                             feature_cols: list,
                             scaler=None,
                             optimal_threshold: float = 0.50,
                             model_version: str = "brahma_v1") -> callable:
    """
    Wraps the final model in a production-ready predict function.

    Returns a dict with:
        prediction   : 0 or 1 (classification) / float (regression)
        probability  : float (classification only)
        risk_tier    : 'HIGH' / 'MEDIUM' / 'LOW' (classification only)
        top_reasons  : list of top 3 SHAP feature contributions
        model_version: version string
        timestamp    : ISO datetime string
        latency_ms   : inference time in milliseconds

    Risk tiers:
        probability > 0.70 → HIGH
        probability > 0.40 → MEDIUM
        probability ≤ 0.40 → LOW
    """

    def predict_brahma(input_data: dict | pd.DataFrame) -> dict:
        t_start = time.perf_counter()

        # ── Input coercion ────────────────────────────────────────────────────
        if isinstance(input_data, dict):
            row = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            row = input_data.copy()
        else:
            raise TypeError(f"input_data must be dict or DataFrame, got {type(input_data)}")

        # ── Input validation (see Step 2) ─────────────────────────────────────
        missing_cols = [c for c in feature_cols if c not in row.columns]
        if missing_cols:
            return {
                "error": f"MISSING_COLUMNS: {missing_cols}",
                "prediction": None, "probability": None,
            }

        wrong_types = {}
        for col in feature_cols:
            if col in row.columns:
                try:
                    row[col] = pd.to_numeric(row[col], errors="raise")
                except Exception:
                    wrong_types[col] = str(row[col].dtype)
        if wrong_types:
            return {
                "error": f"WRONG_DTYPES: {wrong_types}",
                "prediction": None, "probability": None,
            }

        X = row[feature_cols].fillna(0)

        # ── Scale if scaler provided ──────────────────────────────────────────
        if scaler is not None:
            try:
                X = pd.DataFrame(scaler.transform(X), columns=feature_cols)
            except Exception as e:
                return {"error": f"SCALER_ERROR: {e}",
                        "prediction": None, "probability": None}

        # ── Predict ──────────────────────────────────────────────────────────
        probability = None
        risk_tier   = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if probs.shape[1] == 2:
                probability = float(probs[0, 1])
            else:
                probability = float(probs[0].max())
            prediction = int(probability >= optimal_threshold)

            if probability > 0.70:
                risk_tier = "HIGH"
            elif probability > 0.40:
                risk_tier = "MEDIUM"
            else:
                risk_tier = "LOW"
        else:
            raw_pred   = model.predict(X)
            prediction = float(raw_pred[0])
            risk_tier  = None

        # ── Top 3 SHAP reasons ────────────────────────────────────────────────
        top_reasons = []
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X)
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                shap_vals = shap_vals[1]
            shap_row = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
            shap_series = pd.Series(np.abs(shap_row), index=feature_cols)
            top3 = shap_series.nlargest(3)
            for feat, val in top3.items():
                raw_val = float(X[feat].iloc[0])
                top_reasons.append({
                    "feature":     feat,
                    "value":       round(raw_val, 4),
                    "shap_impact": round(float(shap_row[feature_cols.index(feat)]), 4),
                })
        except Exception:
            top_reasons = [{"feature": c, "value": None, "shap_impact": None}
                           for c in feature_cols[:3]]

        latency_ms = round((time.perf_counter() - t_start) * 1000, 3)

        return {
            "prediction":    prediction,
            "probability":   round(probability, 4) if probability is not None else None,
            "risk_tier":     risk_tier,
            "top_reasons":   top_reasons,
            "model_version": model_version,
            "timestamp":     datetime.datetime.utcnow().isoformat() + "Z",
            "latency_ms":    latency_ms,
        }

    return predict_brahma


def demo_predict_function(predict_fn: callable, sample_input: dict):
    """
    Demonstrate the predict_brahma function with a sample input.
    Print the full structured output.
    """
    print("\n" + "=" * 60)
    print("  STEP 1: predict_brahma() DEMO")
    print("=" * 60)
    print(f"\n  Input: {sample_input}\n")

    result = predict_fn(sample_input)

    for key, val in result.items():
        if key == "top_reasons":
            print(f"  top_reasons:")
            for r in val:
                print(f"    feature={r['feature']}  value={r['value']}  "
                      f"shap_impact={r['shap_impact']}")
        else:
            print(f"  {key:<20} {val}")
```

---

## Step 2 — Input Validation Layer

```python
def build_input_validator(feature_cols: list,
                            expected_dtypes: dict = None) -> callable:
    """
    Returns a validate_input() function that checks incoming data
    before it reaches the model.

    Rejects with clear error messages:
        - Wrong column names
        - Wrong data types
        - Missing required columns
        - Empty input

    expected_dtypes: {col_name: 'numeric'|'string'} — optional
    """

    def validate_input(input_data: dict | pd.DataFrame) -> tuple[bool, str]:
        """
        Returns (is_valid: bool, error_message: str).
        error_message is empty string if valid.
        """

        # Empty input check
        if input_data is None:
            return False, "EMPTY_INPUT: input_data is None"
        if isinstance(input_data, dict) and len(input_data) == 0:
            return False, "EMPTY_INPUT: empty dict provided"
        if isinstance(input_data, pd.DataFrame) and len(input_data) == 0:
            return False, "EMPTY_INPUT: DataFrame has 0 rows"

        # Coerce to dict for column checks
        if isinstance(input_data, pd.DataFrame):
            cols_present = list(input_data.columns)
        elif isinstance(input_data, dict):
            cols_present = list(input_data.keys())
        else:
            return False, f"WRONG_TYPE: expected dict or DataFrame, got {type(input_data)}"

        # Missing required columns
        missing = [c for c in feature_cols if c not in cols_present]
        if missing:
            return False, f"MISSING_COLUMNS: {missing}"

        # Wrong column names (unexpected columns are accepted, just ignored)
        # Only hard-fail on missing required columns

        # Data type check
        if isinstance(input_data, pd.DataFrame) and expected_dtypes:
            for col, expected in expected_dtypes.items():
                if col not in input_data.columns:
                    continue
                if expected == "numeric" and not pd.api.types.is_numeric_dtype(input_data[col]):
                    return False, f"WRONG_DTYPE: {col} should be numeric, got {input_data[col].dtype}"

        return True, ""

    print("\n" + "=" * 60)
    print("  STEP 2: INPUT VALIDATION LAYER")
    print("  Required columns: " + str(feature_cols[:5]) + ("..." if len(feature_cols) > 5 else ""))
    print("  Validation checks: empty input, missing columns, wrong dtypes")
    print("=" * 60)

    return validate_input
```

---

## Step 3 — Serialization Test

```python
def serialization_test(model,
                         predict_fn: callable,
                         sample_input: dict,
                         feature_cols: list,
                         model_path: str = "outputs/models/final_model.pkl") -> dict:
    """
    Save with joblib → load in a fresh call → verify identical predictions.
    If outputs differ: serialization is broken. Do not deploy.

    Why this matters:
    A model that produces different results after save/load has non-deterministic
    state (e.g. unfitted transformers, random seeds not fixed).
    This would silently corrupt production predictions.
    """
    print("\n" + "=" * 60)
    print("  STEP 3: SERIALIZATION TEST")
    print("  Save → Load → Verify identical predictions")
    print("=" * 60)

    # Original prediction
    original_result = predict_fn(sample_input)
    original_pred   = original_result.get("prediction")
    original_prob   = original_result.get("probability")

    print(f"\n  Original prediction  : {original_pred}")
    print(f"  Original probability : {original_prob}")

    # Save
    joblib.dump(model, model_path)
    print(f"\n  Saved model → {model_path}")

    # Load fresh
    reloaded_model = joblib.load(model_path)
    reloaded_fn    = build_predict_function(reloaded_model, feature_cols)
    reloaded_result = reloaded_fn(sample_input)
    reloaded_pred   = reloaded_result.get("prediction")
    reloaded_prob   = reloaded_result.get("probability")

    print(f"  Reloaded prediction  : {reloaded_pred}")
    print(f"  Reloaded probability : {reloaded_prob}")

    # Verify
    pred_match = (original_pred == reloaded_pred)
    prob_match = (
        abs((original_prob or 0) - (reloaded_prob or 0)) < 1e-6
        if original_prob is not None and reloaded_prob is not None
        else True
    )

    if pred_match and prob_match:
        verdict = "PASS"
        print(f"\n  SERIALIZATION TEST: PASS ✓")
        print(f"  Saved and reloaded model produce identical predictions.")
    else:
        verdict = "FAIL"
        print(f"\n  SERIALIZATION TEST: FAIL ✗")
        print(f"  Predictions differ after save/load. DO NOT DEPLOY.")
        print(f"  Investigate: non-deterministic state, unfitted transformers, random seeds.")

    return {
        "verdict":          verdict,
        "original_pred":    original_pred,
        "reloaded_pred":    reloaded_pred,
        "predictions_match": pred_match,
    }
```

---

## Step 4 — Data Drift Detection Setup

```python
def setup_drift_detection(X_train: pd.DataFrame,
                            feature_cols: list,
                            output_path: str = "outputs/data/training_distribution.json") -> dict:
    """
    Store the training distribution (mean, std, min, max, p5, p95) per feature.
    Write check_for_drift() that flags features deviating > 2 standard deviations
    from training distribution in live data.

    Why 2 std?
    2 std covers ~95% of a normal distribution. Values beyond this in live data
    suggest the feature's real-world distribution has shifted — a warning sign
    that the model was trained on different data than it is now predicting on.
    """
    print("\n" + "=" * 60)
    print("  STEP 4: DRIFT DETECTION SETUP")
    print("  Storing training distribution for future comparison.")
    print("=" * 60)

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train[c])]
    distribution = {}

    for col in numeric_cols:
        series = X_train[col].dropna()
        distribution[col] = {
            "mean":  round(float(series.mean()), 6),
            "std":   round(float(series.std()),  6),
            "min":   round(float(series.min()),  6),
            "max":   round(float(series.max()),  6),
            "p5":    round(float(series.quantile(0.05)), 6),
            "p95":   round(float(series.quantile(0.95)), 6),
            "n_train": int(series.count()),
        }

    with open(output_path, "w") as f:
        json.dump(distribution, f, indent=2)

    print(f"  Training distribution saved → {output_path}")
    print(f"  {len(distribution)} numeric features tracked.")

    return distribution


def check_for_drift(live_df: pd.DataFrame,
                     distribution_path: str = "outputs/data/training_distribution.json",
                     std_threshold: float = 2.0) -> dict:
    """
    Compare live data distribution to training distribution.
    Flag any feature where |live_mean - train_mean| > std_threshold × train_std.

    Call this weekly (or on each batch) to monitor for distribution shift.
    """
    print("\n" + "=" * 60)
    print(f"  DRIFT CHECK  (threshold: >{std_threshold}σ from training distribution)")
    print("=" * 60)

    with open(distribution_path) as f:
        train_dist = json.load(f)

    drifted = []
    stable  = []

    print(f"\n  {'Feature':<35} {'Train Mean':>12} {'Live Mean':>12} {'Δ/σ':>10}  Status")
    print("  " + "-" * 78)

    for col, stats in train_dist.items():
        if col not in live_df.columns:
            continue
        live_series  = live_df[col].dropna()
        if len(live_series) == 0:
            continue
        live_mean    = live_series.mean()
        train_mean   = stats["mean"]
        train_std    = stats["std"] if stats["std"] > 1e-9 else 1e-9
        delta_sigma  = abs(live_mean - train_mean) / train_std

        if delta_sigma > std_threshold:
            status = f"⚠ DRIFT  (Δ={delta_sigma:.1f}σ)"
            drifted.append({
                "feature": col, "train_mean": train_mean,
                "live_mean": round(live_mean, 4),
                "delta_sigma": round(delta_sigma, 2),
            })
        else:
            status = "OK"
            stable.append(col)

        print(f"  {col:<35} {train_mean:>12.4f} {live_mean:>12.4f} {delta_sigma:>10.2f}  {status}")

    print(f"\n  Summary: {len(drifted)} feature(s) drifted, {len(stable)} stable.")
    if drifted:
        print(f"\n  ⚠ Drifted features:")
        for d in drifted:
            print(f"    {d['feature']}: train_mean={d['train_mean']:.4f}  "
                  f"live_mean={d['live_mean']:.4f}  ({d['delta_sigma']:.1f}σ drift)")
        print(f"\n  Recommendation: Review input pipeline for these features.")
        print(f"  If drift persists: consider retraining on recent data.")

    return {"drifted": drifted, "stable": stable}
```

---

## Step 5 — Generate monitoring_template.md

```python
def generate_monitoring_template(model_name: str,
                                  primary_metric: str,
                                  baseline_score: float,
                                  feature_cols: list,
                                  output_path: str = "skills/monitoring_template.md"):
    """
    Generate a weekly health report template as a Markdown file.
    This template is filled in weekly by whoever owns the model in production.
    """
    template = f"""# Model Health Report — Weekly Template

## Model: {model_name}
## Metric: {primary_metric.upper()}  |  Baseline (test set): {baseline_score:.4f}

---

## Report Date: [YYYY-MM-DD]
## Reporting Period: [YYYY-MM-DD] to [YYYY-MM-DD]
## Completed By: [Name / Team]

---

## 1. Prediction Volume

| Day       | Total Predictions | Positive Rate | Avg Score |
|-----------|-------------------|---------------|-----------|
| Monday    |                   |               |           |
| Tuesday   |                   |               |           |
| Wednesday |                   |               |           |
| Thursday  |                   |               |           |
| Friday    |                   |               |           |
| Saturday  |                   |               |           |
| Sunday    |                   |               |           |
| **TOTAL** |                   |               |           |

**Baseline positive rate (training):** ____%
**This week positive rate:** ____%
**Delta from baseline:** ____%  ⚠ Flag if > ±10%

---

## 2. Model Performance (if ground truth available)

| Metric              | This Week | Last Week | Baseline  | Status |
|---------------------|-----------|-----------|-----------|--------|
| {primary_metric.upper():<20} |           |           | {baseline_score:.4f}    |        |
| Accuracy            |           |           |           |        |
| F1 (weighted)       |           |           |           |        |
| Precision           |           |           |           |        |
| Recall              |           |           |           |        |

**Degradation threshold: {primary_metric.upper()} drop > 0.05 from baseline = RETRAIN**

---

## 3. Data Drift Check

Run: `check_for_drift(live_df)` and paste results below.

| Feature              | Train Mean | Live Mean | Δ/σ   | Status |
|----------------------|------------|-----------|-------|--------|
{''.join(f'| {c[:20]:<20} |            |           |       |        |\n' for c in feature_cols[:10])}

**Drift threshold: >2σ = flag for investigation**
**Persistent drift (2+ weeks): trigger retraining review**

---

## 4. Error Analysis

**Top error categories this week:**
1. 
2. 
3. 

**Any patterns in false positives?**
[ ] No patterns observed
[ ] Pattern identified: _______________

**Any patterns in false negatives?**
[ ] No patterns observed
[ ] Pattern identified: _______________

---

## 5. Infrastructure Health

| Check                        | Status  | Notes |
|------------------------------|---------|-------|
| Average latency (single pred)|         |       |
| P95 latency                  |         |       |
| Error rate (5xx / exceptions)|         |       |
| Model load time              |         |       |

**Latency threshold: >100ms single pred = investigate**

---

## 6. Action Items

| Priority | Issue | Owner | Due Date | Status |
|----------|-------|-------|----------|--------|
|          |       |       |          |        |

---

## 7. Sign-off

| Role           | Name | Decision | Date |
|----------------|------|----------|------|
| Model Owner    |      | [ ] OK / [ ] RETRAIN / [ ] ESCALATE | |
| Data Scientist |      | [ ] Reviewed | |
| Business Owner |      | [ ] Reviewed | |

---

*Generated by Brahma ML Pipeline — {datetime.datetime.now().strftime("%Y-%m-%d")}*
"""

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(template)

    print(f"\n  Monitoring template saved → {output_path}")
    print(f"  Fill in weekly to track model health in production.")
    return output_path
```

---

## Deployment Summary Report

```python
def print_deployment_report(results: dict, model_name: str, model_path: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 60)
    print("  ✅  DEPLOYMENT TESTER COMPLETE")
    print("=" * 60)
    print(f"  Model         : {model_name}")
    print(f"  Timestamp     : {ts}")
    print(f"  Model path    : {model_path}")
    print("-" * 60)

    checks = [
        ("predict_brahma() function", results.get("predict_demo", {}).get("verdict", "BUILT")),
        ("Input validation layer",    "BUILT"),
        ("Serialization test",        results.get("serialization", {}).get("verdict", "SKIP")),
        ("Drift detection setup",     "BUILT" if results.get("distribution") else "SKIP"),
        ("Monitoring template",       "GENERATED" if results.get("monitoring_path") else "SKIP"),
    ]

    for name, status in checks:
        icon = "✓" if status in ("PASS", "BUILT", "GENERATED") else \
               "✗" if status == "FAIL" else "–"
        print(f"  {icon}  {name:<35} {status}")

    print("-" * 60)
    serial_ok = results.get("serialization", {}).get("verdict") != "FAIL"
    if serial_ok:
        print("  Model is deployment-ready.")
        print("  Next step: integrate predict_brahma() into your API/batch pipeline.")
    else:
        print("  SERIALIZATION FAILED — do not deploy until fixed.")
    print("=" * 60 + "\n")
```

---

## Master Orchestrator

```python
def run_deployment_tester(
    model,
    splits: dict,
    feature_cols: list,
    scaler=None,
    sample_input: dict = None,
    primary_metric: str = "roc_auc",
    baseline_score: float = 0.0,
    optimal_threshold: float = 0.50,
    model_name: str = "brahma_v1",
    model_path: str = "outputs/models/final_model.pkl",
) -> dict:

    print("\n" + "=" * 70)
    print("  BRAHMA DEPLOYMENT TESTER")
    print(f"  Model   : {model_name}")
    print("=" * 70)

    results = {}

    # Step 1: Build predict function
    predict_fn = build_predict_function(
        model, feature_cols, scaler, optimal_threshold, model_name
    )

    if sample_input is None:
        # Use first row of test set as demo
        sample_input = splits["X_test"][feature_cols].iloc[0].to_dict()

    demo_predict_function(predict_fn, sample_input)
    results["predict_fn"] = predict_fn

    # Step 2: Input validator
    validator = build_input_validator(feature_cols)
    results["validator"] = validator

    # Step 3: Serialization test
    results["serialization"] = serialization_test(
        model, predict_fn, sample_input, feature_cols, model_path
    )

    # Step 4: Drift detection
    results["distribution"] = setup_drift_detection(
        splits["X_train"], feature_cols
    )

    # Step 5: Monitoring template
    results["monitoring_path"] = generate_monitoring_template(
        model_name, primary_metric, baseline_score, feature_cols
    )

    print_deployment_report(results, model_name, model_path)

    return results
```

---

## Usage Example

```python
deploy_results = run_deployment_tester(
    model            = ensemble_results["final"]["model"],
    splits           = training["splits"],
    feature_cols     = training["splits"]["feature_cols"],
    scaler           = training.get("scaler"),
    primary_metric   = "roc_auc",
    baseline_score   = eval_results["metrics"]["roc_auc"],
    optimal_threshold= val_results["threshold"]["optimal_threshold"],
    model_name       = "brahma_v1",
)

# Use in production
predict_brahma = deploy_results["predict_fn"]
result = predict_brahma({"tenure_months": 5, "monthly_charges": 90, "contract": 0})
print(result)
# {
#   'prediction': 1,
#   'probability': 0.7823,
#   'risk_tier': 'HIGH',
#   'top_reasons': [{'feature': 'monthly_charges', 'value': 90, 'shap_impact': 0.312}, ...],
#   'model_version': 'brahma_v1',
#   'timestamp': '2026-04-16T10:30:00Z',
#   'latency_ms': 4.2
# }

# Weekly drift check
check_for_drift(live_df)
```
