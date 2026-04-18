import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import pickle, json, time, os, warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/data',   exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
with open('outputs/data/splits.pkl', 'rb') as f:
    splits = pickle.load(f)
with open('outputs/models/xgb_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

X_train = splits['X_train']; y_train = splits['y_train']
X_test  = splits['X_test'];  y_test  = splits['y_test']
feature_cols = splits['feature_cols']

MODEL_VERSION = "brahma-churn-v1.0.0"
print("="*60)
print(f"STAGE 11 — DEPLOYMENT TESTING")
print(f"Model version: {MODEL_VERSION}")
print("="*60)

# ── Build predict_brahma() function ───────────────────────────────────────────
print("\n[1/4] Building predict_brahma() function...")

def predict_brahma(input_data: dict) -> dict:
    """
    Production prediction function.

    Parameters
    ----------
    input_data : dict  —  feature values keyed by feature name

    Returns
    -------
    dict with keys:
        prediction   : int   (0=Retain, 1=Churn)
        probability  : float (churn probability)
        risk_tier    : str   (HIGH / MEDIUM / LOW)
        top_reasons  : list  (top 3 feature contributions)
        model_version: str
        timestamp    : str
        latency_ms   : float
    """
    t_start = time.perf_counter()

    # Build feature vector in correct order
    row = np.array([[input_data.get(f, 0.0) for f in feature_cols]], dtype=float)

    # Predict
    prob        = float(model.predict_proba(row)[0, 1])
    prediction  = int(prob >= 0.5)

    # Risk tier
    if prob > 0.70:
        risk_tier = "HIGH"
    elif prob > 0.40:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    # Top reasons (feature * coefficient approximation for interpretability)
    try:
        importances = model.feature_importances_
        feature_vals = row[0]
        contributions = np.abs(importances * feature_vals)
        top3_idx = contributions.argsort()[-3:][::-1]
        top_reasons = [
            {
                "feature": feature_cols[i],
                "importance": round(float(importances[i]), 4),
                "value": round(float(feature_vals[i]), 4)
            }
            for i in top3_idx
        ]
    except Exception:
        top_reasons = []

    latency_ms = round((time.perf_counter() - t_start) * 1000, 3)

    return {
        "prediction":    prediction,
        "probability":   round(prob, 4),
        "risk_tier":     risk_tier,
        "top_reasons":   top_reasons,
        "model_version": MODEL_VERSION,
        "timestamp":     pd.Timestamp.now().isoformat(),
        "latency_ms":    latency_ms
    }

# Test the function
sample_input = {f: float(X_test[0, i]) for i, f in enumerate(feature_cols)}
result = predict_brahma(sample_input)
print(f"\n  Sample prediction:")
print(f"  Input customer (first test row):")
print(f"    True label : {int(y_test[0])}")
print(f"  Output:")
for k, v in result.items():
    if k != 'top_reasons':
        print(f"    {k:<18} : {v}")
print(f"    top_reasons:")
for r in result['top_reasons']:
    print(f"      - {r['feature']:<35} importance={r['importance']}  value={r['value']}")

print(f"\n  predict_brahma() built successfully.")

# ── Build input validator ──────────────────────────────────────────────────────
print("\n[2/4] Building input validator...")

def validate_input(input_data: dict) -> dict:
    errors   = []
    warnings = []

    # Check all features present
    missing = [f for f in feature_cols if f not in input_data]
    if missing:
        errors.append(f"Missing features: {missing}")

    # Check for nulls
    nulls = [f for f, v in input_data.items() if v is None or (isinstance(v, float) and np.isnan(v))]
    if nulls:
        warnings.append(f"Null values in: {nulls} — will be filled with 0")

    # Check numeric
    non_numeric = [f for f, v in input_data.items()
                   if not isinstance(v, (int, float, np.integer, np.floating))]
    if non_numeric:
        errors.append(f"Non-numeric values: {non_numeric}")

    return {
        "valid":    len(errors) == 0,
        "errors":   errors,
        "warnings": warnings
    }

# Test validator
valid_result   = validate_input(sample_input)
invalid_result = validate_input({"total_trans_ct": 45})
print(f"  Valid input   → valid={valid_result['valid']}")
print(f"  Partial input → valid={invalid_result['valid']}, errors={invalid_result['errors'][:1]}")

# ── Setup drift detection ──────────────────────────────────────────────────────
print("\n[3/4] Setting up drift detection...")

# Save training distribution statistics
train_stats = {}
for i, col in enumerate(feature_cols):
    vals = X_train[:, i]
    train_stats[col] = {
        "mean":  float(np.mean(vals)),
        "std":   float(np.std(vals)),
        "min":   float(np.min(vals)),
        "max":   float(np.max(vals)),
        "p25":   float(np.percentile(vals, 25)),
        "p75":   float(np.percentile(vals, 75)),
    }

drift_config = {
    "model_version":   MODEL_VERSION,
    "training_date":   pd.Timestamp.now().isoformat(),
    "n_training_rows": int(len(X_train)),
    "features":        train_stats,
    "drift_threshold_sigma": 2.0,
    "alert_on":        ["total_trans_ct", "months_inactive_12_mon",
                        "contacts_count_12_mon", "re_engagement_signal"]
}

with open('outputs/data/drift_config.json', 'w') as f:
    json.dump(drift_config, f, indent=2)
print(f"  Saved drift config: outputs/data/drift_config.json")
print(f"  Monitoring {len(feature_cols)} features | Alert threshold: 2-sigma")
print(f"  Key features to watch: {drift_config['alert_on']}")

def check_for_drift(new_batch: np.ndarray) -> dict:
    alerts = []
    for i, col in enumerate(feature_cols):
        if col not in drift_config['alert_on']:
            continue
        new_mean = np.mean(new_batch[:, i])
        train_mu = train_stats[col]['mean']
        train_sd = train_stats[col]['std'] + 1e-9
        z_score  = abs(new_mean - train_mu) / train_sd
        if z_score > drift_config['drift_threshold_sigma']:
            alerts.append({
                "feature": col, "z_score": round(z_score, 2),
                "train_mean": round(train_mu, 4),
                "new_mean":   round(new_mean, 4)
            })
    return {"drift_detected": len(alerts) > 0, "alerts": alerts}

# Test drift check on test set (should be minimal drift)
drift_result = check_for_drift(X_test)
print(f"\n  Drift check on holdout test set:")
print(f"  Drift detected: {drift_result['drift_detected']}")
if drift_result['alerts']:
    for a in drift_result['alerts']:
        print(f"  ALERT: {a['feature']} z={a['z_score']:.2f}")
else:
    print(f"  All monitored features within 2-sigma of training distribution.")

# ── Batch prediction performance ──────────────────────────────────────────────
print("\n[4/4] Batch performance test (750 rows)...")
t0 = time.perf_counter()
batch_probs = model.predict_proba(X_test)[:, 1]
batch_time  = (time.perf_counter() - t0) * 1000
print(f"  750 predictions in {batch_time:.1f} ms  ({batch_time/750*1000:.1f} us/prediction)")
print(f"  Throughput: {750 / (batch_time/1000):,.0f} predictions/second")

# ── Save deployment package ────────────────────────────────────────────────────
deploy_pkg = {
    "model":          model,
    "predict_fn":     predict_brahma,
    "validate_fn":    validate_input,
    "drift_fn":       check_for_drift,
    "feature_cols":   feature_cols,
    "model_version":  MODEL_VERSION,
    "threshold":      0.50,
    "risk_tiers":     {"HIGH": 0.70, "MEDIUM": 0.40, "LOW": 0.0},
}
with open('outputs/models/deployment_package.pkl', 'wb') as f:
    pickle.dump(deploy_pkg, f)
print(f"\n  Saved: outputs/models/deployment_package.pkl")

# ── Run 10 example predictions ─────────────────────────────────────────────────
print("\n--- SAMPLE PREDICTIONS (10 test customers) ---")
print(f"\n  {'#':<4} {'True':>6} {'Prob':>7} {'Pred':>6} {'Risk':>8}")
print(f"  {'-'*35}")
for i in range(10):
    inp = {f: float(X_test[i, j]) for j, f in enumerate(feature_cols)}
    out = predict_brahma(inp)
    true_l = "CHURN"   if y_test[i] == 1 else "retain"
    pred_l = "CHURN"   if out['prediction'] == 1 else "retain"
    match  = "OK" if true_l.lower() == pred_l.lower() else "MISS"
    print(f"  {i+1:<4} {true_l:>6}  {out['probability']:>6.3f}  {pred_l:>6}  "
          f"{out['risk_tier']:>8}  [{match}]  {out['latency_ms']:.2f}ms")

print("\n" + "="*60)
print("STAGE 11 DEPLOYMENT TESTING COMPLETE")
print("="*60)
print(f"  predict_brahma()       : ready")
print(f"  validate_input()       : ready")
print(f"  check_for_drift()      : ready")
print(f"  Deployment package     : outputs/models/deployment_package.pkl")
print(f"  Drift config           : outputs/data/drift_config.json")
print(f"  Batch throughput       : {750 / (batch_time/1000):,.0f} predictions/sec")
print("="*60)
