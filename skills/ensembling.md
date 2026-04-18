# Skill: Ensembling

## Purpose
Combine trained models to produce predictions more robust than any individual model.
The whole is greater than the sum of its parts — but only if the parts are diverse.
Correlated models produce correlated errors. Diverse models produce complementary errors.

**Depends on:** `skills/visualization_style.md`

---

## Core Rule — Diversity Before Combination

Before ensembling, always check prediction correlation.
Two models that make the same mistakes cancel no errors when combined.
The maximum ensemble gain comes from models that disagree on hard cases.

---

## Standard Import Block

```python
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "skills")

from visualization_style import (
    apply_brahma_style, new_figure, annotate_chart,
    save_chart, BRAHMA_COLORS, BRAHMA_PALETTE,
)
apply_brahma_style()

import matplotlib.ticker as ticker

OUTPUT_DIR = "outputs/charts/ensembling"
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

---

## Step 1 — Prediction Correlation Check

```python
def check_prediction_correlation(models: dict,
                                   X_val: pd.DataFrame,
                                   y_val: pd.Series,
                                   problem_type: str) -> pd.DataFrame:
    """
    Compute pairwise correlation between model predictions on the validation set.

    If r > 0.95 between two models:
        → Flag: ensembling these two adds little value.
        → Reason: they agree on nearly every prediction, so their errors are aligned.
                  The ensemble inherits rather than cancels those shared errors.

    Returns a correlation matrix of predictions.
    """
    print("\n" + "=" * 60)
    print("  STEP 1: PREDICTION CORRELATION CHECK")
    print("  Rule: Diverse algorithms give the most ensemble gain.")
    print("        Correlated predictions = correlated errors = no gain.")
    print("=" * 60)

    pred_matrix = {}

    for name, model in models.items():
        if "classification" in problem_type and hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_val)
            if preds.ndim > 1 and preds.shape[1] == 2:
                preds = preds[:, 1]
        else:
            preds = model.predict(X_val).astype(float)
        pred_matrix[name] = preds

    pred_df  = pd.DataFrame(pred_matrix)
    corr_mat = pred_df.corr(method="pearson")

    print(f"\n  {'Model A':<30} {'Model B':<30} {'r':>8}  Verdict")
    print("  " + "-" * 80)

    flagged_pairs = []
    n = len(models)
    model_names = list(models.keys())

    for i in range(n):
        for j in range(i + 1, n):
            a, b = model_names[i], model_names[j]
            r    = corr_mat.loc[a, b]
            if r > 0.95:
                verdict = "⚠ HIGH CORRELATION — ensemble gain will be minimal"
                flagged_pairs.append((a, b, r))
            elif r > 0.85:
                verdict = "MODERATE — some diversity, moderate gain expected"
            else:
                verdict = "✓ DIVERSE — good ensemble candidate"
            print(f"  {a:<30} {b:<30} {r:>8.4f}  {verdict}")

    if flagged_pairs:
        print(f"\n  ⚠ {len(flagged_pairs)} highly correlated pair(s):")
        for a, b, r in flagged_pairs:
            print(f"    {a} ↔ {b}  r={r:.4f}")
            print(f"    Consider dropping one — keeping both inflates compute without improving accuracy.")
    else:
        print(f"\n  All model pairs are sufficiently diverse (r ≤ 0.95).")
        print(f"  Proceed with ensembling — diversity is present.")

    return corr_mat
```

---

## Step 2 — Hard Voting (Classification)

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def hard_voting(models: dict,
                 X_val: pd.DataFrame,
                 y_val: pd.Series,
                 problem_type: str) -> dict:
    """
    Majority class vote across all models.
    Each model casts one vote. Plurality wins.
    Simplest ensemble — no probability estimates required.
    """
    print("\n" + "=" * 60)
    print("  STEP 2: HARD VOTING (Majority Class Vote)")
    print("  Each model casts one vote. Plurality wins.")
    print("=" * 60)

    all_preds = np.array([m.predict(X_val) for m in models.values()])

    from scipy import stats as sp_stats
    voted, _ = sp_stats.mode(all_preds, axis=0, keepdims=False)
    voted    = voted.flatten()

    scores = {
        "accuracy":    round(accuracy_score(y_val, voted), 4),
        "f1_weighted": round(f1_score(y_val, voted, average="weighted", zero_division=0), 4),
    }

    print(f"  Models voting : {list(models.keys())}")
    print(f"  Val Accuracy  : {scores['accuracy']}")
    print(f"  Val F1 (wtd)  : {scores['f1_weighted']}")

    return {"predictions": voted, "scores": scores, "method": "hard_voting"}
```

---

## Step 3 — Soft Voting (Average Probabilities — Preferred)

```python
def soft_voting(models: dict,
                 X_val: pd.DataFrame,
                 y_val: pd.Series,
                 problem_type: str) -> dict:
    """
    Average probability outputs across all models, then argmax.
    Preferred over hard voting — uses confidence, not just direction.
    Models with higher certainty carry proportionally more weight.

    Requires all models to expose predict_proba().
    """
    print("\n" + "=" * 60)
    print("  STEP 3: SOFT VOTING (Average Probabilities)")
    print("  Preferred over hard voting — uses model confidence, not just votes.")
    print("=" * 60)

    prob_stack  = []
    skip_models = []

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            prob_stack.append(model.predict_proba(X_val))
        else:
            skip_models.append(name)
            print(f"  ⚠ {name} has no predict_proba — excluded from soft voting.")

    if not prob_stack:
        print("  No models support predict_proba. Falling back to hard voting.")
        return hard_voting(models, X_val, y_val, problem_type)

    avg_probs = np.mean(np.array(prob_stack), axis=0)
    voted     = np.argmax(avg_probs, axis=1)

    scores = {
        "accuracy":    round(accuracy_score(y_val, voted), 4),
        "f1_weighted": round(f1_score(y_val, voted, average="weighted", zero_division=0), 4),
    }
    if "binary" in problem_type:
        try:
            scores["roc_auc"] = round(roc_auc_score(y_val, avg_probs[:, 1]), 4)
        except Exception:
            pass

    print(f"  Models included : {[n for n in models if n not in skip_models]}")
    if skip_models:
        print(f"  Models skipped  : {skip_models}")
    print(f"  Val Accuracy    : {scores['accuracy']}")
    print(f"  Val F1 (wtd)    : {scores['f1_weighted']}")
    if "roc_auc" in scores:
        print(f"  Val AUC-ROC     : {scores['roc_auc']}")

    return {
        "predictions":  voted,
        "probabilities": avg_probs,
        "scores":        scores,
        "method":        "soft_voting",
    }
```

---

## Step 4 — Stacking (Meta-Learner)

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict

def stacking(models: dict,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              X_val: pd.DataFrame,
              y_val: pd.Series,
              problem_type: str) -> dict:
    """
    Level 0: All trained models generate out-of-fold predictions on X_train.
    Level 1: Logistic Regression (classification) or Ridge (regression)
             meta-learner trained on those OOF predictions.

    Out-of-fold predictions prevent the meta-learner from overfitting
    to models that have already seen the training data.

    Meta-learner coefficients reveal which Level 0 models the stacker trusts most.
    """
    print("\n" + "=" * 60)
    print("  STEP 4: STACKING (Meta-Learner)")
    print("  Level 0: All models → OOF predictions")
    print("  Level 1: Logistic Regression meta-learner")
    print("  OOF prevents meta-learner overfitting to already-seen data.")
    print("=" * 60)

    is_clf = "classification" in problem_type
    n_splits = 5

    cv = (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
          if is_clf else
          KFold(n_splits=n_splits, shuffle=True, random_state=42))

    # ── Level 0: Generate OOF predictions ────────────────────────────────────
    oof_train = {}
    oof_val   = {}

    for name, model in models.items():
        print(f"\n  Generating OOF predictions for: {name}")
        use_proba = is_clf and hasattr(model, "predict_proba")

        if use_proba:
            oof = cross_val_predict(model, X_train, y_train,
                                    cv=cv, method="predict_proba", n_jobs=-1)
            # Binary: keep positive class probability only
            if oof.shape[1] == 2:
                oof = oof[:, 1].reshape(-1, 1)
            val_pred = model.predict_proba(X_val)
            if val_pred.shape[1] == 2:
                val_pred = val_pred[:, 1].reshape(-1, 1)
        else:
            oof      = cross_val_predict(model, X_train, y_train,
                                          cv=cv, n_jobs=-1).reshape(-1, 1)
            val_pred = model.predict(X_val).reshape(-1, 1)

        oof_train[name] = oof
        oof_val[name]   = val_pred
        print(f"    OOF shape: {oof.shape}  Val shape: {val_pred.shape}")

    # Stack OOF predictions into meta-features
    meta_X_train = np.hstack(list(oof_train.values()))
    meta_X_val   = np.hstack(list(oof_val.values()))
    meta_cols    = list(models.keys())

    print(f"\n  Meta-feature matrix: {meta_X_train.shape}  ({len(meta_cols)} models)")

    # ── Level 1: Meta-learner ─────────────────────────────────────────────────
    if is_clf:
        meta_learner = LogisticRegression(max_iter=1000, C=1.0,
                                           class_weight="balanced", random_state=42)
    else:
        meta_learner = Ridge(alpha=1.0)

    meta_learner.fit(meta_X_train, y_train)

    # ── Meta-learner coefficients (trust per model) ───────────────────────────
    print(f"\n  Meta-learner coefficients (how much the stacker trusts each model):")
    print(f"  {'Model':<35} {'Coefficient':>12}  Interpretation")
    print("  " + "-" * 70)

    if hasattr(meta_learner, "coef_"):
        coefs = meta_learner.coef_.flatten()
        for name, coef in zip(meta_cols, coefs):
            abs_coef = abs(coef)
            interp = (
                "HIGH TRUST"    if abs_coef > 1.0 else
                "MODERATE"      if abs_coef > 0.3 else
                "LOW TRUST"
            )
            direction = "positive" if coef > 0 else "negative"
            print(f"  {name:<35} {coef:>12.4f}  {interp} ({direction})")
        print(f"\n  Note: Negative coefficient = this model's confidence is inversely")
        print(f"        weighted. May indicate this model's scale inverts the meta-signal.")

    # ── Evaluate stacking ────────────────────────────────────────────────────
    if is_clf and hasattr(meta_learner, "predict_proba"):
        stacked_probs = meta_learner.predict_proba(meta_X_val)
        stacked_preds = np.argmax(stacked_probs, axis=1)
        if stacked_probs.shape[1] == 2:
            stacked_probs_pos = stacked_probs[:, 1]
        scores = {
            "accuracy":    round(accuracy_score(y_val, stacked_preds), 4),
            "f1_weighted": round(f1_score(y_val, stacked_preds,
                                           average="weighted", zero_division=0), 4),
        }
        if "binary" in problem_type:
            try:
                scores["roc_auc"] = round(roc_auc_score(y_val, stacked_probs_pos), 4)
            except Exception:
                pass
    else:
        stacked_preds = meta_learner.predict(meta_X_val)
        from sklearn.metrics import mean_squared_error, r2_score
        scores = {
            "rmse": round(np.sqrt(mean_squared_error(y_val, stacked_preds)), 4),
            "r2":   round(r2_score(y_val, stacked_preds), 4),
        }

    print(f"\n  Stacking Val Scores:")
    for metric, val in scores.items():
        print(f"    {metric:<20} {val}")

    path = os.path.join("outputs/models", "ensemble_stacker.pkl")
    joblib.dump(meta_learner, path)
    print(f"\n  Meta-learner saved → {path}")

    return {
        "meta_learner":  meta_learner,
        "predictions":   stacked_preds,
        "scores":        scores,
        "method":        "stacking",
        "meta_cols":     meta_cols,
    }
```

---

## Step 5 — Weighted Average (Regression)

```python
def weighted_average_regression(models: dict,
                                  rmse_scores: dict,
                                  X_val: pd.DataFrame,
                                  y_val: pd.Series) -> dict:
    """
    Weighted average ensemble for regression.
    Weights = 1 / RMSE per model.
    Better models (lower RMSE) receive higher weight automatically.

    Why 1/RMSE?
    A model with RMSE=5 has 2× the weight of a model with RMSE=10.
    This is principled: better accuracy = more say in the final prediction.
    """
    print("\n" + "=" * 60)
    print("  STEP 5: WEIGHTED AVERAGE REGRESSION (weights = 1/RMSE)")
    print("  Better models receive proportionally higher weight.")
    print("=" * 60)

    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    weights = {}
    preds   = {}

    print(f"\n  {'Model':<35} {'RMSE':>10} {'Weight':>10}")
    print("  " + "-" * 58)

    for name, model in models.items():
        rmse = rmse_scores.get(name)
        if rmse is None:
            y_pred_m = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred_m))
        weight = 1.0 / max(rmse, 1e-9)
        weights[name] = weight
        preds[name]   = model.predict(X_val)
        print(f"  {name:<35} {rmse:>10.4f} {weight:>10.4f}")

    total_weight = sum(weights.values())
    weighted_pred = sum(
        preds[n] * (weights[n] / total_weight)
        for n in models
    )

    # Normalised weights
    print(f"\n  Normalised weights:")
    for name in models:
        print(f"    {name:<35} {weights[name]/total_weight:.4f} ({weights[name]/total_weight*100:.1f}%)")

    scores = {
        "rmse": round(np.sqrt(mean_squared_error(y_val, weighted_pred)), 4),
        "mae":  round(mean_absolute_error(y_val, weighted_pred), 4),
        "r2":   round(r2_score(y_val, weighted_pred), 4),
    }

    print(f"\n  Weighted Ensemble Val Scores:")
    for metric, val in scores.items():
        print(f"    {metric:<20} {val}")

    return {
        "predictions": weighted_pred,
        "scores":      scores,
        "weights":     {n: round(w / total_weight, 4) for n, w in weights.items()},
        "method":      "weighted_average",
    }
```

---

## Step 6 — Ensemble Comparison Chart

```python
def plot_ensemble_comparison(all_scores: dict,
                              best_individual_name: str,
                              best_individual_score: float,
                              primary_metric: str,
                              dataset_name: str = "dataset") -> str:
    """
    Horizontal bar chart comparing all models + ensemble variants.
    Dashed vertical line marks best single model.
    Title answers: does ensembling beat any individual model?
    """
    # Sort by score descending
    sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=False)
    names  = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    # Color: ensemble methods in highlight, individual in muted, best individual in primary
    colors = []
    for n in names:
        if any(kw in n.lower() for kw in ["voting", "stacking", "weighted", "ensemble"]):
            colors.append(BRAHMA_COLORS["highlight"])
        elif n == best_individual_name:
            colors.append(BRAHMA_COLORS["primary"])
        else:
            colors.append(BRAHMA_COLORS["muted"])

    fig, ax = new_figure(figsize=(14, max(8, len(names) * 0.65 + 2)))

    bars = ax.barh(names, scores, color=colors, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, scores):
        ax.text(bar.get_width() + max(scores) * 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=10,
                color=BRAHMA_COLORS["dark"], fontweight="bold")

    # Best single model reference line
    ax.axvline(best_individual_score,
               color=BRAHMA_COLORS["primary"], linewidth=2,
               linestyle="--", alpha=0.8,
               label=f"Best single model: {best_individual_name} ({best_individual_score:.4f})")
    ax.legend(fontsize=10, loc="lower right")

    ax.set_xlim(min(scores) * 0.97, max(scores) * 1.06)

    # Finding-first title
    best_ensemble_score = max(
        v for k, v in all_scores.items()
        if any(kw in k.lower() for kw in ["voting", "stacking", "weighted", "ensemble"])
    ) if any(
        kw in k.lower() for k in all_scores
        for kw in ["voting", "stacking", "weighted", "ensemble"]
    ) else None

    if best_ensemble_score and best_ensemble_score > best_individual_score + 0.005:
        gain = best_ensemble_score - best_individual_score
        finding = (f"Ensembling Adds +{gain:.4f} {primary_metric.upper()} — "
                   f"Worth the Complexity")
    elif best_ensemble_score and best_ensemble_score > best_individual_score:
        finding = (f"Ensemble Marginally Better (+{best_ensemble_score - best_individual_score:.4f}) — "
                   f"Occam's Razor: Use Simpler Model")
    else:
        finding = f"Best Single Model Wins — Ensembling Adds No Value Here"

    annotate_chart(ax,
        title=f"Does Combining Models Beat Any Individual Model? {finding}",
        subtitle=(f"Blue = best individual  |  Red = ensemble variants  |  "
                  f"Metric: {primary_metric}  |  Val set"),
        xlabel=primary_metric.upper().replace("_", " "),
        ylabel="Model",
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, "ensemble_comparison.png")
    save_chart(fig, path)
    return path
```

---

## Step 7 — Final Model Selection (Occam's Razor)

```python
def select_final_model(all_results: dict,
                        models: dict,
                        primary_metric: str,
                        higher_is_better: bool = True) -> dict:
    """
    Apply Occam's Razor: if ensemble beats best individual by < 0.005,
    choose the simpler model.

    Complexity costs: stacking > soft_voting > hard_voting > individual.
    Simpler wins when performance is equal. Always.

    Returns: {'name': ..., 'model': ..., 'score': ..., 'reason': ...}
    """
    print("\n" + "=" * 60)
    print("  STEP 7: FINAL MODEL SELECTION (Occam's Razor)")
    print("  Simpler wins when performance is equal.")
    print("=" * 60)

    # Rank all results
    ranked = sorted(
        all_results.items(),
        key=lambda x: x[1]["scores"].get(primary_metric, 0),
        reverse=higher_is_better,
    )

    # Find best individual model
    individual_names = [k for k in all_results if all_results[k].get("type") == "individual"]
    ensemble_names   = [k for k in all_results if all_results[k].get("type") == "ensemble"]

    if not individual_names:
        print("  No individual model scores provided. Returning top result.")
        best = ranked[0]
        return {"name": best[0], "score": best[1]["scores"].get(primary_metric), "reason": "top ranked"}

    best_individual = max(individual_names,
                          key=lambda k: all_results[k]["scores"].get(primary_metric, 0))
    best_ind_score  = all_results[best_individual]["scores"].get(primary_metric, 0)

    best_ensemble = None
    best_ens_score = None
    if ensemble_names:
        best_ensemble  = max(ensemble_names,
                             key=lambda k: all_results[k]["scores"].get(primary_metric, 0))
        best_ens_score = all_results[best_ensemble]["scores"].get(primary_metric, 0)

    print(f"\n  Best individual   : {best_individual:<35} {primary_metric}={best_ind_score:.4f}")
    if best_ensemble:
        print(f"  Best ensemble     : {best_ensemble:<35} {primary_metric}={best_ens_score:.4f}")

    OCCAM_THRESHOLD = 0.005

    if best_ensemble and best_ens_score and (best_ens_score - best_ind_score) > OCCAM_THRESHOLD:
        final_name  = best_ensemble
        final_score = best_ens_score
        reason = (f"Ensemble beats best individual by {best_ens_score - best_ind_score:.4f} "
                  f"(> {OCCAM_THRESHOLD} threshold) — complexity justified.")
        print(f"\n  DECISION: USE ENSEMBLE — {best_ensemble}")
    else:
        final_name  = best_individual
        final_score = best_ind_score
        if best_ensemble:
            margin = (best_ens_score or 0) - best_ind_score
            reason = (f"Ensemble gain = {margin:.4f} (< {OCCAM_THRESHOLD} Occam threshold). "
                      f"Simpler model chosen — {best_individual}.")
        else:
            reason = f"No ensemble available. Best individual: {best_individual}."
        print(f"\n  DECISION: USE INDIVIDUAL MODEL — {best_individual}")
        print(f"  Occam's Razor: ensemble gain below {OCCAM_THRESHOLD} threshold.")
        print(f"  Simpler model is easier to explain, maintain, and debug.")

    print(f"\n  Reason    : {reason}")
    print(f"  Final score ({primary_metric}): {final_score:.4f}")

    final_model = all_results[final_name].get("model") or models.get(final_name)
    if final_model:
        path = "outputs/models/final_model.pkl"
        os.makedirs("outputs/models", exist_ok=True)
        joblib.dump(final_model, path)
        print(f"  Saved as  : {path}")

    return {
        "name":  final_name,
        "model": final_model,
        "score": final_score,
        "reason": reason,
    }
```

---

## Master Orchestrator

```python
def run_ensembling(
    models: dict,
    splits: dict,
    problem_type: str,
    individual_scores: dict,
    rmse_scores: dict = None,
    dataset_name: str = "dataset",
) -> dict:
    """
    Full ensembling pipeline:
    1. Diversity check
    2. Hard voting
    3. Soft voting (preferred)
    4. Stacking (if voting doesn't beat best individual)
    5. Weighted average (regression)
    6. Comparison chart
    7. Final model selection

    individual_scores: {model_name: {metric: value}}
    rmse_scores: {model_name: rmse_value}  (regression only)
    """
    X_train = splits["X_train"]
    X_val   = splits["X_val"]
    y_train = splits["y_train"]
    y_val   = splits["y_val"]

    is_clf         = "classification" in problem_type
    primary_metric = ("roc_auc"     if "binary"         in problem_type else
                      "f1_weighted" if "classification" in problem_type else "r2")
    higher_is_better = primary_metric != "rmse"

    print("\n" + "=" * 70)
    print("  BRAHMA ENSEMBLING")
    print(f"  Models     : {list(models.keys())}")
    print(f"  Problem    : {problem_type}")
    print(f"  Metric     : {primary_metric}")
    print("=" * 70)

    all_results = {}

    # Register individual model scores
    for name, score_dict in individual_scores.items():
        all_results[name] = {"scores": score_dict, "type": "individual",
                              "model": models.get(name)}

    # Step 1: Diversity check
    corr_mat = check_prediction_correlation(models, X_val, y_val, problem_type)

    if is_clf:
        # Step 2: Hard voting
        hard_result = hard_voting(models, X_val, y_val, problem_type)
        all_results["Hard Voting"] = {**hard_result, "type": "ensemble"}

        # Step 3: Soft voting
        soft_result = soft_voting(models, X_val, y_val, problem_type)
        all_results["Soft Voting"] = {**soft_result, "type": "ensemble"}

        # Step 4: Stacking (if voting didn't beat best individual)
        best_ind_score = max(
            v["scores"].get(primary_metric, 0)
            for v in all_results.values()
            if v.get("type") == "individual"
        )
        best_vote_score = max(
            soft_result["scores"].get(primary_metric, 0),
            hard_result["scores"].get(primary_metric, 0),
        )

        if best_vote_score <= best_ind_score:
            print(f"\n  Voting ({best_vote_score:.4f}) did not beat best individual ({best_ind_score:.4f}).")
            print(f"  Proceeding to stacking...")
            stack_result = stacking(models, X_train, y_train, X_val, y_val, problem_type)
            all_results["Stacking"] = {**stack_result, "type": "ensemble"}
        else:
            print(f"\n  Soft voting ({best_vote_score:.4f}) beats best individual ({best_ind_score:.4f}).")
            print(f"  Skipping stacking — soft voting is sufficient.")
    else:
        # Step 5: Weighted average (regression)
        weighted_result = weighted_average_regression(models, rmse_scores or {}, X_val, y_val)
        all_results["Weighted Average"] = {**weighted_result, "type": "ensemble"}

    # Step 6: Comparison chart
    chart_scores = {
        name: res["scores"].get(primary_metric, 0)
        for name, res in all_results.items()
        if primary_metric in res.get("scores", {})
    }
    best_individual_name  = max(
        (k for k in all_results if all_results[k].get("type") == "individual"),
        key=lambda k: all_results[k]["scores"].get(primary_metric, 0),
    )
    best_individual_score = all_results[best_individual_name]["scores"].get(primary_metric, 0)

    plot_ensemble_comparison(
        chart_scores, best_individual_name, best_individual_score,
        primary_metric, dataset_name,
    )

    # Step 7: Final model selection
    final = select_final_model(all_results, models, primary_metric, higher_is_better)

    print("\n" + "=" * 70)
    print(f"  ENSEMBLING COMPLETE")
    print(f"  Final model : {final['name']}")
    print(f"  Score       : {final['score']:.4f}  ({primary_metric})")
    print(f"  Reason      : {final['reason']}")
    print(f"  Brahma is ready for UAT.\n")

    return {
        "all_results":    all_results,
        "final":          final,
        "corr_matrix":    corr_mat,
        "primary_metric": primary_metric,
    }
```

---

## Usage Example

```python
# After model_trainer.md produces trained models:
ensemble_results = run_ensembling(
    models={
        "XGBoost (tuned)":  training["tuned_result"]["model"],
        "Random Forest":    training["baselines"]["RandomForestClassifier"]["model"],
        "Logistic Reg":     training["baselines"]["LogisticRegression"]["model"],
    },
    splits=training["splits"],
    problem_type=selection["problem_type"],
    individual_scores={
        "XGBoost (tuned)":  {"roc_auc": 0.891, "f1_weighted": 0.832},
        "Random Forest":    {"roc_auc": 0.874, "f1_weighted": 0.818},
        "Logistic Reg":     {"roc_auc": 0.801, "f1_weighted": 0.763},
    },
    dataset_name="telco_churn.csv",
)

final_model = ensemble_results["final"]["model"]
```
