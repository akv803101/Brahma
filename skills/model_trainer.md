# Skill: Model Trainer

## Purpose
Train every model with discipline: baseline first, defaults second, tuned third.
Training without tuning is amateur. Brahma always tunes.

**Depends on:** `skills/visualization_style.md`, `skills/algorithm_selector.md`

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

OUTPUT_MODELS  = "outputs/models"
OUTPUT_CHARTS  = "outputs/charts/training"
os.makedirs(OUTPUT_MODELS,  exist_ok=True)
os.makedirs(OUTPUT_CHARTS,  exist_ok=True)
```

---

## Step 1 — Train / Val / Test Split

```python
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame,
               target_col: str,
               problem_type: str = "binary_classification",
               random_state: int = 42) -> dict:
    """
    70 / 15 / 15 stratified split.
    Stratified for all classification tasks.
    Returns dict with X_train, X_val, X_test, y_train, y_val, y_test.
    """
    print("\n" + "=" * 60)
    print("  STEP 1: TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)")
    print("=" * 60)

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    stratify = y if "classification" in problem_type else None

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=stratify
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.15 / 0.85,   # ~15% of total
        random_state=random_state,
        stratify=y_train_val if stratify is not None else None,
    )

    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_cols": feature_cols,
    }

    print(f"  Train : {len(X_train):,} rows  ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val   : {len(X_val):,} rows  ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test  : {len(X_test):,} rows  ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Features: {len(feature_cols)}")
    if stratify is not None:
        for cls in sorted(y_train.unique()):
            pct = (y_train == cls).mean() * 100
            print(f"  Train class {cls}: {pct:.1f}%")
    print(f"  random_state = {random_state}  (reproducible)")

    return splits
```

---

## Step 2 — Train Baselines

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
)

def score_model(model, X_val, y_val, problem_type: str) -> dict:
    y_pred = model.predict(X_val)
    scores = {}
    if "classification" in problem_type:
        scores["accuracy"] = round(accuracy_score(y_val, y_pred), 4)
        scores["f1_weighted"] = round(f1_score(y_val, y_pred, average="weighted", zero_division=0), 4)
        if "binary" in problem_type and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
            scores["roc_auc"] = round(roc_auc_score(y_val, y_prob), 4)
    else:
        scores["rmse"] = round(np.sqrt(mean_squared_error(y_val, y_pred)), 4)
        scores["mae"]  = round(mean_absolute_error(y_val, y_pred), 4)
        scores["r2"]   = round(r2_score(y_val, y_pred), 4)
    return scores


def train_baselines(splits: dict, selection: dict) -> dict:
    """
    Always train Dummy + Logistic/Ridge baselines.
    Any model that cannot beat these is not worth shipping.
    """
    print("\n" + "=" * 60)
    print("  STEP 2: BASELINE MODELS")
    print("  Rule: No model is accepted unless it beats these baselines.")
    print("=" * 60)

    problem_type = selection["problem_type"]
    baseline_results = {}

    baseline_configs = {
        "DummyClassifier":   DummyClassifier(strategy="most_frequent"),
        "DummyRegressor":    DummyRegressor(strategy="mean"),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Ridge":             Ridge(alpha=1.0),
    }

    active_baselines = selection["baselines"]

    for name in active_baselines:
        model = baseline_configs[name]
        model.fit(splits["X_train"], splits["y_train"])
        scores = score_model(model, splits["X_val"], splits["y_val"], problem_type)
        baseline_results[name] = {"model": model, "scores": scores}

        path = os.path.join(OUTPUT_MODELS, f"baseline_{name}.pkl")
        joblib.dump(model, path)

        print(f"\n  [{name}]")
        for metric, val in scores.items():
            print(f"    {metric:<20} {val}")
        print(f"  Saved → {path}")

    return baseline_results
```

---

## Step 3 — Train Primary Model with Defaults

```python
def train_default_model(splits: dict, selection: dict) -> dict:
    """
    Train the selected primary algorithm with its default parameters.
    Record val score — this is the pre-tuning baseline for the primary model.
    """
    print("\n" + "=" * 60)
    print("  STEP 3: PRIMARY MODEL — DEFAULT PARAMETERS")
    print("=" * 60)

    algo   = selection["primary"]["algorithm"]
    params = selection["primary"]["params"].copy()
    problem_type = selection["problem_type"]

    # Handle scale_pos_weight for XGBoost imbalanced
    if selection["primary"].get("scale_pos_weight"):
        counts = splits["y_train"].value_counts()
        params["scale_pos_weight"] = counts.max() / counts.min()
        print(f"  scale_pos_weight set to {params['scale_pos_weight']:.2f}")

    model = _instantiate(algo, params)
    model.fit(splits["X_train"], splits["y_train"])
    scores = score_model(model, splits["X_val"], splits["y_val"], problem_type)

    path = os.path.join(OUTPUT_MODELS, f"primary_{algo}_default.pkl")
    joblib.dump(model, path)

    print(f"  Algorithm     : {algo}")
    print(f"  Parameters    : {params}")
    print(f"\n  Val Scores (default):")
    for metric, val in scores.items():
        print(f"    {metric:<20} {val}")
    print(f"  Saved → {path}")

    return {"model": model, "scores": scores, "algo": algo}


def _instantiate(algo: str, params: dict):
    """Dynamically instantiate a sklearn/xgboost/lightgbm model by class name."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge, HuberRegressor
    from sklearn.svm import SVC

    registry = {
        "RandomForestClassifier":  RandomForestClassifier,
        "RandomForestRegressor":   RandomForestRegressor,
        "LogisticRegression":      LogisticRegression,
        "Ridge":                   Ridge,
        "HuberRegressor":          HuberRegressor,
        "SVC":                     SVC,
    }

    try:
        from xgboost import XGBClassifier, XGBRegressor
        registry["XGBClassifier"] = XGBClassifier
        registry["XGBRegressor"]  = XGBRegressor
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        registry["LGBMClassifier"] = LGBMClassifier
        registry["LGBMRegressor"]  = LGBMRegressor
    except ImportError:
        pass

    cls = registry.get(algo)
    if cls is None:
        raise ValueError(f"Unknown algorithm: {algo}. Available: {list(registry.keys())}")

    # Remove params not accepted by this class
    import inspect
    valid_params = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in params.items() if k in valid_params}
    return cls(**filtered)
```

---

## Step 4 — Hyperparameter Tuning with Optuna

```python
def tune_model(splits: dict, selection: dict,
               n_trials_override: int = None) -> dict:
    """
    Optuna-based hyperparameter search.
    Trials: RandomForest=30, XGBoost/LightGBM=50, others=20.
    Optimises val score. Returns best model and best params.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed. Run: pip install optuna")
        print("  Falling back to default model.")
        return None

    print("\n" + "=" * 60)
    print("  STEP 4: HYPERPARAMETER TUNING (Optuna)")
    print("=" * 60)

    algo         = selection["primary"]["algorithm"]
    problem_type = selection["problem_type"]
    X_tr, y_tr   = splits["X_train"], splits["y_train"]
    X_vl, y_vl   = splits["X_val"],   splits["y_val"]

    is_clf = "classification" in problem_type

    def objective(trial):
        if "RandomForest" in algo:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
                "max_depth":         trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
                "random_state":      42, "n_jobs": -1,
            }
            if is_clf:
                params["class_weight"] = "balanced"
        elif "XGB" in algo:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
                "max_depth":         trial.suggest_int("max_depth", 3, 10),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state":      42, "n_jobs": -1,
                "eval_metric":       "logloss" if is_clf else "rmse",
            }
            if selection["primary"].get("scale_pos_weight"):
                counts = y_tr.value_counts()
                params["scale_pos_weight"] = counts.max() / counts.min()
        elif "LGBM" in algo:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
                "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
                "random_state":      42, "n_jobs": -1,
            }
            if is_clf:
                params["class_weight"] = "balanced"
        elif algo == "LogisticRegression":
            params = {
                "C":          trial.suggest_float("C", 1e-4, 100, log=True),
                "solver":     trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                "max_iter":   1000, "class_weight": "balanced", "random_state": 42,
            }
        elif algo == "Ridge":
            params = {"alpha": trial.suggest_float("alpha", 1e-3, 100, log=True)}
        else:
            params = {}

        model = _instantiate(algo, params)
        model.fit(X_tr, y_tr)
        scores = score_model(model, X_vl, y_vl, problem_type)

        # Return primary metric
        if is_clf:
            return scores.get("roc_auc", scores.get("f1_weighted", scores.get("accuracy", 0)))
        else:
            return -scores.get("rmse", 1e9)   # minimise RMSE → negate for Optuna

    # Trial counts
    default_trials = {"RandomForest": 30, "XGB": 50, "LGBM": 50}
    n_trials = n_trials_override or next(
        (v for k, v in default_trials.items() if k in algo), 20
    )

    print(f"  Algorithm  : {algo}")
    print(f"  Trials     : {n_trials}")

    direction = "maximize"
    study = optuna.create_study(direction=direction,
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    print(f"\n  Best params found:")
    for k, v in best_params.items():
        print(f"    {k:<30} {v}")

    # Merge with fixed params not in search space
    fixed = {k: v for k, v in selection["primary"]["params"].items()
             if k not in best_params}
    final_params = {**fixed, **best_params}

    best_model = _instantiate(algo, final_params)
    best_model.fit(X_tr, y_tr)
    best_scores = score_model(best_model, X_vl, y_vl, problem_type)

    path = os.path.join(OUTPUT_MODELS, f"primary_{algo}_tuned.pkl")
    joblib.dump(best_model, path)

    print(f"\n  Val Scores (tuned):")
    for metric, val in best_scores.items():
        print(f"    {metric:<20} {val}")
    print(f"  Saved → {path}")

    return {"model": best_model, "scores": best_scores, "params": final_params, "study": study}
```

---

## Step 5 — Training Charts

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, splits: dict, problem_type: str,
                         dataset_name: str = "dataset") -> str:
    """
    Learning curve: train vs val score as training set size grows.
    Reveals whether model needs more data or less complexity.
    """
    from sklearn.model_selection import StratifiedKFold, KFold

    X_tr = splits["X_train"]
    y_tr = splits["y_train"]

    cv = (StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
          if "classification" in problem_type
          else KFold(n_splits=5, shuffle=True, random_state=42))

    scoring = "roc_auc" if "binary" in problem_type else \
              "f1_weighted" if "classification" in problem_type else "r2"

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_tr, y_tr,
        cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = new_figure(figsize=(14, 9))

    ax.plot(train_sizes, train_mean, color=BRAHMA_COLORS["primary"],
            linewidth=2.5, label="Train score", marker="o", markersize=5)
    ax.fill_between(train_sizes,
                    train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=BRAHMA_COLORS["primary"])

    ax.plot(train_sizes, val_mean, color=BRAHMA_COLORS["highlight"],
            linewidth=2.5, label="CV val score", marker="s", markersize=5,
            linestyle="--")
    ax.fill_between(train_sizes,
                    val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color=BRAHMA_COLORS["highlight"])

    ax.legend(fontsize=11)

    gap        = train_mean[-1] - val_mean[-1]
    converging = (val_mean[-1] - val_mean[len(val_mean)//2]) > 0.01

    if gap > 0.08:
        finding = "Model Is Overfitting — Consider Regularization or More Data"
    elif not converging:
        finding = "Model Has High Bias — More Data Will Not Help, Try More Complexity"
    else:
        finding = "Model Is Learning Well — Performance Improves with More Data"

    annotate_chart(ax,
        title=f"Learning Curve: {finding}",
        subtitle=(f"Train–Val gap at full data: {gap:.3f}  |  "
                  f"Scoring: {scoring}  |  5-fold CV"),
        xlabel="Training Set Size",
        ylabel=scoring.upper().replace("_", " "),
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_CHARTS, "training_learning_curve.png")
    save_chart(fig, path)
    return path


def plot_boosting_curve(model, splits: dict,
                         problem_type: str,
                         dataset_name: str = "dataset") -> str:
    """
    For XGBoost and LightGBM: plot eval metric over boosting rounds.
    Reveals optimal n_estimators and signs of overfitting.
    Only called if model has evals_result_ or best_iteration attribute.
    """
    algo_name = type(model).__name__

    if not any(x in algo_name for x in ["XGB", "LGBM"]):
        return ""

    fig, ax = new_figure(figsize=(14, 9))

    try:
        if "XGB" in algo_name:
            results = model.evals_result()
            metric  = list(list(results.values())[0].keys())[0]
            for split_name, split_results in results.items():
                vals  = split_results[metric]
                color = BRAHMA_COLORS["primary"] if "train" in split_name \
                        else BRAHMA_COLORS["highlight"]
                ax.plot(vals, label=split_name, color=color, linewidth=2)
            best_round = model.best_iteration
            ax.axvline(best_round, color=BRAHMA_COLORS["warning"],
                       linestyle=":", linewidth=1.5,
                       label=f"Best round: {best_round}")

        elif "LGBM" in algo_name:
            results = model.evals_result_
            for split_name, metrics in results.items():
                for metric, vals in metrics.items():
                    color = BRAHMA_COLORS["primary"] if "train" in split_name \
                            else BRAHMA_COLORS["highlight"]
                    ax.plot(vals, label=f"{split_name}/{metric}",
                            color=color, linewidth=2)

    except AttributeError:
        print(f"  [SKIP] Boosting curve — model not trained with eval_set.")
        return ""

    ax.legend(fontsize=11)

    annotate_chart(ax,
        title=f"Boosting Curve: {algo_name} — Eval Metric Over Training Rounds",
        subtitle="Dashed line = best round  |  Diverging train/val = overfitting",
        xlabel="Boosting Round",
        ylabel="Eval Metric",
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_CHARTS, "training_boosting_curve.png")
    save_chart(fig, path)
    return path
```

---

## Step 6 — Save All Models

```python
def save_all_models(baselines: dict, default_result: dict,
                     tuned_result: dict) -> dict:
    """
    All models already saved during training.
    This step confirms the manifest and prints a summary.
    """
    print("\n" + "=" * 60)
    print("  STEP 6: MODEL MANIFEST")
    print("=" * 60)

    manifest = {}

    for name, res in baselines.items():
        path = os.path.join(OUTPUT_MODELS, f"baseline_{name}.pkl")
        manifest[name] = path
        print(f"  Baseline : {name:<30} → {path}")

    algo = default_result["algo"]
    path = os.path.join(OUTPUT_MODELS, f"primary_{algo}_default.pkl")
    manifest[f"{algo}_default"] = path
    print(f"  Default  : {algo}_default            → {path}")

    if tuned_result:
        path = os.path.join(OUTPUT_MODELS, f"primary_{algo}_tuned.pkl")
        manifest[f"{algo}_tuned"] = path
        print(f"  Tuned    : {algo}_tuned              → {path}")

    return manifest
```

---

## Step 7 — Leaderboard

```python
def print_leaderboard(baselines: dict, default_result: dict,
                       tuned_result: dict, problem_type: str):
    """
    Print all models sorted by primary validation metric.
    """
    print("\n" + "=" * 60)
    print("  STEP 7: MODEL LEADERBOARD (sorted by val score)")
    print("=" * 60)

    rows = []
    for name, res in baselines.items():
        rows.append({"Model": name, "Type": "Baseline", **res["scores"]})

    algo = default_result["algo"]
    rows.append({"Model": f"{algo} (default)", "Type": "Primary", **default_result["scores"]})

    if tuned_result:
        rows.append({"Model": f"{algo} (tuned)", "Type": "Tuned", **tuned_result["scores"]})

    leaderboard = pd.DataFrame(rows)

    # Sort by primary metric
    primary_metric = (
        "roc_auc"     if "binary" in problem_type else
        "f1_weighted" if "classification" in problem_type else
        "r2"
    )
    if primary_metric in leaderboard.columns:
        leaderboard = leaderboard.sort_values(primary_metric, ascending=False)

    print(leaderboard.to_string(index=False))

    best = leaderboard.iloc[0]["Model"]
    print(f"\n  Best model : {best}")
    print(f"  Proceed with '{best}' to model_evaluator.")
    print("=" * 60 + "\n")

    return leaderboard
```

---

## Master Orchestrator

```python
def run_model_training(
    df: pd.DataFrame,
    target_col: str,
    selection: dict,
    dataset_name: str = "dataset",
    n_trials: int = None,
) -> dict:

    problem_type = selection["problem_type"]

    # Step 1: Split
    splits = split_data(df, target_col, problem_type)

    # Step 2: Baselines
    baselines = train_baselines(splits, selection)

    # Step 3: Default primary
    default_result = train_default_model(splits, selection)

    # Step 4: Tune
    tuned_result = tune_model(splits, selection, n_trials_override=n_trials)

    # Step 5: Charts
    best_model = tuned_result["model"] if tuned_result else default_result["model"]
    plot_learning_curve(best_model, splits, problem_type, dataset_name)
    plot_boosting_curve(best_model, splits, problem_type, dataset_name)

    # Step 6: Manifest
    manifest = save_all_models(baselines, default_result, tuned_result)

    # Step 7: Leaderboard
    leaderboard = print_leaderboard(baselines, default_result, tuned_result, problem_type)

    return {
        "splits":         splits,
        "baselines":      baselines,
        "default_result": default_result,
        "tuned_result":   tuned_result,
        "best_model":     best_model,
        "leaderboard":    leaderboard,
        "manifest":       manifest,
    }
```

---

## Usage Example

```python
# Run algorithm selection first
selection = run_algorithm_selection(df_engineered, target_col="churn")

# Train all models
training = run_model_training(
    df_engineered,
    target_col="churn",
    selection=selection,
    dataset_name="telco_churn.csv",
    n_trials=50,
)

# Access best model
best_model = training["best_model"]
splits     = training["splits"]
```
