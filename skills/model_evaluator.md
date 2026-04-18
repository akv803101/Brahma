# Skill: Model Evaluator

## Purpose
Brahma's most chart-heavy skill. 8 chart types, all boardroom quality.
Every metric is computed. Every chart tells a decision-maker something actionable.

**Depends on:** `skills/visualization_style.md`

---

## Standard Import Block

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "skills")

from visualization_style import (
    apply_brahma_style, new_figure, annotate_chart,
    save_chart, BRAHMA_COLORS, BRAHMA_PALETTE, BRAHMA_DIVERGING,
)
apply_brahma_style()

OUTPUT_DIR = "outputs/charts/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

---

## Metrics Computation

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, cohen_kappa_score,
    matthews_corrcoef, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss,
    calibration_curve,
)

def compute_classification_metrics(y_true, y_pred, y_prob=None,
                                    average="binary") -> dict:
    """
    Compute full classification metric suite.
    average: 'binary' for binary tasks, 'macro'/'weighted' for multiclass.
    """
    metrics = {
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "precision_weighted": round(precision_score(y_true, y_pred,
                                    average="weighted", zero_division=0), 4),
        "recall_weighted":   round(recall_score(y_true, y_pred,
                                    average="weighted", zero_division=0), 4),
        "f1_macro":          round(f1_score(y_true, y_pred,
                                    average="macro", zero_division=0), 4),
        "f1_weighted":       round(f1_score(y_true, y_pred,
                                    average="weighted", zero_division=0), 4),
        "cohens_kappa":      round(cohen_kappa_score(y_true, y_pred), 4),
        "mcc":               round(matthews_corrcoef(y_true, y_pred), 4),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"]    = round(roc_auc_score(y_true, y_prob), 4)
            metrics["avg_precision"] = round(average_precision_score(y_true, y_prob), 4)
            metrics["brier_score"]= round(brier_score_loss(y_true, y_prob), 4)
        except Exception:
            pass
    return metrics
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, median_absolute_error,
)

def compute_regression_metrics(y_true, y_pred) -> dict:
    """Compute full regression metric suite."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # MAPE — avoid division by zero
    mask = y_true_arr != 0
    mape = np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100

    return {
        "mae":        round(mean_absolute_error(y_true, y_pred), 4),
        "rmse":       round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "r2":         round(r2_score(y_true, y_pred), 4),
        "mape_pct":   round(mape, 2),
        "median_ae":  round(median_absolute_error(y_true, y_pred), 4),
    }
```

---

## Chart 1 — Confusion Matrix

```python
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, model_name: str,
                           class_names: list = None,
                           dataset_name: str = "dataset") -> str:
    """
    Annotated confusion matrix heatmap with both counts and row-normalised %.
    Title names the model and the primary failure mode.
    """
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    labels  = class_names or [str(c) for c in sorted(set(y_true))]
    n       = len(labels)
    annot   = np.array([
        [f"{cm[i,j]:,}\n({cm_norm[i,j]*100:.1f}%)"
         for j in range(n)]
        for i in range(n)
    ])

    fig, ax = new_figure(figsize=(max(10, n*2.5), max(8, n*2.2)))

    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap=BRAHMA_SEQUENTIAL,
        linewidths=0.8, linecolor="white",
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar_kws={"shrink": 0.7, "label": "Row-Normalised Rate"},
        vmin=0, vmax=1,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=10)

    # Identify primary failure mode (highest off-diagonal cell)
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    if off_diag.sum() > 0:
        worst_i, worst_j = np.unravel_index(off_diag.argmax(), off_diag.shape)
        worst_n    = off_diag[worst_i, worst_j]
        worst_rate = cm_norm[worst_i, worst_j] * 100
        finding = (f"{model_name} Most Often Confuses '{labels[worst_i]}' → "
                   f"'{labels[worst_j]}' ({worst_rate:.1f}% miss rate)")
    else:
        finding = f"{model_name} Has Perfect Classification on Test Set"

    annotate_chart(ax,
        title=finding,
        subtitle=f"Test set  |  n={len(y_true):,}  |  Counts + row-normalised %",
        xlabel="Predicted Label",
        ylabel="True Label",
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, f"eval_confusion_matrix.png")
    save_chart(fig, path)
    return path
```

---

## Chart 2 — ROC Curve

```python
def plot_roc_curve(y_true, y_prob, model_name: str,
                   dataset_name: str = "dataset") -> str:
    """
    ROC curve with random-chance diagonal and shaded AUC area.
    Title quantifies improvement over random.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score   = roc_auc_score(y_true, y_prob)
    improvement = (auc_score - 0.5) / 0.5 * 100

    fig, ax = new_figure(figsize=(10, 9))

    ax.plot(fpr, tpr, color=BRAHMA_COLORS["primary"], linewidth=2.5,
            label=f"{model_name}  (AUC = {auc_score:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color=BRAHMA_COLORS["primary"])
    ax.plot([0,1], [0,1], color=BRAHMA_COLORS["muted"], linewidth=1.5,
            linestyle="--", label="Random chance (AUC = 0.500)")

    # Mark point closest to top-left
    dist    = np.sqrt((1 - tpr)**2 + fpr**2)
    opt_idx = np.argmin(dist)
    ax.scatter(fpr[opt_idx], tpr[opt_idx],
               color=BRAHMA_COLORS["highlight"], s=100, zorder=5,
               label=f"Optimal threshold  (FPR={fpr[opt_idx]:.3f}, TPR={tpr[opt_idx]:.3f})")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend(fontsize=10, loc="lower right")

    annotate_chart(ax,
        title=f"ROC Curve — {model_name} Performs {improvement:.0f}% Better Than Random",
        subtitle=f"AUC = {auc_score:.4f}  |  Shaded area = discriminative power  |  Test set n={len(y_true):,}",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate (Recall)",
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, "eval_roc_curve.png")
    save_chart(fig, path)
    return path
```

---

## Chart 3 — Precision-Recall Curve (Imbalanced Only)

```python
def plot_precision_recall_curve(y_true, y_prob, model_name: str,
                                  dataset_name: str = "dataset") -> str:
    """
    Precision-Recall curve with no-skill baseline.
    Only rendered when minority class < 20% of data.
    """
    minority_rate = y_true.mean() if hasattr(y_true, "mean") else np.mean(y_true)
    if minority_rate >= 0.20:
        print(f"  [SKIP] PR Curve — classes are balanced ({minority_rate*100:.1f}% minority). "
              f"Use ROC curve instead.")
        return ""

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)

    fig, ax = new_figure(figsize=(10, 9))

    ax.plot(recall, precision, color=BRAHMA_COLORS["primary"], linewidth=2.5,
            label=f"{model_name}  (AP = {avg_prec:.3f})")
    ax.fill_between(recall, precision, alpha=0.12, color=BRAHMA_COLORS["primary"])
    ax.axhline(minority_rate, color=BRAHMA_COLORS["muted"], linewidth=1.5,
               linestyle="--",
               label=f"No-skill baseline = {minority_rate:.3f} ({minority_rate*100:.1f}%)")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10, loc="upper right")

    lift = avg_prec / minority_rate
    annotate_chart(ax,
        title=f"Precision-Recall — {model_name} Achieves {lift:.1f}× Lift Over No-Skill Baseline",
        subtitle=(f"Average Precision = {avg_prec:.4f}  |  "
                  f"Class imbalance: minority = {minority_rate*100:.1f}%  |  n={len(y_true):,}"),
        xlabel="Recall",
        ylabel="Precision",
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, "eval_precision_recall_curve.png")
    save_chart(fig, path)
    return path
```

---

## Chart 4 — Residual Plots × 4 (Regression Only)

```python
import scipy.stats as stats

def plot_residuals(y_true, y_pred, model_name: str,
                   dataset_name: str = "dataset") -> str:
    """
    2×2 grid of residual diagnostics:
      1. Residuals vs Fitted
      2. QQ-plot of residuals
      3. Residual distribution (histogram + KDE)
      4. Scale-Location (√|residuals| vs fitted)
    """
    residuals = np.array(y_true) - np.array(y_pred)
    fitted    = np.array(y_pred)
    std_res   = residuals / (residuals.std() + 1e-9)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#FFFFFF")
    axes = axes.flatten()

    # 1. Residuals vs Fitted
    axes[0].scatter(fitted, residuals, alpha=0.4, s=15,
                    color=BRAHMA_COLORS["primary"], edgecolors="none")
    axes[0].axhline(0, color=BRAHMA_COLORS["highlight"],
                    linestyle="--", linewidth=1.5)
    axes[0].set_title("Residuals vs Fitted", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].spines[["top","right"]].set_visible(False)

    # 2. QQ-Plot
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    axes[1].scatter(osm, osr, alpha=0.4, s=15,
                    color=BRAHMA_COLORS["primary"], edgecolors="none")
    x_line = np.array([osm.min(), osm.max()])
    axes[1].plot(x_line, slope * x_line + intercept,
                 color=BRAHMA_COLORS["highlight"], linewidth=2, linestyle="--")
    axes[1].set_title("Normal QQ-Plot", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Sample Quantiles")
    axes[1].spines[["top","right"]].set_visible(False)

    # 3. Residual Distribution
    pd.Series(residuals).plot.hist(ax=axes[2], bins=40, alpha=0.65,
                                    color=BRAHMA_COLORS["primary"], density=True)
    pd.Series(residuals).plot.kde(ax=axes[2], color=BRAHMA_COLORS["highlight"], linewidth=2)
    axes[2].axvline(0, color=BRAHMA_COLORS["neutral"], linestyle="--", linewidth=1.5)
    axes[2].set_title("Residual Distribution", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Residual")
    axes[2].set_ylabel("Density")
    axes[2].spines[["top","right"]].set_visible(False)

    # 4. Scale-Location
    sqrt_std_res = np.sqrt(np.abs(std_res))
    axes[3].scatter(fitted, sqrt_std_res, alpha=0.4, s=15,
                    color=BRAHMA_COLORS["primary"], edgecolors="none")
    z  = np.polyfit(fitted, sqrt_std_res, 1)
    xr = np.linspace(fitted.min(), fitted.max(), 100)
    axes[3].plot(xr, np.poly1d(z)(xr), color=BRAHMA_COLORS["highlight"],
                 linewidth=2, linestyle="--")
    axes[3].set_title("Scale-Location", fontsize=13, fontweight="bold")
    axes[3].set_xlabel("Fitted Values")
    axes[3].set_ylabel("√|Standardised Residuals|")
    axes[3].spines[["top","right"]].set_visible(False)

    # Overall title
    skew_val = pd.Series(residuals).skew()
    finding  = (
        "Residuals Are Approximately Normal — Good Model Fit"
        if abs(skew_val) < 0.5 else
        f"Residuals Are Skewed ({skew_val:.2f}) — Check for Outliers or Transformations"
    )
    fig.suptitle(f"{model_name}: {finding}", fontsize=15, fontweight="bold",
                 y=1.01, x=0.02, ha="left", color="#111827")
    fig.text(0.02, 0.995,
             f"Test set n={len(y_true):,}  |  Residual skew={skew_val:.3f}  |  Source: {dataset_name}",
             fontsize=11, color="#6B7280", va="top")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eval_residuals.png")
    fig.savefig(path, dpi=150, facecolor="#FFFFFF", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {path}")
    return path
```

---

## Chart 5 — SHAP Summary Plot (Beeswarm)

```python
def plot_shap_summary(model, X_test: pd.DataFrame,
                       model_name: str,
                       dataset_name: str = "dataset") -> str:
    """
    SHAP beeswarm plot showing direction and magnitude of feature impact.
    Each dot = one prediction. Colour = feature value (red=high, blue=low).
    Title answers: which features drive the target most?
    """
    try:
        import shap
    except ImportError:
        print("  [SKIP] SHAP not installed. Run: pip install shap")
        return ""

    print("  Computing SHAP values (this may take a moment)...")

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_test)
    except Exception:
        try:
            explainer = shap.Explainer(model, X_test)
            shap_vals  = explainer(X_test).values
        except Exception as e:
            print(f"  [SKIP] SHAP failed: {e}")
            return ""

    # For binary classification, SHAP returns list — take positive class
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals = shap_vals[1]

    # Identify top feature for title
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_feat  = X_test.columns[np.argmax(mean_abs)]

    fig, ax = plt.subplots(figsize=(14, max(8, X_test.shape[1] * 0.45 + 2)),
                            facecolor="#FFFFFF")
    shap.summary_plot(shap_vals, X_test,
                      plot_type="dot",
                      show=False,
                      color_bar=True,
                      max_display=min(20, X_test.shape[1]),
                      plot_size=None)

    fig = plt.gcf()
    fig.set_size_inches(14, max(8, X_test.shape[1] * 0.45 + 2))
    fig.set_facecolor("#FFFFFF")
    ax  = fig.axes[0]
    ax.spines[["top","right"]].set_visible(False)

    ax.set_title(
        f"Which Features Drive {model_name}'s Predictions? '{top_feat}' Has Most Impact",
        fontsize=15, fontweight="bold", color="#111827", pad=14, loc="left"
    )
    fig.text(0.02, 0.98,
             f"SHAP beeswarm  |  Each dot = one prediction  |  Red=high feature value, Blue=low",
             fontsize=10, color="#6B7280", va="top")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eval_shap_summary.png")
    fig.savefig(path, dpi=150, facecolor="#FFFFFF", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {path}")
    return path
```

---

## Chart 6 — SHAP Bar Plot (Mean Absolute SHAP)

```python
def plot_shap_bar(model, X_test: pd.DataFrame,
                   model_name: str,
                   dataset_name: str = "dataset") -> str:
    """
    Horizontal bar of mean |SHAP| per feature — the global importance chart.
    Answers: on average, how much does each feature move the prediction?
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_test)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]
    except Exception as e:
        print(f"  [SKIP] SHAP bar plot failed: {e}")
        return ""

    mean_abs  = pd.Series(np.abs(shap_vals).mean(axis=0),
                           index=X_test.columns).sort_values()
    top20     = mean_abs.tail(20)

    colors = [
        BRAHMA_COLORS["primary"] if i == len(top20) - 1
        else BRAHMA_COLORS["muted"]
        for i in range(len(top20))
    ]

    fig, ax = new_figure(figsize=(14, max(8, len(top20) * 0.55 + 2)))
    bars = ax.barh(top20.index, top20.values, color=colors,
                   edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, top20.values):
        ax.text(bar.get_width() + top20.max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9,
                color=BRAHMA_COLORS["neutral"])

    ax.set_xlim(0, top20.max() * 1.18)
    top_feat = top20.index[-1]

    annotate_chart(ax,
        title=f"Top Features by Average Impact on Prediction — '{top_feat}' Drives the Model Most",
        subtitle=f"Mean |SHAP value| across test set  |  n={len(X_test):,}",
        xlabel="Mean |SHAP Value| (average impact on prediction)",
        ylabel="Feature",
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, "eval_shap_bar.png")
    save_chart(fig, path)
    return path
```

---

## Chart 7 — SHAP Waterfall (Highest-Risk Individual Prediction)

```python
def plot_shap_waterfall(model, X_test: pd.DataFrame,
                         y_prob: np.ndarray,
                         model_name: str,
                         dataset_name: str = "dataset") -> str:
    """
    SHAP waterfall for the highest-risk prediction in the test set.
    Shows exactly why Brahma flagged this specific case.
    """
    try:
        import shap
        explainer  = shap.TreeExplainer(model)
        shap_exp   = explainer(X_test)
        if isinstance(shap_exp.values, list):
            # binary: take positive class
            shap_exp = shap.Explanation(
                values=shap_exp.values[1],
                base_values=shap_exp.base_values[1] if hasattr(shap_exp.base_values, "__len__") else shap_exp.base_values,
                data=shap_exp.data,
                feature_names=shap_exp.feature_names,
            )
    except Exception as e:
        print(f"  [SKIP] SHAP waterfall failed: {e}")
        return ""

    # Find highest-risk prediction
    highest_risk_idx = np.argmax(y_prob)
    risk_score       = y_prob[highest_risk_idx]

    fig, ax = plt.subplots(figsize=(14, 9), facecolor="#FFFFFF")
    shap.waterfall_plot(shap_exp[highest_risk_idx], max_display=15, show=False)

    fig = plt.gcf()
    fig.set_size_inches(14, 9)
    fig.set_facecolor("#FFFFFF")

    fig.suptitle(
        f"Why Did {model_name} Flag This Case as High Risk? (Score = {risk_score:.3f})",
        fontsize=15, fontweight="bold", color="#111827", x=0.02, ha="left"
    )
    fig.text(0.02, 0.97,
             f"Highest-risk prediction in test set  |  Each bar = one feature's contribution",
             fontsize=10, color="#6B7280", va="top")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eval_shap_waterfall_highest_risk.png")
    fig.savefig(path, dpi=150, facecolor="#FFFFFF", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {path}")
    return path
```

---

## Chart 8 — Calibration Curve

```python
def plot_calibration_curve(y_true, y_prob, model_name: str,
                             dataset_name: str = "dataset") -> str:
    """
    Calibration curve: predicted probability vs actual fraction positive.
    A perfectly calibrated model follows the diagonal.
    Title tells decision-makers whether to trust the model's confidence scores.
    """
    from sklearn.calibration import calibration_curve as sk_calibration_curve

    fraction_pos, mean_pred = sk_calibration_curve(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = new_figure(figsize=(10, 9))

    ax.plot(mean_pred, fraction_pos,
            color=BRAHMA_COLORS["primary"], linewidth=2.5,
            marker="o", markersize=7, label=f"{model_name}")
    ax.plot([0,1], [0,1], color=BRAHMA_COLORS["muted"],
            linewidth=1.5, linestyle="--", label="Perfect calibration")
    ax.fill_between(mean_pred, mean_pred, fraction_pos,
                    alpha=0.10, color=BRAHMA_COLORS["highlight"],
                    label="Calibration gap")

    ax.set_xlim([0,1])
    ax.set_ylim([0,1.05])
    ax.legend(fontsize=10)

    # Determine calibration quality
    max_gap = np.max(np.abs(fraction_pos - mean_pred))
    if max_gap < 0.05:
        finding = f"Model Confidence Is Trustworthy — Max Calibration Gap = {max_gap:.3f}"
    elif max_gap < 0.10:
        finding = f"Model Is Slightly Overconfident — Consider Platt Scaling (gap={max_gap:.3f})"
    else:
        finding = f"Model Confidence Is NOT Trustworthy — Calibrate Before Deployment (gap={max_gap:.3f})"

    annotate_chart(ax,
        title=f"Is {model_name}'s Confidence Trustworthy? {finding}",
        subtitle=f"Brier Score = {brier:.4f}  |  Lower Brier = better calibration  |  n={len(y_true):,}",
        xlabel="Mean Predicted Probability",
        ylabel="Fraction of Positives (Actual Rate)",
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, "eval_calibration_curve.png")
    save_chart(fig, path)
    return path
```

---

## Print Metrics Report

```python
import datetime

def print_metrics_report(metrics: dict, model_name: str,
                          problem_type: str, dataset_name: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 60)
    print(f"  MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Model       : {model_name}")
    print(f"  Problem     : {problem_type}")
    print(f"  Dataset     : {dataset_name}")
    print(f"  Timestamp   : {ts}")
    print("-" * 60)
    for metric, val in metrics.items():
        print(f"  {metric:<25} {val}")
    print("=" * 60)
```

---

## Master Orchestrator

```python
def run_model_evaluation(
    model,
    splits: dict,
    problem_type: str,
    model_name: str = "Model",
    dataset_name: str = "dataset",
    class_names: list = None,
    minority_threshold: float = 0.20,
) -> dict:

    X_test = splits["X_test"]
    y_test = splits["y_test"]
    chart_paths = {}

    print("\n" + "=" * 60)
    print("  BRAHMA MODEL EVALUATION")
    print(f"  Model   : {model_name}")
    print(f"  Problem : {problem_type}")
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba") and "classification" in problem_type:
        y_prob = model.predict_proba(X_test)
        if y_prob.ndim > 1 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]

    # Compute metrics
    if "classification" in problem_type:
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    else:
        metrics = compute_regression_metrics(y_test, y_pred)

    print_metrics_report(metrics, model_name, problem_type, dataset_name)

    # Charts
    if "classification" in problem_type:
        # Chart 1: Confusion matrix
        chart_paths["confusion_matrix"] = plot_confusion_matrix(
            y_test, y_pred, model_name, class_names, dataset_name)

        if y_prob is not None and "binary" in problem_type:
            # Chart 2: ROC curve
            chart_paths["roc_curve"] = plot_roc_curve(
                y_test, y_prob, model_name, dataset_name)

            # Chart 3: PR curve (imbalanced only)
            minority_rate = y_test.mean() if hasattr(y_test, "mean") else np.mean(y_test)
            if minority_rate < minority_threshold:
                chart_paths["pr_curve"] = plot_precision_recall_curve(
                    y_test, y_prob, model_name, dataset_name)

            # Chart 8: Calibration
            chart_paths["calibration"] = plot_calibration_curve(
                y_test, y_prob, model_name, dataset_name)

    else:
        # Chart 4: Residuals (regression)
        chart_paths["residuals"] = plot_residuals(
            y_test, y_pred, model_name, dataset_name)

    # Charts 5, 6, 7: SHAP
    chart_paths["shap_summary"]   = plot_shap_summary(model, X_test, model_name, dataset_name)
    chart_paths["shap_bar"]       = plot_shap_bar(model, X_test, model_name, dataset_name)
    if y_prob is not None:
        chart_paths["shap_waterfall"] = plot_shap_waterfall(
            model, X_test, y_prob, model_name, dataset_name)

    print(f"\n  Charts saved to: {OUTPUT_DIR}/")
    print(f"  Total charts   : {sum(1 for v in chart_paths.values() if v)}")
    print(f"  Brahma is ready for model validation.\n")

    return {"metrics": metrics, "chart_paths": chart_paths, "y_pred": y_pred, "y_prob": y_prob}
```

---

## Usage Example

```python
eval_results = run_model_evaluation(
    model        = training["best_model"],
    splits       = training["splits"],
    problem_type = selection["problem_type"],
    model_name   = "XGBoost (tuned)",
    dataset_name = "telco_churn.csv",
    class_names  = ["No Churn", "Churned"],
)
```
