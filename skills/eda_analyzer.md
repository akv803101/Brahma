# Skill: EDA Analyzer

## Purpose
Generate a complete Exploratory Data Analysis suite — univariate, bivariate, correlation,
and missingness charts — and distill findings into a Key Findings report.
Every chart is hypothesis-first: the title states the finding, not the metric.

**Depends on:** `skills/visualization_style.md` — must be imported before any chart is drawn.

---

## Standard Import Block

```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "skills")

from visualization_style import (
    apply_brahma_style,
    new_figure,
    annotate_chart,
    save_chart,
    BRAHMA_COLORS,
    BRAHMA_PALETTE,
    BRAHMA_DIVERGING,
    BRAHMA_SEQUENTIAL,
)

apply_brahma_style()

OUTPUT_DIR = "outputs/charts/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

---

## Chart 1 — Univariate: Distribution Plot (Histogram + KDE) per Numeric Column

```python
def plot_numeric_distribution(df: pd.DataFrame, col: str,
                               target_col: str = None,
                               dataset_name: str = "dataset") -> str:
    """
    Histogram + KDE overlay for a single numeric column.
    If target_col is provided, overlay distributions per class (hue).
    Title states the finding about the distribution shape.
    """
    fig, ax = new_figure(figsize=(14, 9))

    # Determine finding-first title
    skew = df[col].skew()
    skew_desc = (
        f"Right-Skewed (skew={skew:.2f}) — Consider Log Transform"
        if skew > 1 else
        f"Left-Skewed (skew={skew:.2f})"
        if skew < -1 else
        f"Approximately Normal (skew={skew:.2f})"
    )

    if target_col and target_col in df.columns:
        classes = df[target_col].unique()
        for i, cls in enumerate(classes):
            subset = df[df[target_col] == cls][col].dropna()
            color  = BRAHMA_PALETTE[i % len(BRAHMA_PALETTE)]
            subset.plot.hist(ax=ax, bins=40, alpha=0.45, color=color,
                             density=True, label=f"{target_col}={cls}")
            subset.plot.kde(ax=ax, color=color, linewidth=2.5)
        ax.legend(title=target_col, framealpha=0.9)
        title    = f"{col}: Distribution Differs Across {target_col} Classes — {skew_desc}"
        subtitle = f"n={len(df):,} rows | Histogram + KDE per class"
    else:
        df[col].dropna().plot.hist(ax=ax, bins=40, alpha=0.7,
                                   color=BRAHMA_COLORS["primary"], density=True)
        df[col].dropna().plot.kde(ax=ax, color=BRAHMA_COLORS["highlight"], linewidth=2.5)
        title    = f"{col} Is {skew_desc}"
        subtitle = f"n={df[col].notna().sum():,} non-null values  |  mean={df[col].mean():.2f}  median={df[col].median():.2f}  std={df[col].std():.2f}"

    annotate_chart(ax,
        title=title,
        subtitle=subtitle,
        xlabel=f"{col}",
        ylabel="Density",
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, f"eda_distribution_{col}.png")
    save_chart(fig, path)
    return path
```

---

## Chart 2 — Univariate: Horizontal Sorted Bar Chart per Categorical Column

```python
def plot_categorical_bar(df: pd.DataFrame, col: str,
                          top_n: int = 20,
                          dataset_name: str = "dataset") -> str:
    """
    Horizontal bar chart of value counts, sorted descending.
    Highlights the dominant category in primary blue; rest in muted grey.
    Title states the dominant category and its share.
    """
    counts    = df[col].value_counts().head(top_n)
    total     = df[col].notna().sum()
    top_cat   = counts.index[0]
    top_pct   = counts.iloc[0] / total * 100

    # Color: dominant = primary, rest = muted
    colors = [
        BRAHMA_COLORS["primary"] if c == top_cat else BRAHMA_COLORS["muted"]
        for c in counts.index
    ]

    fig, ax = new_figure(figsize=(14, max(6, len(counts) * 0.55 + 2)))

    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, counts.values[::-1]):
        pct = val / total * 100
        ax.text(bar.get_width() + total * 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)",
                va="center", ha="left",
                fontsize=9, color=BRAHMA_COLORS["neutral"])

    ax.set_xlim(0, counts.max() * 1.18)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    title    = f"'{top_cat}' Dominates {col} with {top_pct:.1f}% of All Records"
    subtitle = (f"Top {min(top_n, len(counts))} of {df[col].nunique()} categories shown  "
                f"|  n={total:,} non-null")

    annotate_chart(ax,
        title=title,
        subtitle=subtitle,
        xlabel="Count",
        ylabel=col,
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, f"eda_categorical_{col}.png")
    save_chart(fig, path)
    return path
```

---

## Chart 3 — Target Deep Dive: Class Distribution with % Labels

```python
def plot_target_distribution(df: pd.DataFrame, target_col: str,
                              dataset_name: str = "dataset") -> str:
    """
    Bar chart of target class distribution with count and % labels.
    Flags imbalance visually: minority class bars coloured Signal Red.
    Title states whether the classes are balanced or imbalanced.
    """
    counts      = df[target_col].value_counts().sort_index()
    total       = len(df)
    proportions = counts / total
    min_prop    = proportions.min()
    imbalanced  = min_prop < 0.20

    colors = [
        BRAHMA_COLORS["highlight"] if proportions[c] < 0.20
        else BRAHMA_COLORS["primary"]
        for c in counts.index
    ]

    fig, ax = new_figure(figsize=(14, 9))

    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=colors, edgecolor="white", linewidth=0.8, width=0.55)

    # Labels on bars
    for bar, val, prop in zip(bars, counts.values, proportions.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f"{val:,}\n({prop*100:.1f}%)",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                color=BRAHMA_COLORS["dark"])

    ax.set_ylim(0, counts.max() * 1.20)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    if imbalanced:
        title    = (f"CLASS IMBALANCE DETECTED: Minority Class = {min_prop*100:.1f}% "
                    f"— Model Will Need Correction")
        subtitle = "Red bars = classes below 20% threshold | Consider SMOTE or class_weight='balanced'"
    else:
        title    = f"Target Classes Are Balanced (minimum {min_prop*100:.1f}%) — No Correction Needed"
        subtitle = f"{len(counts)} classes  |  n={total:,} rows"

    annotate_chart(ax,
        title=title,
        subtitle=subtitle,
        xlabel=target_col,
        ylabel="Count",
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, f"eda_target_{target_col}.png")
    save_chart(fig, path)
    return path
```

---

## Chart 4 — Bivariate: Violin Plots (Numeric vs Target, Classification)

```python
def plot_violin_numeric_vs_target(df: pd.DataFrame, col: str,
                                   target_col: str,
                                   dataset_name: str = "dataset") -> str:
    """
    Violin plot showing distribution of a numeric feature split by target class.
    Overlays individual data points (strip) for small datasets.
    Title quantifies the difference between classes.
    """
    fig, ax = new_figure(figsize=(14, 9))

    classes = sorted(df[target_col].dropna().unique())
    palette = {cls: BRAHMA_PALETTE[i % len(BRAHMA_PALETTE)] for i, cls in enumerate(classes)}

    sns.violinplot(
        data=df, x=target_col, y=col,
        palette=palette,
        inner="box",        # show IQR box inside violin
        linewidth=1.5,
        ax=ax,
        order=classes,
    )

    # Overlay strip for small datasets
    if len(df) <= 5000:
        sns.stripplot(
            data=df, x=target_col, y=col,
            color=BRAHMA_COLORS["dark"], alpha=0.15, size=2.5,
            jitter=True, ax=ax, order=classes,
        )

    # Compute medians per class for subtitle
    medians    = df.groupby(target_col)[col].median()
    median_str = "  |  ".join([f"{cls}: median={medians[cls]:.2f}" for cls in classes])

    # Finding: what is the ratio between the highest and lowest median?
    if len(medians) >= 2:
        ratio    = medians.max() / medians.min() if medians.min() != 0 else float("inf")
        top_cls  = medians.idxmax()
        title    = (f"{col}: '{top_cls}' Class Has {ratio:.1f}× Higher Median "
                    f"— Strong Discriminative Feature")
    else:
        title = f"Distribution of {col} Across {target_col} Classes"

    subtitle = median_str

    annotate_chart(ax,
        title=title,
        subtitle=subtitle,
        xlabel=target_col,
        ylabel=f"{col}",
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, f"eda_violin_{col}.png")
    save_chart(fig, path)
    return path
```

---

## Chart 5 — Bivariate: Grouped Bar Charts (Categorical vs Target)

```python
def plot_grouped_bar_categorical_vs_target(df: pd.DataFrame,
                                            col: str,
                                            target_col: str,
                                            top_n: int = 10,
                                            dataset_name: str = "dataset") -> str:
    """
    Grouped bar chart: for each category in col, show % breakdown by target class.
    Reveals which categories are disproportionately associated with target outcomes.
    Title identifies the category with the highest target rate.
    """
    top_cats  = df[col].value_counts().head(top_n).index
    subset    = df[df[col].isin(top_cats)].copy()
    crosstab  = pd.crosstab(subset[col], subset[target_col], normalize="index") * 100

    classes   = crosstab.columns.tolist()
    n_cats    = len(crosstab)
    bar_width = 0.8 / len(classes)
    x         = np.arange(n_cats)

    fig, ax = new_figure(figsize=(14, 9))

    for i, cls in enumerate(classes):
        offset = (i - len(classes) / 2 + 0.5) * bar_width
        bars   = ax.bar(x + offset, crosstab[cls],
                        width=bar_width * 0.9,
                        color=BRAHMA_PALETTE[i % len(BRAHMA_PALETTE)],
                        label=f"{target_col}={cls}",
                        edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, crosstab[cls]):
            if val >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{val:.0f}%",
                        ha="center", va="bottom",
                        fontsize=8, color=BRAHMA_COLORS["dark"])

    ax.set_xticks(x)
    ax.set_xticklabels(crosstab.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(title=target_col, loc="upper right")

    # Find most discriminating category
    if len(classes) == 2:
        target_class = classes[-1]
        max_cat = crosstab[target_class].idxmax()
        max_val = crosstab[target_class].max()
        title   = (f"'{max_cat}' in {col} Has Highest {target_col}={target_class} "
                   f"Rate at {max_val:.1f}%")
    else:
        title = f"Rate of {target_col} Varies Significantly Across {col} Categories"

    subtitle = f"Top {n_cats} categories shown  |  % within each category"

    annotate_chart(ax,
        title=title,
        subtitle=subtitle,
        xlabel=col,
        ylabel=f"% of {target_col} class",
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, f"eda_grouped_bar_{col}.png")
    save_chart(fig, path)
    return path
```

---

## Chart 6 — Correlation Heatmap (Annotated, Diverging, Upper Triangle Masked)

```python
def plot_correlation_heatmap(df: pd.DataFrame,
                              target_col: str = None,
                              dataset_name: str = "dataset") -> tuple[str, pd.DataFrame]:
    """
    Pearson correlation heatmap.
    - Upper triangle masked (no redundant information)
    - Diverging RdBu_r palette centred at 0
    - Annotated with correlation coefficients
    - Target column highlighted if provided
    Returns path and the correlation matrix.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col and target_col not in numeric_df.columns:
        # Try to include target if it was encoded as int
        if target_col in df.columns:
            numeric_df[target_col] = df[target_col]

    corr = numeric_df.corr(method="pearson")
    n    = len(corr)

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig_h = max(9, n * 0.7)
    fig_w = max(14, n * 0.8)
    fig, ax = new_figure(figsize=(fig_w, fig_h))

    sns.heatmap(
        corr,
        mask=mask,
        cmap=BRAHMA_DIVERGING,
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.4,
        linecolor="#E5E7EB",
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        square=True,
    )

    # Rotate axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    # Find strongest off-diagonal pair for title
    corr_pairs = (
        corr.where(~mask)
            .stack()
            .reset_index()
    )
    corr_pairs.columns = ["col_a", "col_b", "r"]
    corr_pairs = corr_pairs[corr_pairs["col_a"] != corr_pairs["col_b"]]
    if len(corr_pairs):
        corr_pairs["abs_r"] = corr_pairs["r"].abs()
        top_pair = corr_pairs.sort_values("abs_r", ascending=False).iloc[0]
        title    = (f"Strongest Correlation: {top_pair['col_a']} ↔ {top_pair['col_b']} "
                    f"(r={top_pair['r']:.2f})")
    else:
        title = "Feature Correlation Matrix"

    # Count severe multicollinearity
    severe = corr_pairs[corr_pairs["abs_r"] > 0.85]
    subtitle = (f"{len(severe)} severely collinear pair(s) found (|r| > 0.85)  "
                f"|  Lower triangle only  |  n={len(df):,}")

    annotate_chart(ax,
        title=title,
        subtitle=subtitle,
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, "eda_correlation_heatmap.png")
    save_chart(fig, path)
    return path, corr
```

---

## Chart 7 — Missing Value Map (Heatmap of Nulls Across Columns)

```python
def plot_missing_value_map(df: pd.DataFrame,
                            dataset_name: str = "dataset") -> str:
    """
    Heatmap where each cell = 1 (missing) or 0 (present).
    Rows are sampled if dataset is large.
    Columns sorted by missingness % descending.
    Title states the most urgent action required.
    """
    null_pcts = df.isnull().mean().sort_values(ascending=False)
    cols_with_nulls = null_pcts[null_pcts > 0].index.tolist()

    if not cols_with_nulls:
        print("  [SKIP] Missing Value Map — no missing values in dataset.")
        return ""

    # Sample rows for display (heatmap becomes illegible above ~500 rows)
    max_display_rows = 500
    display_df = df[cols_with_nulls]
    if len(display_df) > max_display_rows:
        display_df = display_df.sample(max_display_rows, random_state=42)

    fig_w = max(14, len(cols_with_nulls) * 0.9)
    fig, ax = new_figure(figsize=(fig_w, 9))

    sns.heatmap(
        display_df.isnull().astype(int),
        cmap=["#FFFFFF", BRAHMA_COLORS["highlight"]],
        cbar=False,
        ax=ax,
        yticklabels=False,
    )

    # X-axis: column names with % annotation
    ax.set_xticklabels(
        [f"{c}\n({null_pcts[c]*100:.1f}%)" for c in cols_with_nulls],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_ylabel("")

    # Finding-first title
    worst_col  = null_pcts.idxmax()
    worst_pct  = null_pcts.max() * 100
    drop_count = (null_pcts > 0.50).sum()

    if drop_count > 0:
        title = (f"{drop_count} Column(s) Are >50% Missing and Should Be Dropped — "
                 f"'{worst_col}' Is Worst at {worst_pct:.1f}%")
    else:
        title = (f"'{worst_col}' Has the Most Missing Data ({worst_pct:.1f}%) — "
                 f"All Columns Below 50% Drop Threshold")

    subtitle = (f"{len(cols_with_nulls)} of {df.shape[1]} columns have missing values  "
                f"|  Red = missing  |  Rows sampled: {len(display_df):,}")

    annotate_chart(ax,
        title=title,
        subtitle=subtitle,
        source=dataset_name,
    )

    path = os.path.join(OUTPUT_DIR, "eda_missing_value_map.png")
    save_chart(fig, path)
    return path
```

---

## Chart 8 — Pairplot of Top 6 Features (Only If Rows < 100,000)

```python
def plot_pairplot(df: pd.DataFrame,
                  target_col: str = None,
                  corr_matrix: pd.DataFrame = None,
                  dataset_name: str = "dataset") -> str:
    """
    Seaborn pairplot of the top 6 most correlated features with the target.
    Only rendered if len(df) < 100,000 (too slow and illegible at scale).
    Diagonal: KDE. Off-diagonal: scatter with target hue.
    """
    if len(df) >= 100_000:
        print(f"  [SKIP] Pairplot — {len(df):,} rows exceeds 100,000 limit. "
              f"Skipping to preserve performance.")
        return ""

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Select top 6 features by absolute correlation with target
    if target_col and corr_matrix is not None and target_col in corr_matrix.columns:
        top_features = (
            corr_matrix[target_col]
            .drop(index=target_col, errors="ignore")
            .abs()
            .sort_values(ascending=False)
            .head(6)
            .index.tolist()
        )
    elif target_col and target_col in numeric_cols:
        top_features = (
            df[numeric_cols].corr()[target_col]
            .drop(index=target_col, errors="ignore")
            .abs()
            .sort_values(ascending=False)
            .head(6)
            .index.tolist()
        )
    else:
        top_features = numeric_cols[:6]

    plot_cols = top_features + ([target_col] if target_col and target_col not in top_features else [])
    plot_df   = df[plot_cols].dropna()

    # Sample for speed
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=42)

    palette = {cls: BRAHMA_PALETTE[i % len(BRAHMA_PALETTE)]
               for i, cls in enumerate(sorted(plot_df[target_col].unique()))} \
              if target_col else None

    g = sns.pairplot(
        plot_df,
        hue=target_col if target_col else None,
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 15},
        diag_kws={"linewidth": 2},
        palette=palette,
    )
    g.figure.set_size_inches(14, 12)
    g.figure.set_facecolor("#FFFFFF")

    # Apply Brahma title
    finding = (f"Top 6 Features vs {target_col} — "
               f"Look for Clean Separation Between Classes"
               if target_col else
               "Pairwise Relationships Across Top 6 Numeric Features")
    subtitle = f"n={len(plot_df):,} rows (sampled)  |  KDE on diagonal  |  Scatter off-diagonal"

    g.figure.suptitle(finding, fontsize=16, fontweight="bold",
                      color="#111827", y=1.01, x=0.05, ha="left")
    g.figure.text(0.05, 1.0, subtitle, fontsize=11,
                  color="#6B7280", ha="left", va="top")

    path = os.path.join(OUTPUT_DIR, "eda_pairplot_top6.png")
    g.figure.tight_layout()
    g.figure.savefig(path, dpi=150, facecolor="#FFFFFF", bbox_inches="tight")
    plt.close(g.figure)
    print(f"  Chart saved → {path}")
    return path
```

---

## Key Findings Report

```python
def print_key_findings(
    df: pd.DataFrame,
    target_col: str,
    corr_matrix: pd.DataFrame,
    imbalanced: dict,
    dataset_name: str = "dataset",
):
    """
    Print a structured Key Findings report after all charts are generated.
    Covers: top correlated features, multicollinearity, class balance, anomalies,
    and feature engineering recommendations.
    """
    print("\n" + "=" * 70)
    print("  BRAHMA KEY FINDINGS — EDA COMPLETE")
    print("=" * 70)
    print(f"  Dataset : {dataset_name}")
    print(f"  Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Target  : {target_col}")
    print("=" * 70)

    # ── Finding 1: Top 3 features correlated with target ────────────────────
    print("\n  [1] TOP 3 FEATURES MOST CORRELATED WITH TARGET")
    print("  " + "-" * 50)
    if target_col in corr_matrix.columns:
        top3 = (
            corr_matrix[target_col]
            .drop(index=target_col, errors="ignore")
            .abs()
            .sort_values(ascending=False)
            .head(3)
        )
        for rank, (feat, r_abs) in enumerate(top3.items(), 1):
            direction = "positive" if corr_matrix.loc[feat, target_col] > 0 else "negative"
            print(f"  #{rank}  {feat:<35}  |r| = {r_abs:.3f}  ({direction} correlation)")
            print(f"       → Higher {feat} is associated with "
                  f"{'higher' if direction == 'positive' else 'lower'} {target_col}.")
    else:
        print("  Target column not found in correlation matrix.")

    # ── Finding 2: Severe multicollinearity pairs (|r| > 0.85) ──────────────
    print("\n  [2] SEVERE MULTICOLLINEARITY (|r| > 0.85)")
    print("  " + "-" * 50)
    mask     = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    corr_df  = corr_matrix.where(~mask).stack().reset_index()
    corr_df.columns = ["col_a", "col_b", "r"]
    severe   = corr_df[corr_df["r"].abs() > 0.85].sort_values("r", key=abs, ascending=False)

    if len(severe) == 0:
        print("  No severe multicollinearity detected. All pairs |r| ≤ 0.85.")
    else:
        for _, row in severe.iterrows():
            print(f"  {row['col_a']} ↔ {row['col_b']}  r={row['r']:.3f}")
            print(f"  → Consider dropping one. Keeping both inflates variance in linear models.")

    # ── Finding 3: Target distribution ──────────────────────────────────────
    print("\n  [3] TARGET DISTRIBUTION")
    print("  " + "-" * 50)
    counts      = df[target_col].value_counts()
    proportions = df[target_col].value_counts(normalize=True)
    for cls in counts.index:
        flag = " ⚠ MINORITY" if proportions[cls] < 0.20 else ""
        print(f"  {str(cls):<20} {counts[cls]:>7,}  ({proportions[cls]*100:.1f}%){flag}")
    if imbalanced:
        print(f"\n  STATUS: IMBALANCED — minority class(es): {list(imbalanced.keys())}")
        print("  Action: Apply SMOTE on training set, OR use class_weight='balanced'.")
        print("          Use F1 / AUC-ROC as primary metrics, NOT accuracy.")
    else:
        print("\n  STATUS: BALANCED — standard metrics (accuracy, F1) are reliable.")

    # ── Finding 4: Anomalies ─────────────────────────────────────────────────
    print("\n  [4] ANOMALIES DETECTED")
    print("  " + "-" * 50)
    anomalies_found = False

    # Constant columns
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        anomalies_found = True
        print(f"  Constant columns (zero variance): {constant_cols}")
        print("  → Drop these — they carry no information.")

    # Extreme skew
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    extreme_skew = [(c, round(df[c].skew(), 2))
                    for c in numeric_cols if abs(df[c].skew()) > 3]
    if extreme_skew:
        anomalies_found = True
        for col, skew in extreme_skew:
            print(f"  Extreme skew: {col}  (skew={skew}) → consider log1p transform")

    # Near-duplicate columns (high correlation not flagged above)
    if not anomalies_found:
        print("  No major anomalies detected.")

    # ── Finding 5: Feature engineering recommendations ───────────────────────
    print("\n  [5] RECOMMENDED FEATURES TO ENGINEER")
    print("  " + "-" * 50)

    recs = []

    # Log transform for extreme skew
    for col, skew in extreme_skew:
        recs.append(f"  log1p({col})  — reduce right skew (current skew={skew})")

    # Ratio features from correlated pairs
    if len(severe) > 0:
        row = severe.iloc[0]
        recs.append(f"  {row['col_a']} / {row['col_b']}  — ratio may outperform both correlated columns")

    # Date features if datetime columns exist
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    for dc in date_cols:
        recs.append(f"  Extract from {dc}: year, month, day_of_week, is_weekend, days_since_epoch")

    # Interaction terms for top-2 correlated features
    if target_col in corr_matrix.columns and len(top3) >= 2:
        f1, f2 = top3.index[0], top3.index[1]
        recs.append(f"  {f1} × {f2}  — interaction term between top-2 correlated features")

    if recs:
        for r in recs:
            print(f"  → {r}")
    else:
        print("  No immediate engineering opportunities identified.")
        print("  Proceed to model training with current features.")

    print("\n" + "=" * 70)
    print("  Brahma is ready for model training.\n")
```

---

## Master Orchestrator

```python
def run_eda(
    df: pd.DataFrame,
    target_col: str = None,
    dataset_name: str = "dataset",
    ordinal_cols: dict = None,
) -> dict:
    """
    Run the complete EDA suite in sequence.
    Returns a dict of all chart paths and the Key Findings data.
    """
    chart_paths = {}
    corr_matrix = pd.DataFrame()
    imbalanced  = {}

    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    print("\n" + "=" * 70)
    print("  BRAHMA EDA — EXPLORATORY DATA ANALYSIS")
    print(f"  Dataset : {dataset_name}")
    print(f"  Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Target  : {target_col or 'None (unsupervised mode)'}")
    print(f"  Charts  → {OUTPUT_DIR}/")
    print("=" * 70)

    # ── Chart 1: Numeric distributions ───────────────────────────────────────
    print(f"\n  Generating {len(numeric_cols)} distribution plot(s)...")
    for col in numeric_cols:
        p = plot_numeric_distribution(df, col, target_col, dataset_name)
        chart_paths[f"distribution_{col}"] = p

    # ── Chart 2: Categorical bar charts ──────────────────────────────────────
    print(f"\n  Generating {len(categorical_cols)} categorical bar chart(s)...")
    for col in categorical_cols:
        p = plot_categorical_bar(df, col, dataset_name=dataset_name)
        chart_paths[f"categorical_{col}"] = p

    # ── Chart 3: Target distribution ─────────────────────────────────────────
    if target_col:
        print("\n  Generating target distribution chart...")
        p = plot_target_distribution(df, target_col, dataset_name)
        chart_paths["target_distribution"] = p
        counts = df[target_col].value_counts(normalize=True)
        imbalanced = {cls: round(pct * 100, 2)
                      for cls, pct in counts.items() if pct < 0.20}

    # ── Chart 4: Violin plots (numeric vs target) ─────────────────────────────
    if target_col:
        print(f"\n  Generating {len(numeric_cols)} violin plot(s)...")
        for col in numeric_cols:
            p = plot_violin_numeric_vs_target(df, col, target_col, dataset_name)
            chart_paths[f"violin_{col}"] = p

    # ── Chart 5: Grouped bar charts (categorical vs target) ───────────────────
    if target_col:
        print(f"\n  Generating {len(categorical_cols)} grouped bar chart(s)...")
        for col in categorical_cols:
            p = plot_grouped_bar_categorical_vs_target(df, col, target_col,
                                                       dataset_name=dataset_name)
            chart_paths[f"grouped_bar_{col}"] = p

    # ── Chart 6: Correlation heatmap ──────────────────────────────────────────
    print("\n  Generating correlation heatmap...")
    p, corr_matrix = plot_correlation_heatmap(df, target_col, dataset_name)
    chart_paths["correlation_heatmap"] = p

    # ── Chart 7: Missing value map ────────────────────────────────────────────
    print("\n  Generating missing value map...")
    p = plot_missing_value_map(df, dataset_name)
    chart_paths["missing_value_map"] = p

    # ── Chart 8: Pairplot (top 6 features, rows < 100k only) ─────────────────
    print("\n  Generating pairplot (top 6 features)...")
    p = plot_pairplot(df, target_col, corr_matrix, dataset_name)
    chart_paths["pairplot_top6"] = p

    # ── Key Findings ──────────────────────────────────────────────────────────
    if target_col:
        print_key_findings(df, target_col, corr_matrix, imbalanced, dataset_name)

    print(f"  Total charts saved: {sum(1 for v in chart_paths.values() if v)}")
    print(f"  Location          : {OUTPUT_DIR}/\n")

    return {"chart_paths": chart_paths, "corr_matrix": corr_matrix, "imbalanced": imbalanced}
```

---

## Chart Naming Convention

```
outputs/charts/eda/
├── eda_distribution_{col}.png          # Chart 1 — one per numeric column
├── eda_categorical_{col}.png           # Chart 2 — one per categorical column
├── eda_target_{target_col}.png         # Chart 3 — target class balance
├── eda_violin_{col}.png                # Chart 4 — one per numeric column
├── eda_grouped_bar_{col}.png           # Chart 5 — one per categorical column
├── eda_correlation_heatmap.png         # Chart 6 — single heatmap
├── eda_missing_value_map.png           # Chart 7 — single map
└── eda_pairplot_top6.png               # Chart 8 — only if rows < 100,000
```

---

## Usage Examples

```python
# Full EDA — classification task
results = run_eda(
    df,
    target_col="churn",
    dataset_name="telco_churn.csv"
)

# Unsupervised / regression (no target class charts)
results = run_eda(
    df,
    target_col="revenue",
    dataset_name="sales_data.parquet"
)

# Access findings programmatically
corr_matrix = results["corr_matrix"]
imbalanced  = results["imbalanced"]
chart_paths = results["chart_paths"]
```
