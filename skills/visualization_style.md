# Skill: Visualization Style

## Purpose
Defines Brahma's visual identity. Every chart produced by every skill imports and
applies this module. No chart is ever rendered without it.

---

## Core Rule — Charts Tell Findings, Not Metrics

**Title = THE FINDING. Never the metric name.**

| Wrong (metric name)                        | Right (finding)                                         |
|--------------------------------------------|---------------------------------------------------------|
| "Monthly Transactions by Churn Status"     | "Churned Customers Have 3× Lower Monthly Transactions"  |
| "Age Distribution"                         | "Most Churners Are Concentrated in the 25–34 Age Band"  |
| "Correlation Matrix"                       | "Tenure and Contract Length Are Nearly Perfectly Correlated (r=0.91)" |
| "Missing Values"                           | "3 Columns Are >40% Missing — Drop Before Modelling"    |

Every chart produced by Brahma must follow this rule. If you cannot state the finding yet,
use a placeholder but flag it: `"[FINDING TBD] Monthly Transactions by Churn Status"`.

---

## BRAHMA_COLORS Palette

```python
BRAHMA_COLORS = {
    "primary":     "#2563EB",   # Electric Blue   — main series, default bars/lines
    "highlight":   "#DC2626",   # Signal Red       — anomalies, outliers, flagged items
    "muted":       "#D1D5DB",   # Light Grey       — background series, non-highlighted elements
    "success":     "#16A34A",   # Forest Green     — positive outcomes, good metrics
    "warning":     "#D97706",   # Amber            — caution zones, moderate risk
    "neutral":     "#6B7280",   # Slate Grey       — grid lines, annotations, secondary text
    "dark":        "#111827",   # Near Black       — titles, axis labels
    "background":  "#FFFFFF",   # White            — figure and axes background
}

# Ordered list for multi-series charts (never use default matplotlib cycle)
BRAHMA_PALETTE = [
    "#2563EB",  # Electric Blue
    "#DC2626",  # Signal Red
    "#16A34A",  # Forest Green
    "#D97706",  # Amber
    "#7C3AED",  # Purple
    "#0891B2",  # Cyan
    "#BE185D",  # Pink
    "#6B7280",  # Slate Grey
]

# Diverging palette for correlation heatmaps (cool → white → warm)
BRAHMA_DIVERGING = "RdBu_r"

# Sequential palette for single-variable heatmaps
BRAHMA_SEQUENTIAL = "Blues"
```

---

## Canvas Standards

```python
CANVAS = {
    "figsize":   (14, 9),       # inches — wide landscape, CXO-presentation ready
    "dpi":       150,           # crisp on retina / projectors
    "facecolor": "#FFFFFF",     # white background always
}
```

---

## Typography

```python
TYPOGRAPHY = {
    "title_size":    18,
    "title_weight":  "bold",
    "subtitle_size": 13,
    "subtitle_color": "#6B7280",   # Slate Grey — visually subordinate to title
    "axis_label_size": 12,
    "axis_label_weight": "semibold",
    "tick_size":     10,
    "annotation_size": 9,
    "source_size":   8,
    "source_color":  "#9CA3AF",    # Light slate — unobtrusive
}
```

---

## Style Module

```python
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Palette ─────────────────────────────────────────────────────────────────
BRAHMA_COLORS = {
    "primary":    "#2563EB",
    "highlight":  "#DC2626",
    "muted":      "#D1D5DB",
    "success":    "#16A34A",
    "warning":    "#D97706",
    "neutral":    "#6B7280",
    "dark":       "#111827",
    "background": "#FFFFFF",
}

BRAHMA_PALETTE = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706",
    "#7C3AED", "#0891B2", "#BE185D", "#6B7280",
]

BRAHMA_DIVERGING  = "RdBu_r"
BRAHMA_SEQUENTIAL = "Blues"


def apply_brahma_style():
    """
    Set global matplotlib rcParams to Brahma defaults.
    Call once at the top of any script before any chart is drawn.
    """
    plt.rcParams.update({
        # Figure
        "figure.figsize":        (14, 9),
        "figure.dpi":            150,
        "figure.facecolor":      "#FFFFFF",
        "figure.autolayout":     False,   # we call tight_layout() explicitly

        # Axes
        "axes.facecolor":        "#FFFFFF",
        "axes.edgecolor":        "#D1D5DB",
        "axes.linewidth":        0.8,
        "axes.spines.top":       False,   # NEVER top spine
        "axes.spines.right":     False,   # NEVER right spine
        "axes.spines.left":      True,
        "axes.spines.bottom":    True,
        "axes.prop_cycle":       matplotlib.cycler(color=BRAHMA_PALETTE),
        "axes.titlesize":        18,
        "axes.titleweight":      "bold",
        "axes.titlepad":         14,
        "axes.labelsize":        12,
        "axes.labelweight":      "semibold",
        "axes.labelcolor":       "#111827",
        "axes.grid":             True,
        "axes.axisbelow":        True,   # grid behind data

        # Grid
        "grid.color":            "#E5E7EB",
        "grid.linewidth":        0.6,
        "grid.alpha":            0.7,

        # Ticks
        "xtick.labelsize":       10,
        "ytick.labelsize":       10,
        "xtick.color":           "#6B7280",
        "ytick.color":           "#6B7280",
        "xtick.direction":       "out",
        "ytick.direction":       "out",

        # Legend
        "legend.fontsize":       10,
        "legend.framealpha":     0.9,
        "legend.edgecolor":      "#D1D5DB",

        # Font
        "font.family":           "sans-serif",
        "font.size":             11,

        # Save
        "savefig.dpi":           150,
        "savefig.facecolor":     "#FFFFFF",
        "savefig.bbox":          "tight",
    })


def new_figure(nrows: int = 1, ncols: int = 1,
               figsize: tuple = (14, 9)) -> tuple:
    """
    Create a new figure with Brahma canvas settings applied.
    Returns (fig, axes).
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize,
                             facecolor="#FFFFFF")
    return fig, axes


def annotate_chart(
    ax,
    title: str,
    subtitle: str = "",
    xlabel: str = "",
    ylabel: str = "",
    source: str = "",
):
    """
    Apply the full Brahma annotation layer to an axes object.

    Parameters
    ----------
    ax       : matplotlib Axes
    title    : THE FINDING — not the metric name
    subtitle : supporting context (sample size, date range, method note)
    xlabel   : x-axis label with units, e.g. "Monthly Transactions (count)"
    ylabel   : y-axis label with units, e.g. "Customer Count"
    source   : dataset name, e.g. "telco_churn.csv"
               Auto-formatted as: "Source: {source} | Brahma ML Pipeline"
    """
    # Title — the finding
    ax.set_title(title, fontsize=18, fontweight="bold",
                 color="#111827", pad=14, loc="left")

    # Subtitle — context
    if subtitle:
        ax.text(0, 1.01, subtitle,
                transform=ax.transAxes,
                fontsize=13, color="#6B7280",
                va="bottom", ha="left")

    # Axis labels with units
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight="semibold",
                      color="#111827", labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight="semibold",
                      color="#111827", labelpad=8)

    # Source annotation — bottom-right corner of the figure
    if source:
        source_text = f"Source: {source} | Brahma ML Pipeline"
        ax.annotate(
            source_text,
            xy=(1, -0.08), xycoords="axes fraction",
            fontsize=8, color="#9CA3AF",
            ha="right", va="top",
        )

    # Remove top and right spines (belt-and-suspenders over rcParams)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_chart(fig, path: str):
    """
    Save figure using Brahma save standards.
    Always: tight_layout, dpi=150, facecolor=white.
    """
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#FFFFFF", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {path}")
```

---

## NEVER Rules

These are hard bans. No exception, no override.

| Rule | Why |
|------|-----|
| **No rainbow colormaps** (`jet`, `rainbow`, `hsv`) | Perceptually non-uniform — misleads the eye about magnitude. Use `BRAHMA_DIVERGING` or `BRAHMA_SEQUENTIAL`. |
| **No 3D charts** | 3D distorts perception of values. Every 3D chart has a better 2D equivalent. |
| **No pie charts** | Human eyes cannot accurately judge arc angles. Use a sorted horizontal bar chart instead. |
| **No default matplotlib blue** (`#1f77b4`) | Signals an uncustomised chart. Always use `BRAHMA_COLORS["primary"]`. |
| **No truncated Y-axes** | Truncating makes small differences look enormous. Always start Y at 0 for bar charts. For line charts, show the full meaningful range and annotate the min. |
| **No chartjunk** | No decorative 3D effects, drop shadows, unnecessary gridlines, or busy backgrounds. |

---

## Standard Import Block

Every skill that generates charts must begin with:

```python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, "skills")          # allow relative skill imports

# Apply Brahma style globally before any chart is drawn
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
```

---

## Quick Reference Card

```
Canvas    : 14×9 in, 150 DPI, white background
Spines    : left + bottom only (top/right always off)
Title     : THE FINDING (bold, 18pt, left-aligned)
Subtitle  : context text (13pt, slate grey, below title)
Axis labels: include units in parentheses
Source    : bottom-right, 8pt, "Source: X | Brahma ML Pipeline"
Colors    : primary=#2563EB  highlight=#DC2626  muted=#D1D5DB
Diverging : RdBu_r (heatmaps)
Sequential: Blues (single-var heatmaps)
Save      : tight_layout + dpi=150 + facecolor=white
NEVER     : rainbow cmaps, 3D, pie charts, default blue, truncated Y
```
