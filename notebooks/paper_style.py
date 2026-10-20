"""
Paper plot style — reusable style module.

Usage:
    from paper_figs.paper_style import apply_style, PALETTE, method_colors

    apply_style()
    colors = method_colors(["Ours", "QRPO", "SimPO", "DPO", "REBEL"])
"""
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch

# ── Color Palette ────────────────────────────────────────────────────
ACCENT = "#00897B"  # Teal — bold accent for "Ours"

BASELINES = {
    "QRPO":  "#D4ADA0",   # washed coral
    "SimPO": "#A8B8C8",   # washed slate blue
    "DPO":   "#D6C5A0",   # washed sand
    "REBEL": "#B5B5CC",   # washed lavender
}

# Full palette dict (convenient for lookups)
PALETTE = {"Ours": ACCENT, **BASELINES}

# Ordered list for consistent legend/axis ordering
METHOD_ORDER = ["Ours", "QRPO", "SimPO", "DPO", "REBEL"]


def method_colors(methods):
    """Return list of colors matching a list of method names.

    Unknown methods get a fallback medium gray.
    """
    return [PALETTE.get(m, "#AAAAAA") for m in methods]


def apply_style():
    """Set matplotlib rcParams for paper-quality plots.

    Call once at the top of your notebook or script.
    """
    sns.set_theme(style="ticks")
    mpl.rcParams.update({
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.labelweight": "medium",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        # Axes
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.grid": False,
        "axes.facecolor": "white",

        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 180,

        # Saving
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def style_bar_chart(ax, methods, bold_ours=True):
    """Apply standard bar-chart styling to an axis.

    - Despines left spine
    - Removes left ticks
    - Adds subtle horizontal reference lines
    - Bolds + colors "Ours" y-label
    """
    import numpy as np
    sns.despine(ax=ax, left=True, bottom=False)
    ax.tick_params(left=False, axis="y")
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    # Subtle reference lines
    for y in range(len(methods)):
        ax.axhline(y, color="#ECECEC", linewidth=0.5, zorder=1)

    # Bold "Ours" label
    if bold_ours:
        labels = ax.get_yticklabels()
        for label in labels:
            if label.get_text() == "Ours":
                label.set_fontweight("bold")
                label.set_color(ACCENT)


def style_legend(
    ax,
    ours_label="Ours",
    *,
    fill_edgecolor=None,
    fill_linewidth=0.8,
    fill_labels=None,
    corner_radius=0.8,
    **legend_kwargs,
):
    """Style legend with a rounded frame and optional fill-handle borders."""
    legend_defaults = {
        "framealpha": 0.95,
        "edgecolor": "#CCCCCC",
        "fancybox": True,
    }
    legend_defaults.update(legend_kwargs)
    leg = ax.legend(**legend_defaults)

    if corner_radius is not None:
        leg.get_frame().set_boxstyle(
            f"round,pad=0.28,rounding_size={corner_radius}"
        )

    fill_labels = None if fill_labels is None else set(fill_labels)
    for text, handle in zip(leg.get_texts(), leg.legend_handles):
        if text.get_text() == ours_label:
            text.set_fontweight("bold")

        if fill_edgecolor is None:
            continue
        if not isinstance(handle, Patch):
            continue
        if fill_labels is not None and text.get_text() not in fill_labels:
            continue

        handle.set_edgecolor(fill_edgecolor)
        handle.set_linewidth(fill_linewidth)

    return leg
