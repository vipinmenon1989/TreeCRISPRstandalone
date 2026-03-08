"""
treecrispr/plots.py — Score visualisation and pairwise statistics.

This module produces publication-quality figures and a pairwise statistics
table from the final scored candidates DataFrame.

Outputs (written to *out_dir*)
------------------------------
scores_boxplot.png
    Box-and-whisker plot of score distributions per editor/model.
scores_ridgeline.png
    Ridgeline density plot of score distributions.
scores_rank_heatmap.png
    Kendall τ-b (or Spearman ρ fallback) rank-concordance matrix.
scores_dominance_heatmap.png
    P(Editor A > Editor B) dominance matrix.
pairwise_stats.csv
    Mann–Whitney U p-value, Cliff's δ + magnitude, Hodges–Lehmann shift,
    and median delta for every pair of editors.

Public API
----------
generate_boxplot_and_stats(df, out_dir) → Dict[str, object]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde, kendalltau, mannwhitneyu
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from .models import pretty_model_name

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that are always present in the candidates DataFrame and are NOT scores.
_BASE_KEYS: frozenset[str] = frozenset(
    {"ID", "Start", "End", "Strand", "Sequence", "ReverseComplement", "PAM"}
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that are numeric and not part of the candidate metadata."""
    from pandas.api.types import is_numeric_dtype
    return [c for c in df.columns if c not in _BASE_KEYS and is_numeric_dtype(df[c])]


def _pvalue_mwu(a: np.ndarray, b: np.ndarray) -> float:
    """Mann–Whitney U p-value (two-sided).  Falls back to a permutation test."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if _HAS_SCIPY:
        try:
            return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except Exception:
            pass
    # Permutation fallback.
    rng = np.random.default_rng(42)
    obs = abs(np.nanmean(a) - np.nanmean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    n_iters = min(5000, 200 + 20 * len(pooled))
    ge = sum(
        1 for _ in range(n_iters)
        if (rng.shuffle(pooled), abs(pooled[:n_a].mean() - pooled[n_a:].mean()))[1] >= obs - 1e-12
    )
    return (ge + 1) / (n_iters + 1)


def _cliffs_delta(
    a: np.ndarray,
    b: np.ndarray,
    max_pairs: int = 2_000_000,
    rng_seed: int = 42,
) -> float:
    """Cliff's δ effect size.  Sampled for large arrays."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    rng = np.random.default_rng(rng_seed)
    if n1 * n2 <= max_pairs:
        return float(np.sign(a[:, None] - b[None, :]).sum() / (n1 * n2))
    m = min(max_pairs, n1 * n2)
    idx_a = rng.integers(0, n1, size=m)
    idx_b = rng.integers(0, n2, size=m)
    return float(np.sign(a[idx_a] - b[idx_b]).mean())


def _cliffs_magnitude(delta: float) -> str:
    """Interpret the absolute value of Cliff's δ as a qualitative magnitude."""
    if not np.isfinite(delta):
        return "na"
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def _hodges_lehmann(
    a: np.ndarray,
    b: np.ndarray,
    max_pairs: int = 2_000_000,
    rng_seed: int = 42,
) -> float:
    """Hodges–Lehmann location-shift estimator.  Sampled for large arrays."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    rng = np.random.default_rng(rng_seed)
    if n1 * n2 <= max_pairs:
        return float(np.median((a[:, None] - b[None, :]).ravel()))
    m = min(max_pairs, n1 * n2)
    idx_a = rng.integers(0, n1, size=m)
    idx_b = rng.integers(0, n2, size=m)
    return float(np.median(a[idx_a] - b[idx_b]))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _make_boxplot(
    arrays: List[np.ndarray],
    labels: List[str],
    out_path: Path,
) -> None:
    """Render a box-and-whisker plot of score distributions."""
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    bp = ax.boxplot(arrays, labels=labels, showmeans=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_alpha(0.75)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("Editors")
    ax.set_title("Editor score distribution (boxplot)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _kde_density(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return a normalised KDE (or smoothed histogram fallback) on *grid*."""
    if _HAS_SCIPY and len(x) >= 2:
        try:
            kde = gaussian_kde(x, bw_method="scott")
            y = kde(grid)
            if y.max() > 0:
                y = y / y.max()
            return y
        except Exception:
            pass
    # Smoothed-histogram fallback.
    hist, edges = np.histogram(x, bins=40, range=(0.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = np.interp(grid, centers, hist, left=0.0, right=0.0)
    if len(y) > 5:
        win = 5
        y = np.convolve(y, np.ones(win) / win, mode="same")
    if y.max() > 0:
        y = y / y.max()
    return y


def _make_ridgeline(
    arrays: List[np.ndarray],
    labels: List[str],
    out_path: Path,
) -> None:
    """Render a ridgeline density plot."""
    grid = np.linspace(0.0, 1.0, 400)
    h = 1.0
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 1.2 + 0.8 * len(labels)))
    for idx, (lab, arr) in enumerate(zip(labels, arrays)):
        y = _kde_density(arr, grid)
        base = (len(labels) - idx - 1) * h
        ax.fill_between(grid, base, base + y, alpha=0.9)
        ax.plot(grid, base + y, color="black", linewidth=1.2)
        ax.plot([grid[0], grid[-1]], [base, base], color="black", linewidth=1.2)
        ax.text(grid[0] - 0.05, base + 0.05, lab, ha="right", va="bottom", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Score")
    ax.set_title("Editor score distributions (ridgeline)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _make_rank_heatmap(
    df: pd.DataFrame,
    labels: List[str],
    out_path: Path,
) -> None:
    """Render a Kendall τ-b (Spearman ρ fallback) rank-concordance heatmap."""
    col_by_label = {
        pretty_model_name(c): c
        for c in df.columns
        if pretty_model_name(c) in labels and pd.api.types.is_numeric_dtype(df[c])
    }
    labs = [l for l in labels if l in col_by_label]
    n = len(labs)
    M = np.full((n, n), np.nan, dtype=float)

    for i, li in enumerate(labs):
        xi = pd.to_numeric(df[col_by_label[li]], errors="coerce")
        for j, lj in enumerate(labs):
            xj = pd.to_numeric(df[col_by_label[lj]], errors="coerce")
            pair = pd.concat([xi, xj], axis=1).dropna()
            if len(pair) < 5:
                continue
            if _HAS_SCIPY:
                try:
                    M[i, j] = kendalltau(
                        pair.iloc[:, 0], pair.iloc[:, 1], nan_policy="omit"
                    ).correlation
                    continue
                except Exception:
                    pass
            M[i, j] = pair.corr(method="spearman").iloc[0, 1]

    fig, ax = plt.subplots(figsize=(max(6, 0.65 * n), max(5, 0.6 * n)))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Kendall τ-b" if _HAS_SCIPY else "Spearman ρ")
    ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labs)
    ax.set_title("Editor rank concordance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _make_dominance_heatmap(
    df: pd.DataFrame,
    labels: List[str],
    out_path: Path,
) -> None:
    """Render a P(Editor A > Editor B) dominance matrix."""
    col_by_label = {
        pretty_model_name(c): c
        for c in df.columns
        if pretty_model_name(c) in labels and pd.api.types.is_numeric_dtype(df[c])
    }
    labs = [l for l in labels if l in col_by_label]
    n = len(labs)
    M = np.full((n, n), np.nan, dtype=float)

    for i, li in enumerate(labs):
        xi = pd.to_numeric(df[col_by_label[li]], errors="coerce")
        for j, lj in enumerate(labs):
            xj = pd.to_numeric(df[col_by_label[lj]], errors="coerce")
            pair = pd.concat([xi, xj], axis=1).dropna()
            if len(pair) == 0:
                continue
            M[i, j] = float((pair.iloc[:, 0] > pair.iloc[:, 1]).mean())
    for k in range(n):
        M[k, k] = 0.5  # diagonal baseline

    fig, ax = plt.subplots(figsize=(max(6, 0.65 * n), max(5, 0.6 * n)))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(Editor i > Editor j)")
    ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labs)
    ax.set_title("Editor dominance matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_boxplot_and_stats(
    df: pd.DataFrame,
    out_dir: "str | Path",
) -> Dict[str, object]:
    """
    Generate visualisations and pairwise statistics from a scored results DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`~treecrispr.pipeline.run_full_pipeline`.  Must
        contain at least one numeric score column beyond the base metadata
        columns (ID, Start, End, Strand, Sequence, ReverseComplement, PAM).
    out_dir : str or Path
        Directory where output files are written (created if absent).

    Returns
    -------
    Dict with keys:
        ``box_png``, ``ridge_png``, ``tau_png``, ``dom_png``
            Absolute paths to the four generated images.
        ``stats``
            :class:`pandas.DataFrame` of pairwise statistics.
        ``stats_csv``
            Path to the saved CSV.
        ``labels``
            Sorted list of clean editor names.
        ``ranking``
            Ordered list of editor names (best first).
        ``ranking_text``
            Human-readable ranking string (e.g. ``"A > B = C > D"``).
        ``wins``
            Dict of {editor: number_of_significant_wins}.
        ``medians``
            Dict of {editor: median_score}.
    """
    _EMPTY: Dict[str, object] = {
        "box_png": "", "ridge_png": "", "tau_png": "", "dom_png": "",
        "stats": pd.DataFrame(), "stats_csv": "", "labels": [],
        "ranking": [], "ranking_text": "", "wins": {}, "medians": {},
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_cols = _score_columns(df)
    if not score_cols:
        return _EMPTY

    arrays: List[np.ndarray] = []
    labels: List[str] = []
    for c in sorted(score_cols, key=pretty_model_name):
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        arrays.append(vals)
        labels.append(pretty_model_name(c))

    if not arrays:
        return _EMPTY

    # Generate plots.
    box_path   = out_dir / "scores_boxplot.png"
    ridge_path = out_dir / "scores_ridgeline.png"
    tau_path   = out_dir / "scores_rank_heatmap.png"
    dom_path   = out_dir / "scores_dominance_heatmap.png"

    _make_boxplot(arrays, labels, box_path)
    _make_ridgeline(arrays, labels, ridge_path)
    _make_rank_heatmap(df, labels, tau_path)
    _make_dominance_heatmap(df, labels, dom_path)

    # Pairwise statistics.
    rows = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            e1, e2 = labels[i], labels[j]
            a, b = arrays[i], arrays[j]
            p   = _pvalue_mwu(a, b)
            m1  = float(np.nanmedian(a)) if a.size else float("nan")
            m2  = float(np.nanmedian(b)) if b.size else float("nan")
            cd  = _cliffs_delta(a, b)
            rows.append({
                "Editor1":      e1,
                "Editor2":      e2,
                "pvalue":       p,
                "median1":      m1,
                "median2":      m2,
                "delta_median": m1 - m2,
                "cliffs_delta": cd,
                "cliffs_mag":   _cliffs_magnitude(cd),
                "hl_shift":     _hodges_lehmann(a, b),
                "Better":       e1 if m1 > m2 else (e2 if m2 > m1 else "tie"),
                "pair":         f"{e1}–{e2}",
            })

    stats_df = pd.DataFrame(rows).sort_values(
        ["Editor1", "Editor2", "pvalue"], na_position="last", kind="stable"
    )
    csv_path = out_dir / "pairwise_stats.csv"
    stats_df.to_csv(csv_path, index=False)

    # Overall ranking: significant wins first, then median as tie-breaker.
    med_by_editor = {lab: float(np.nanmedian(arr)) for lab, arr in zip(labels, arrays)}
    wins: Dict[str, int] = {lab: 0 for lab in labels}
    for r in rows:
        if np.isfinite(r["pvalue"]) and r["pvalue"] <= 0.05 and r["Better"] in (r["Editor1"], r["Editor2"]):
            wins[r["Better"]] += 1

    ordered = sorted(labels, key=lambda e: (-wins[e], -med_by_editor[e], e))

    # Compress ties into equality groups.
    groups: list[list[str]] = []
    for e in ordered:
        if not groups:
            groups.append([e])
        else:
            head = groups[-1][0]
            if wins[e] == wins[head] and round(med_by_editor[e], 3) == round(med_by_editor[head], 3):
                groups[-1].append(e)
            else:
                groups.append([e])
    ranking_text = " > ".join(" = ".join(g) for g in groups)

    return {
        "box_png":      str(box_path),
        "ridge_png":    str(ridge_path),
        "tau_png":      str(tau_path),
        "dom_png":      str(dom_path),
        "stats":        stats_df,
        "stats_csv":    str(csv_path),
        "labels":       labels,
        "ranking":      ordered,
        "ranking_text": ranking_text,
        "wins":         wins,
        "medians":      med_by_editor,
    }
