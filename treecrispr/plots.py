# treecrispr/plots.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu, gaussian_kde, kendalltau
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

_BASE_KEYS = {"ID", "Start", "End", "Strand", "Sequence", "ReverseComplement", "PAM"}

def _pretty(name: str) -> str:
    return (name.replace("_xgb_clf", "")
                .replace("_xgb", "")
                .replace("_clf", "")
                .strip())

def _score_columns(df: pd.DataFrame) -> List[str]:
    from pandas.api.types import is_numeric_dtype
    return [c for c in df.columns if c not in _BASE_KEYS and is_numeric_dtype(df[c])]

def _pvalue_mwu(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if _HAS_SCIPY:
        try:
            return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except Exception:
            pass
    # Fallback: permutation on mean diff
    rng = np.random.default_rng(42)
    obs = abs(np.nanmean(a) - np.nanmean(b))
    pooled = np.concatenate([a, b])
    nA = len(a)
    iters = min(5000, 200 + 20*len(pooled))
    ge = 0
    for _ in range(iters):
        rng.shuffle(pooled)
        diff = abs(pooled[:nA].mean() - pooled[nA:].mean())
        if diff >= obs - 1e-12:
            ge += 1
    return (ge + 1) / (iters + 1)

def _cliffs_delta(a: np.ndarray, b: np.ndarray,
                  max_pairs: int = 2_000_000, rng_seed: int = 42) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    total = n1 * n2
    rng = np.random.default_rng(rng_seed)
    if total <= max_pairs:
        comp = np.sign(a[:, None] - b[None, :])  # +1, 0, -1
        return float(comp.sum() / total)
    m = min(max_pairs, total)
    idx_a = rng.integers(0, n1, size=m)
    idx_b = rng.integers(0, n2, size=m)
    comp = np.sign(a[idx_a] - b[idx_b])
    return float(comp.mean())

def _cliffs_magnitude(delta: float) -> str:
    if not np.isfinite(delta):
        return "na"
    ad = abs(delta)
    if ad < 0.147: return "negligible"
    if ad < 0.33:  return "small"
    if ad < 0.474: return "medium"
    return "large"

def _hodges_lehmann(a: np.ndarray, b: np.ndarray,
                    max_pairs: int = 2_000_000, rng_seed: int = 42) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    total = n1 * n2
    rng = np.random.default_rng(rng_seed)
    if total <= max_pairs:
        diffs = (a[:, None] - b[None, :]).ravel()
        return float(np.median(diffs))
    m = min(max_pairs, total)
    idx_a = rng.integers(0, n1, size=m)
    idx_b = rng.integers(0, n2, size=m)
    diffs = a[idx_a] - b[idx_b]
    return float(np.median(diffs))

# -------- plotting helpers --------

def _make_boxplot(arrays: List[np.ndarray], labels: List[str], out_path: Path) -> None:
    plt.figure(figsize=(max(8, len(labels)*1.1), 5))
    bp = plt.boxplot(arrays, labels=labels, showmeans=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_alpha(0.75)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xlabel("Editors")
    plt.title("Editor score distribution (boxplot)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _kde_density(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if _HAS_SCIPY and len(x) >= 2:
        try:
            kde = gaussian_kde(x, bw_method="scott")
            y = kde(grid)
            if y.max() > 0:
                y = y / y.max()
            return y
        except Exception:
            pass
    # fallback: smoothed histogram density
    hist, edges = np.histogram(x, bins=40, range=(0.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = np.interp(grid, centers, hist, left=0.0, right=0.0)
    if len(y) > 5:
        win = 5
        y = np.convolve(y, np.ones(win)/win, mode="same")
    if y.max() > 0:
        y = y / y.max()
    return y

def _make_ridgeline(arrays: List[np.ndarray], labels: List[str], out_path: Path) -> None:
    grid = np.linspace(0.0, 1.0, 400)
    h = 1.0  # vertical spacing
    plt.figure(figsize=(max(8, len(labels)*1.1), 1.2 + 0.8*len(labels)))
    for idx, (lab, arr) in enumerate(zip(labels, arrays)):
        y = _kde_density(arr, grid)
        base = (len(labels) - idx - 1) * h
        plt.fill_between(grid, base, base + y, alpha=0.9)
        plt.plot(grid, base + y, color="black", linewidth=1.2)
        plt.plot([grid[0], grid[-1]], [base, base], color="black", linewidth=1.2)
        plt.text(grid[0]-0.05, base + 0.05, lab, ha="right", va="bottom", fontsize=10)
    plt.xlim(0, 1)
    plt.yticks([])
    plt.xlabel("Score")
    plt.title("Editor score distributions (ridgeline)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def _make_rank_heatmap(df: pd.DataFrame, labels: List[str], out_path: Path) -> None:
    """Kendall’s τ-b rank concordance between editors."""
    col_by_label = {}
    for c in df.columns:
        p = _pretty(c)
        if p in labels and pd.api.types.is_numeric_dtype(df[c]):
            col_by_label[p] = c
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
                    tau = kendalltau(pair.iloc[:,0], pair.iloc[:,1], nan_policy="omit").correlation
                except Exception:
                    tau = pair.corr(method="spearman").iloc[0,1]
            else:
                tau = pair.corr(method="spearman").iloc[0,1]
            M[i, j] = tau

    plt.figure(figsize=(max(6, 0.65*n), max(5, 0.6*n)))
    im = plt.imshow(M, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Kendall τ-b" if _HAS_SCIPY else "Spearman ρ")
    plt.xticks(range(n), labs, rotation=45, ha="right")
    plt.yticks(range(n), labs)
    plt.title("Editor rank concordance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _make_dominance_heatmap(df: pd.DataFrame, labels: List[str], out_path: Path) -> None:
    """
    For each pair (A,B), compute P(A > B) over guides (ignoring NaN rows).
    Values in [0,1]; diagonal set to 0.5 as a visual baseline.
    """
    col_by_label = {}
    for c in df.columns:
        p = _pretty(c)
        if p in labels and pd.api.types.is_numeric_dtype(df[c]):
            col_by_label[p] = c
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
            M[i, j] = float((pair.iloc[:,0] > pair.iloc[:,1]).mean())

    for k in range(n):
        M[k, k] = 0.5

    plt.figure(figsize=(max(6, 0.65*n), max(5, 0.6*n)))
    im = plt.imshow(M, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="P(Editor i > Editor j)")
    plt.xticks(range(n), labs, rotation=45, ha="right")
    plt.yticks(range(n), labs)
    plt.title("Editor dominance matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -------- public API --------

def generate_boxplot_and_stats(df: pd.DataFrame, out_dir: str | Path) -> Dict[str, object]:
    """
    Create:
      - boxplot (scores per editor, y in [0,1])
      - ridgeline density plot
      - Kendall τ-b rank-concordance heatmap
      - dominance (P(A > B)) heatmap
      - pairwise stats (Mann–Whitney U p, Cliff’s δ + magnitude, HL shift, medians)
      - overall ranking line (wins → median tie-break)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_cols = _score_columns(df)
    if not score_cols:
        return {"box_png": "", "ridge_png": "", "tau_png": "", "dom_png": "",
                "stats": pd.DataFrame(), "stats_csv": "", "labels": [],
                "ranking": [], "ranking_text": "", "wins": {}, "medians": {}}

    arrays: List[np.ndarray] = []
    labels: List[str] = []
    for c in sorted(score_cols, key=_pretty):
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        arrays.append(vals)
        labels.append(_pretty(c))

    if not arrays:
        return {"box_png": "", "ridge_png": "", "tau_png": "", "dom_png": "",
                "stats": pd.DataFrame(), "stats_csv": "", "labels": [],
                "ranking": [], "ranking_text": "", "wins": {}, "medians": {}}

    # Plots
    box_path  = out_dir / "scores_boxplot.png"
    ridge_path= out_dir / "scores_ridgeline.png"
    tau_path  = out_dir / "scores_rank_heatmap.png"
    dom_path  = out_dir / "scores_dominance_heatmap.png"
    _make_boxplot(arrays, labels, box_path)
    _make_ridgeline(arrays, labels, ridge_path)
    _make_rank_heatmap(df, labels, tau_path)
    _make_dominance_heatmap(df, labels, dom_path)

    # Pairwise stats (alphabetical pairs; no self-pairs)
    rows = []
    for i in range(len(arrays)):
        for j in range(i+1, len(arrays)):
            e1, e2 = labels[i], labels[j]
            a, b = arrays[i], arrays[j]
            p  = _pvalue_mwu(a, b)
            m1 = float(np.nanmedian(a)) if a.size else float("nan")
            m2 = float(np.nanmedian(b)) if b.size else float("nan")
            dmed = m1 - m2
            cd = _cliffs_delta(a, b)
            cd_mag = _cliffs_magnitude(cd)
            hl = _hodges_lehmann(a, b)
            rows.append({
                "Editor1": e1,
                "Editor2": e2,
                "pvalue": p,
                "median1": m1,
                "median2": m2,
                "delta_median": dmed,
                "cliffs_delta": cd,
                "cliffs_mag": cd_mag,
                "hl_shift": hl,
                "Better": e1 if dmed > 0 else (e2 if dmed < 0 else "tie"),
                "pair": f"{e1}–{e2}",
            })

    stats_df = pd.DataFrame(rows)
    stats_df = stats_df.sort_values(
        ["Editor1", "Editor2", "pvalue"],
        na_position="last",
        kind="mergesort",   # stable
    )
    csv_path = out_dir / "pairwise_stats.csv"
    stats_df.to_csv(csv_path, index=False)

    # Overall ranking: wins (p<=0.05) then median, then name
    med_by_editor = {lab: float(np.nanmedian(arr)) for lab, arr in zip(labels, arrays)}
    wins = {lab: 0 for lab in labels}
    for r in rows:
        p = r["pvalue"]
        if np.isfinite(p) and p <= 0.05 and r["Better"] in (r["Editor1"], r["Editor2"]):
            wins[r["Better"]] += 1
    ordered = sorted(labels, key=lambda e: (-wins[e], -med_by_editor[e], e))
    # compress groups with same wins & ~same medians
    groups = []
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
        "box_png": str(box_path),
        "ridge_png": str(ridge_path),
        "tau_png": str(tau_path),
        "dom_png": str(dom_path),
        "stats": stats_df,
        "stats_csv": str(csv_path),
        "labels": labels,
        "ranking": ordered,
        "ranking_text": ranking_text,
        "wins": wins,
        "medians": med_by_editor,
    }