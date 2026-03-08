"""
treecrispr/models.py — XGBoost model loading and robust scoring.

Model loading
-------------
All ``.pkl`` and ``.joblib`` files in a given directory are loaded with
``joblib``.  Model names are cleaned by stripping common suffixes
(``_xgb_clf``, ``_xgb``, ``_clf``) so that the output column names in the
results CSV are human-readable.

Scoring
-------
:func:`score_with_models` uses a three-attempt fallback strategy to handle
column-name mismatches between the Python feature extraction pipeline and
models that were originally trained in R:

1. **Standard prediction** — pass features as-is.
2. **Column renaming** — rename ``pos{i}_{N}`` → ``{N}{i+1}`` and
   ``di{i}_{NN}`` → ``{NN}{i+1}`` via :func:`fix_column_names_for_xgboost`.
3. **Strip names** — convert to a NumPy array so XGBoost cannot check names.

Public API
----------
fix_column_names_for_xgboost(df)         → pd.DataFrame
load_models(model_dir, logger)            → Dict[str, Any]
score_with_models(features_df, models, …) → pd.DataFrame
pretty_model_name(raw_name)               → str
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Name utilities
# ---------------------------------------------------------------------------

def pretty_model_name(raw_name: str) -> str:
    """
    Strip common XGBoost file-name suffixes to produce a clean model label.

    Examples
    --------
    >>> pretty_model_name("H3K27ac_xgb_clf")
    'H3K27ac'
    >>> pretty_model_name("SP1_xgb")
    'SP1'
    """
    return (
        raw_name
        .replace("_xgb_clf", "")
        .replace("_xgb", "")
        .replace("_clf", "")
        .strip()
    )


# ---------------------------------------------------------------------------
# Column renaming
# ---------------------------------------------------------------------------

def fix_column_names_for_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename Python-style feature columns to R-trained XGBoost feature names.

    Conversions applied:
    - ``pos{i}_{N}``  → ``{N}{i+1}``   (e.g. ``pos0_A`` → ``A1``)
    - ``di{i}_{NN}``  → ``{NN}{i+1}``  (e.g. ``di0_AA`` → ``AA1``)

    All other column names are left unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with Python-convention column names.

    Returns
    -------
    New DataFrame with renamed columns.
    """
    rename_map: Dict[str, str] = {}

    for col in df.columns:
        if col.startswith("pos") and "_" in col:
            parts = col.split("_", 1)
            suffix = parts[0][3:]  # digits after "pos"
            if suffix.isdigit():
                rename_map[col] = f"{parts[1]}{int(suffix) + 1}"

        elif col.startswith("di") and "_" in col:
            parts = col.split("_", 1)
            suffix = parts[0][2:]  # digits after "di"
            if suffix.isdigit():
                rename_map[col] = f"{parts[1]}{int(suffix) + 1}"

    return df.rename(columns=rename_map) if rename_map else df


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(
    model_dir: Optional[Path],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Load all ``.pkl`` and ``.joblib`` models from *model_dir*.

    Parameters
    ----------
    model_dir : Path or None
        Directory to search.  Returns an empty dict if ``None`` or missing.
    logger : logging.Logger, optional
        Logger for progress and error messages.

    Returns
    -------
    Dict mapping clean model name → fitted model object.
    """
    _log = logger or logging.getLogger(__name__)
    models: Dict[str, Any] = {}

    if not model_dir or not Path(model_dir).exists():
        _log.warning("Model directory missing or not provided: %s", model_dir)
        return models

    model_dir = Path(model_dir)
    paths = sorted(model_dir.glob("*.pkl")) + sorted(model_dir.glob("*.joblib"))

    for p in paths:
        try:
            model = joblib.load(p)
            name = pretty_model_name(p.stem)
            models[name] = model
            _log.info("Loaded model: %s", name)
        except Exception as exc:
            _log.error("Failed to load '%s': %s", p.name, exc)

    _log.info("Total models loaded: %d from %s", len(models), model_dir)
    return models


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------

def _predict_safe(
    model: Any,
    X: pd.DataFrame,
    strip_names: bool = False,
) -> np.ndarray:
    """
    Run ``predict_proba`` (classifiers) or ``predict`` (regressors) safely.

    Parameters
    ----------
    model : fitted sklearn-compatible model
    X : pd.DataFrame
        Feature matrix (only numeric columns are used).
    strip_names : bool
        If ``True``, convert *X* to a raw NumPy array so XGBoost cannot
        perform a feature-name validation check.

    Returns
    -------
    1-D NumPy array of predicted scores.
    """
    X_clean = X.select_dtypes(include=[np.number])
    data = X_clean.values if strip_names else X_clean

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(data)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba.max(axis=1)
        except AttributeError:
            pass  # fall through to predict()

    return model.predict(data)


def score_with_models(
    features_df: pd.DataFrame,
    models: Dict[str, Any],
    model_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Score *features_df* with every model in *models*.

    Uses a three-attempt fallback strategy for robustness against column-name
    mismatches between Python-generated features and R-trained models:

    1. Standard prediction with original column names.
    2. Column renaming (``pos0_A`` → ``A1``, etc.).
    3. Strip column names entirely and pass raw NumPy arrays.

    Parameters
    ----------
    features_df : pd.DataFrame
        Numeric feature matrix.
    models : dict
        Mapping of model name → fitted model, from :func:`load_models`.
    model_dir : Path, optional
        Unused; kept for API compatibility.
    logger : logging.Logger, optional

    Returns
    -------
    pd.DataFrame with one column per model, aligned to *features_df* by index.
    Missing values (``NaN``) indicate that all three attempts failed.
    """
    _log = logger or logging.getLogger(__name__)
    results = pd.DataFrame(index=features_df.index)

    if features_df.empty or not models:
        return results

    for name, model in models.items():
        # Attempt 1: standard prediction.
        try:
            results[name] = _predict_safe(model, features_df, strip_names=False)
            continue
        except Exception as exc1:
            err = str(exc1).lower()

        # Attempt 2: rename columns to R naming convention.
        if "feature" in err or "data did not have" in err or "mismatch" in err:
            df_fixed: Optional[pd.DataFrame] = None
            try:
                _log.info("[%s] Column name mismatch — renaming and retrying …", name)
                df_fixed = fix_column_names_for_xgboost(features_df)
                results[name] = _predict_safe(model, df_fixed, strip_names=False)
                continue
            except Exception:
                pass

            # Attempt 3: strip column names entirely.
            try:
                _log.warning("[%s] Renaming failed — passing raw numpy array …", name)
                df_to_use = df_fixed if df_fixed is not None else features_df
                results[name] = _predict_safe(model, df_to_use, strip_names=True)
                continue
            except Exception as exc3:
                _log.error("[SKIP] %s failed all 3 scoring attempts: %s", name, exc3)
                results[name] = np.nan
        else:
            _log.error("[SKIP] %s crashed unexpectedly: %s", name, exc1)
            results[name] = np.nan

    return results
