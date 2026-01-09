# treecrispr/models.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd
import logging
import joblib 

# --- 1. COLUMN RENAMING LOGIC ---
def fix_column_names_for_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns to match R-based XGBoost models.
    Converts 'pos0_A' -> 'A1' and 'di0_AA' -> 'AA1'.
    """
    new_names = {}
    for col in df.columns:
        # Case 1: Single Nucleotides (pos0_A -> A1)
        if col.startswith("pos") and "_" in col:
            parts = col.split("_")
            if len(parts) == 2 and parts[0][3:].isdigit():
                idx = int(parts[0][3:])
                seq_part = parts[1]
                new_names[col] = f"{seq_part}{idx + 1}"

        # Case 2: Dinucleotides (di0_AA -> AA1)
        elif col.startswith("di") and "_" in col:
            parts = col.split("_")
            if len(parts) == 2 and parts[0][2:].isdigit():
                idx = int(parts[0][2:])
                seq_part = parts[1]
                new_names[col] = f"{seq_part}{idx + 1}"
    
    if new_names:
        return df.rename(columns=new_names)
    return df

# --- 2. LOADING MODELS (UPDATED FOR CLEAN NAMES) ---
def load_models(model_dir: Path, logger=None) -> Dict[str, object]:
    """Load all .pkl or .joblib models from the directory."""
    models = {}
    if not model_dir or not model_dir.exists():
        if logger: logger.warning(f"Model directory missing: {model_dir}")
        return models

    for p in sorted(model_dir.glob("*.pkl")) + sorted(model_dir.glob("*.joblib")):
        try:
            model = joblib.load(p)
            
            # --- CLEAN NAME LOGIC START ---
            # Remove _xgb_clf, _xgb, or _clf from the filename
            clean_name = p.stem.replace("_xgb_clf", "").replace("_xgb", "").replace("_clf", "")
            # --- CLEAN NAME LOGIC END ---

            models[clean_name] = model
            if logger: logger.info(f"Loaded model: {clean_name}")
        except Exception as e:
            if logger: logger.error(f"Failed to load {p.name}: {e}")
            
    if logger: logger.info(f"Total models loaded: {len(models)} from {model_dir}")
    return models

# --- 3. PREDICTION ENGINE ---
def _predict_safe(model, X: pd.DataFrame, strip_names=False):
    """
    Helper to run prediction. 
    If strip_names=True, converts to numpy array to bypass feature name checks.
    """
    # Filter to numeric only
    X_clean = X.select_dtypes(include=[np.number])
    
    if strip_names:
        # Nuclear Option: Pass raw numpy array. XGBoost CANNOT check names here.
        data_to_pass = X_clean.values
    else:
        data_to_pass = X_clean

    # Try Predict Proba (Classifier)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(data_to_pass)
            # Return class 1 probability
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba.max(axis=1)
        except AttributeError:
            pass # Fallback to .predict
            
    # Try Standard Predict (Regressor or generic)
    return model.predict(data_to_pass)

def score_with_models(features_df: pd.DataFrame, models: Dict[str, object], model_dir=None, logger=None) -> pd.DataFrame:
    """
    Robust Scoring Loop with Auto-Correction (Triple Safety Net).
    """
    results = pd.DataFrame(index=features_df.index)
    
    if features_df.empty or not models:
        return results

    for name, model in models.items():
        try:
            # --- ATTEMPT 1: Standard Prediction ---
            # Try passing the features exactly as they are.
            pred = _predict_safe(model, features_df, strip_names=False)
            results[name] = pred

        except Exception as e1:
            # Catch errors like "feature_names mismatch" or ValueError
            err_msg = str(e1)
            
            # If error suggests mismatch, try fixing names
            if "feature" in err_msg.lower() or "data did not have" in err_msg.lower() or "mismatch" in err_msg.lower():
                
                try:
                    # --- ATTEMPT 2: Rename Columns (di0_AA -> AA1) ---
                    if logger: logger.info(f"[{name}] Name mismatch detected. Renaming columns and retrying...")
                    
                    df_fixed = fix_column_names_for_xgboost(features_df)
                    pred = _predict_safe(model, df_fixed, strip_names=False)
                    results[name] = pred
                    
                except Exception as e2:
                    # --- ATTEMPT 3: Nuclear Option (Strip Names) ---
                    if logger: logger.warning(f"[{name}] Renaming failed. Forcing Numpy array (ignoring names)...")
                    
                    try:
                        # Use df_fixed if available, else original, convert to raw numbers
                        df_to_use = df_fixed if 'df_fixed' in locals() else features_df
                        pred = _predict_safe(model, df_to_use, strip_names=True)
                        results[name] = pred
                        
                    except Exception as e3:
                        if logger: logger.error(f"[SKIP] {name} failed all 3 attempts. Error: {e3}")
                        results[name] = np.nan
            else:
                # Some other error (like wrong shape)
                if logger: logger.error(f"[SKIP] {name} crashed unexpectedly. Error: {e1}")
                results[name] = np.nan

    return results