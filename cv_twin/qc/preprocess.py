# preprocess.py
"""
CV Data Pre-processing Utilities (Step 1 - B1)
- Load CSV/TSV/XLSX
- Auto-detect columns (E vs I)
- Remove NaNs / unit normalization
- Switch-point aware de-noising (Savitzky–Golay)
- Spike trimming around vertex potentials
- Baseline correction (linear) per sweep
- Uniform resampling on a monotonic voltage grid
- Cycle and sweep segmentation with metadata

Author: You + ChatGPT
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# -----------------------------
# Config dataclass
# -----------------------------
@dataclass
class PreprocessConfig:
    # Smoothing
    smooth_window_pts: int = 21         # must be odd; auto-fixed if needed
    smooth_polyorder: int = 3
    # Spike trimming near switching points
    spike_zscore_thresh: float = 4.0    # higher => less trimming
    # Resampling
    points_per_volt: int = 600          # resolution of uniform grid
    # Baseline (linear) using tails
    baseline_tail_frac: float = 0.12    # 12% at each end used for baseline fit
    # Min segment length after cleaning
    min_points_segment: int = 120
    # Column name hints (optional)
    col_hint_voltage: Optional[str] = None
    col_hint_current: Optional[str] = None
    # Units: try to coerce to V and A
    voltage_scale: float = 1.0          # e.g., if mV provided set 1e-3
    current_scale: float = 1.0          # e.g., if mA provided set 1e-3

# -----------------------------
# Public API
# -----------------------------
def load_cv(filepath: str, cfg: PreprocessConfig = PreprocessConfig()) -> pd.DataFrame:
    """
    Load CV data from CSV/TSV/XLSX and return raw DataFrame with columns:
    ['E_V', 'I_A', 't_s'] where 't_s' may be NaN if not available.
    Tries to auto-detect column names and units.
    """
    if filepath.lower().endswith(".xlsx"):
        df = pd.read_excel(filepath)
    else:
        # Let pandas sniff delimiter
        df = pd.read_csv(filepath, engine="python")
    if df.empty:
        raise ValueError("Loaded file is empty.")

    # Normalize column names
    cols = {c.lower().strip(): c for c in df.columns}
    # Potential candidates
    voltage_candidates = [
        cfg.col_hint_voltage,
        "e/v", "e (v)", "potential (v)", "potential/v", "potential", "ewe/v", "voltage", "e",
        "ewe", "Ewe/V".lower()
    ]
    current_candidates = [
        cfg.col_hint_current,
        "i/a", "i (a)", "current (a)", "current/a", "current", "i", "j/a", "j (a)", "i/mA".lower(),
        "current (ma)".lower()
    ]
    time_candidates = ["time", "time (s)", "t/s", "t (s)", "t", "scan time(s)", "runtime (s)"]

    def pick(cands):
        for k in cands:
            if not k: 
                continue
            if k in cols: 
                return cols[k]
            # fuzzy contains
            for lc, orig in cols.items():
                if k in lc:
                    return orig
        return None

    vcol = pick(voltage_candidates)
    icol = pick(current_candidates)
    tcol = pick(time_candidates)

    if vcol is None or icol is None:
        raise ValueError(
            f"Could not find voltage/current columns. Found columns: {list(df.columns)}.\n"
            "Hint: pass col_hint_voltage/col_hint_current in PreprocessConfig."
        )

    out = pd.DataFrame({
        "E_V": df[vcol].astype(float) * cfg.voltage_scale,
        "I_A": df[icol].astype(float) * cfg.current_scale
    })
    if tcol is not None:
        out["t_s"] = pd.to_numeric(df[tcol], errors="coerce")
    else:
        out["t_s"] = np.nan

    # Drop NaNs
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["E_V", "I_A"])
    if len(out) < 50:
        raise ValueError("Too few points after initial cleaning.")
    return out.reset_index(drop=True)

def preprocess_cv(df_raw: pd.DataFrame, cfg: PreprocessConfig = PreprocessConfig()
                 ) -> Dict[str, object]:
    """
    Full cleaning pipeline:
      1) Segment into monotonic sweeps by sign of dE
      2) Smooth I(E) with Savitzky-Golay (switch-point aware)
      3) Remove spikes near vertex potentials using derivative z-score
      4) Baseline-correct per sweep using tail linear fit
      5) Resample each sweep to a uniform E grid
    Returns:
      {
        'df_clean': concatenated cleaned & resampled DataFrame with columns ['E_V','I_A','sweep_id','cycle_id'],
        'segments': list of segment metadata dicts,
        'cfg': cfg
      }
    """
    df = df_raw.copy().reset_index(drop=True)
    if "E_V" not in df or "I_A" not in df:
        raise ValueError("Input must have columns E_V and I_A.")

    # 1) Identify sweeps by sign changes in dE (switching points)
    dE = np.gradient(df["E_V"].values)
    sign = np.sign(dE)
    sign[sign == 0] = np.nan
    sign = pd.Series(sign).ffill().bfill().values

    switch_idx = np.where(np.diff(np.signbit(sign)))[0] + 1
    # Always include start and end
    cut_idx = np.unique(np.concatenate(([0], switch_idx, [len(df)])))
    segments = []
    for i in range(len(cut_idx) - 1):
        a, b = int(cut_idx[i]), int(cut_idx[i+1])
        seg = df.iloc[a:b].copy()
        if len(seg) < cfg.min_points_segment:
            continue
        # enforce monotonic order in E within this sweep
        if seg["E_V"].iloc[0] > seg["E_V"].iloc[-1]:
            seg = seg.sort_values("E_V")
            direction = "reverse"
        else:
            seg = seg.sort_values("E_V")
            direction = "forward"
        seg["sweep_local_id"] = i
        seg["direction"] = direction
        segments.append(seg.reset_index(drop=True))

    if not segments:
        raise ValueError("No valid sweeps detected. Try lowering min_points_segment.")

    # 2) Smooth I(E) per sweep (ensure odd window and <= length)
    def _smooth(y, window, poly):
        w = min(window, len(y) - (len(y)+1)%2)  # make it odd and <= len-1
        if w < 5:  # too small to smooth
            return y
        if w % 2 == 0:
            w -= 1
        try:
            return savgol_filter(y, window_length=w, polyorder=min(poly, w-1))
        except Exception:
            return y

    cleaned_segments = []
    meta = []
    for sid, seg in enumerate(segments):
        E = seg["E_V"].values
        I = seg["I_A"].values

        I_s = _smooth(I, cfg.smooth_window_pts, cfg.smooth_polyorder)

        # 3) Spike trimming near switching points via derivative z-score
        dIdE = np.gradient(I_s, E, edge_order=2)
        z = (dIdE - np.nanmean(dIdE)) / (np.nanstd(dIdE) + 1e-12)
        keep = np.abs(z) < cfg.spike_zscore_thresh
        # Guarantee we keep middle 80% to avoid over-trimming
        n = len(keep)
        core_keep = np.zeros(n, dtype=bool)
        core_keep[int(0.1*n):int(0.9*n)] = True
        keep = keep | core_keep

        E2 = E[keep]
        I2 = I_s[keep]
        if len(E2) < cfg.min_points_segment:
            # fallback to original if too aggressive
            E2, I2 = E, I_s

        # 4) Baseline correction (linear) using tails
        m = len(E2)
        k = max(int(cfg.baseline_tail_frac * m), 10)
        tail_idx = np.r_[np.arange(k), np.arange(m-k, m)]
        A = np.vstack([np.ones_like(E2[tail_idx]), E2[tail_idx]]).T
        try:
            beta, _, _, _ = np.linalg.lstsq(A, I2[tail_idx], rcond=None)
            c0, c1 = beta
            I2_corr = I2 - (c0 + c1*E2)
        except Exception:
            I2_corr = I2

        # 5) Resample to uniform E-grid
        E_min, E_max = np.min(E2), np.max(E2)
        # grid density based on points_per_volt
        density = max(cfg.points_per_volt, 200)
        n_pts = int(max(density * max(E_max - E_min, 0.1), cfg.min_points_segment))
        E_grid = np.linspace(E_min, E_max, n_pts)

        # robust interp allowing duplicates
        uniq_mask = np.diff(E2, prepend=E2[0]-1e-12) != 0
        E2u = E2[uniq_mask]
        I2u = I2_corr[uniq_mask]
        if len(E2u) < 10:
            continue
        f = interp1d(E2u, I2u, kind="linear", bounds_error=False, fill_value="extrapolate")
        I_grid = f(E_grid)

        seg_out = pd.DataFrame({
            "E_V": E_grid,
            "I_A": I_grid,
            "sweep_id": sid,
        })
        cleaned_segments.append(seg_out)
        meta.append({
            "sweep_id": sid,
            "direction": seg["direction"].iloc[0],
            "E_min": float(E_min),
            "E_max": float(E_max),
            "n_raw": int(len(seg)),
            "n_clean": int(len(E_grid))
        })

    if not cleaned_segments:
        raise ValueError("All sweeps were discarded. Relax thresholds in PreprocessConfig.")

    # Heuristic: pair consecutive forward+reverse as a cycle
    df_clean = pd.concat(cleaned_segments, ignore_index=True)
    # assign cycle_id as floor(sweep_id/2)
    df_clean["cycle_id"] = (df_clean["sweep_id"] // 2).astype(int)

    return {
        "df_clean": df_clean,
        "segments": meta,
        "cfg": cfg
    }

# -----------------------------
# Convenience helper for pipeline
# -----------------------------
def load_and_preprocess(filepath: str, cfg: PreprocessConfig = PreprocessConfig()
                       ) -> Dict[str, object]:
    df_raw = load_cv(filepath, cfg)
    return preprocess_cv(df_raw, cfg)
