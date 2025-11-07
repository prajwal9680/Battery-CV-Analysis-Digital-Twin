# cv_twin/fitting/fit_b1.py
"""
Step 3 (B1): Robust bounded fitter for the CV Digital Twin.

- Uses scipy.optimize.least_squares with sensible bounds & fallbacks
- Auto-initializes parameters from data
- Fits across one or multiple sweeps (recommended: first forward & reverse)
- Returns best-fit params, metrics (RMSE, R2), and per-sweep residuals
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable
from scipy.optimize import least_squares

from cv_twin.simulator.model import simulate_cv, CVParams


@dataclass
class FitConfig:
    # which sweeps to use for fitting (ids from preprocessing)
    sweeps: Iterable[int] = (0, 1)        # pair of forward+reverse by default
    scan_rate: float = 0.05               # V/s (SET THIS to your experiment)
    # bounds (min, max) for parameter vector:
    # [Cdl, Rs, k_w, E0, peak_width, peak_scale, peak_sep, peak_asym, base_slope, base_off]
    bounds_lo: Tuple[float, ...] = (
        1e-6,   0.0,   0.0,   -10.0,  0.005, 1e-6, 0.0,   0.2,  -1e-2, -1e-2
    )
    bounds_hi: Tuple[float, ...] = (
        1e-1, 100.0, 1e-1,    10.0,   0.15,  1e-2, 0.20,  5.0,   1e-2,  1e-2
    )
    # solver
    robust_loss: str = "huber"            # options: 'linear','soft_l1','huber','cauchy'
    f_scale: float = 1e-3                 # scale for robust loss
    max_nfev: int = 1500


def _linear_baseline(E: np.ndarray, I: np.ndarray) -> Tuple[float, float]:
    """Least-squares linear fit I = a + b*E."""
    A = np.vstack([np.ones_like(E), E]).T
    beta, *_ = np.linalg.lstsq(A, I, rcond=None)
    a, b = beta
    return float(b), float(a)  # slope, offset


def _initial_guess(df_clean: pd.DataFrame, cfg: FitConfig) -> np.ndarray:
    """Heuristics to pick reasonable starting values from data."""
    # concat selected sweeps
    chunks = []
    for sid in cfg.sweeps:
        d = df_clean[df_clean["sweep_id"] == sid]
        if len(d) > 0:
            chunks.append(d[["E_V", "I_A"]].values)
    if not chunks:
        raise ValueError("Selected sweeps not found in df_clean.")
    D = np.vstack(chunks)
    E = D[:, 0]; I = D[:, 1]

    # baseline
    b_slope, b_off = _linear_baseline(E, I)

    # rough peak center & scale (remove baseline)
    I0 = I - (b_off + b_slope * E)
    idx = int(np.argmax(np.abs(I0)))
    E0 = float(E[idx])
    peak_scale = float(np.clip(np.percentile(np.abs(I0), 98), 1e-6, 5e-3))

    # Cdl ~ median(|I|) / |scan_rate|
    Cdl = float(np.clip(np.median(np.abs(I)) / max(abs(cfg.scan_rate), 1e-9), 1e-6, 1e-2))

    x0 = np.array([
        Cdl,        # Cdl
        2.0,        # Rs
        1e-4,       # k_w
        E0,         # E0
        0.04,       # peak_width
        peak_scale, # peak_scale
        0.02,       # peak_sep
        1.0,        # peak_asym
        b_slope,    # baseline_slope
        b_off       # baseline_offset
    ], dtype=float)

    # tighten E0 bounds to observed range
    Emin, Emax = float(np.min(E)), float(np.max(E))
    lo = np.array(cfg.bounds_lo, dtype=float).copy()
    hi = np.array(cfg.bounds_hi, dtype=float).copy()
    lo[3] = max(lo[3], Emin)
    hi[3] = min(hi[3], Emax)
    return x0, lo, hi


def _vec_to_params(x: np.ndarray) -> CVParams:
    return CVParams(
        Cdl=float(x[0]),
        R_s=float(x[1]),
        k_w=float(x[2]),
        E0=float(x[3]),
        peak_width=float(x[4]),
        peak_scale=float(x[5]),
        peak_sep=float(x[6]),
        peak_asym=float(x[7]),
        baseline_slope=float(x[8]),
        baseline_offset=float(x[9]),
    )


def _residual_vector(x: np.ndarray, data: List[Tuple[np.ndarray, np.ndarray]], scan_rate: float) -> np.ndarray:
    """Stack residuals for all sweeps: (I_model - I_data)."""
    params = _vec_to_params(x)
    res = []
    for E, I in data:
        I_model = simulate_cv(E, scan_rate, params)
        res.append(I_model - I)
    return np.concatenate(res)


def fit_cv(df_clean: pd.DataFrame, cfg: FitConfig = FitConfig()) -> Dict[str, object]:
    """
    Main entry:
    - df_clean: output of preprocessing with columns ['E_V','I_A','sweep_id']
    - cfg.scan_rate: REQUIRED to be set to your experiment's V/s
    Returns dict with:
      'x': best-fit vector
      'params': CVParams dataclass
      'metrics': {'rmse':..., 'r2':...}
      'per_sweep': [{'sweep_id':..,'rmse':..,'n':..}, ...]
    """
    if "E_V" not in df_clean or "I_A" not in df_clean or "sweep_id" not in df_clean:
        raise ValueError("df_clean must have E_V, I_A, sweep_id.")

    # Gather data
    data = []
    used_ids = []
    for sid in cfg.sweeps:
        d = df_clean[df_clean["sweep_id"] == sid].sort_values("E_V")
        if len(d) == 0:
            continue
        E = d["E_V"].to_numpy()
        I = d["I_A"].to_numpy()
        data.append((E, I))
        used_ids.append(int(sid))
    if not data:
        raise ValueError("No matching sweeps found for the ids in cfg.sweeps.")

    # Init & bounds
    x0, lo, hi = _initial_guess(df_clean[df_clean["sweep_id"].isin(used_ids)], cfg)

    # Solve
    res = least_squares(
        fun=_residual_vector,
        x0=x0,
        bounds=(lo, hi),
        args=(data, float(cfg.scan_rate)),
        loss=cfg.robust_loss,
        f_scale=cfg.f_scale,
        max_nfev=cfg.max_nfev,
        jac="2-point",
        verbose=0
    )

    x = res.x
    params = _vec_to_params(x)

    # Metrics
    y_true = np.concatenate([I for _, I in data])
    y_pred = np.concatenate([simulate_cv(E, cfg.scan_rate, params) for E, _ in data])
    resid = y_pred - y_true
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-30
    r2 = float(1.0 - ss_res / ss_tot)

    per_sweep = []
    for sid, (E, I) in zip(used_ids, data):
        Ip = simulate_cv(E, cfg.scan_rate, params)
        r = Ip - I
        per_sweep.append({
            "sweep_id": sid,
            "rmse": float(np.sqrt(np.mean(r ** 2))),
            "n": int(len(I))
        })

    return {
        "x": x,
        "params": params,
        "metrics": {"rmse": rmse, "r2": r2},
        "per_sweep": per_sweep,
        "success": bool(res.success),
        "message": res.message
    }
