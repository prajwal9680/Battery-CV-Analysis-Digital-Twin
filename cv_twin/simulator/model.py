# cv_twin/simulator/model.py
"""
CV Digital Twin — Step 2 (B1):
General, robust simulator combining
- Double-layer (capacitive / CPE-like) current
- Simple Warburg diffusion tail (t^{-1/2} since last vertex)
- Reversible redox peaks (anodic & cathodic) with adjustable width/offset
- Ohmic drop (R_s) solved by fixed-point iteration

Inputs: voltage array for a single sweep (monotonic), scan_rate (V/s), params
Outputs: modeled current array matching E-grid of preprocessed sweep.

Author: You + ChatGPT
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class CVParams:
    # Electrical / transport
    Cdl: float = 1e-3          # F (double-layer capacitance)
    R_s: float = 2.0           # ohm (solution/ohmic resistance)
    # Warburg-like diffusion tail coefficient: A * sqrt(s) / V
    k_w: float = 0.0           # set >0 if diffusion tails are visible
    # Redox peak (general reversible-ish)
    E0: float = 0.5            # V (formal potential)
    peak_width: float = 0.035  # V (std dev of Gaussian peak)
    peak_scale: float = 5e-4   # A (controls peak height)
    peak_sep: float = 0.02     # V separation of anodic vs cathodic peak centers
    peak_asym: float = 1.0     # anodic/cathodic amplitude ratio (1 = symmetric)
    # Baseline (small residual slope/offset after preprocessing)
    baseline_slope: float = 0.0   # A/V
    baseline_offset: float = 0.0  # A
    # Solver
    max_iter: int = 15
    tol: float = 1e-9
# Backwards compatibility: older UI code imports ModelParams
ModelParams = CVParams


def _scan_segments(E: np.ndarray) -> np.ndarray:
    """Return indices where scan direction flips (include start index 0)."""
    dE = np.gradient(E)
    sgn = np.sign(dE)
    sgn[sgn == 0] = np.nan
    sgn = np.where(np.isnan(sgn), np.interp(
        np.flatnonzero(np.isnan(sgn)),
        np.flatnonzero(~np.isnan(sgn)),
        sgn[~np.isnan(sgn)]
    ), sgn)
    flips = np.where(np.diff(np.signbit(sgn)))[0] + 1
    return np.unique(np.r_[0, flips, len(E)]).astype(int)


def _warburg_tail(n_pts: int, dt: float) -> np.ndarray:
    """
    t^{-1/2} kernel since vertex (reset at segment starts).
    Returns array w[t] ~ 1/sqrt(t+eps) normalized by dt^{1/2}.
    """
    t = np.arange(n_pts, dtype=float) * dt
    return 1.0 / np.sqrt(t + 1e-6)


def _faradaic_peaks(E: np.ndarray, params: CVParams, direction: int) -> np.ndarray:
    """
    Simple reversible-like anodic/cathodic Gaussian peaks.
    direction: +1 forward (increasing E), -1 reverse (decreasing E)
    """
    # centers separated around E0
    Ea = params.E0 + params.peak_sep
    Ec = params.E0 - params.peak_sep
    ga = np.exp(-0.5 * ((E - Ea) / params.peak_width) ** 2)
    gc = np.exp(-0.5 * ((E - Ec) / params.peak_width) ** 2)
    # sign: anodic +, cathodic -
    # direction slightly weights which peak dominates on each branch
    w_a = 0.6 if direction > 0 else 0.4
    w_c = 0.6 if direction < 0 else 0.4
    I = params.peak_scale * (params.peak_asym * w_a * ga - w_c * gc)
    return I


def simulate_sweep(E: np.ndarray, scan_rate: float, params: CVParams | Dict) -> np.ndarray:
    """
    Simulate current for a single monotonic sweep.
    - E: array of voltages (monotonic)
    - scan_rate: V/s for this sweep (assumed constant after preprocessing)
    - params: CVParams or dict
    """
    from dataclasses import fields as _dc_fields

if isinstance(params, dict):
    _allowed = {f.name for f in _dc_fields(CVParams)}
    _filtered = {k: v for k, v in params.items() if k in _allowed}
    params = CVParams(**_filtered)


    E = np.asarray(E, dtype=float)
    n = len(E)
    if n < 3:
        return np.zeros_like(E)

    # dt from constant scan rate and average dE
    dE = np.gradient(E)
    # keep sign of direction to label peaks
    direction = 1 if (E[-1] - E[0]) >= 0 else -1
    dt = np.mean(np.abs(dE)) / (abs(scan_rate) + 1e-12)

    # Build time-since-vertex vector (resets at scan flips)
    segments = _scan_segments(E)
    t_since = np.zeros(n, dtype=float)
    for i in range(len(segments) - 1):
        a, b = segments[i], segments[i + 1]
        tloc = np.arange(b - a, dtype=float) * dt
        t_since[a:b] = tloc

    # Capacitive current (Cdl * dE/dt). Use sign of local slope.
    I_cap = params.Cdl * scan_rate * np.sign(dE)

    # Warburg tail (optional)
    if params.k_w > 0:
        I_w = np.zeros(n, dtype=float)
        for i in range(len(segments) - 1):
            a, b = segments[i], segments[i + 1]
            w = _warburg_tail(b - a, dt)
            I_w[a:b] = params.k_w * np.sign(scan_rate) * w
    else:
        I_w = 0.0

    # Faradaic peaks
    I_f = _faradaic_peaks(E, params, direction)

    # Baseline
    I_base = params.baseline_offset + params.baseline_slope * (E - E[0])

    # Combine without ohmic, then solve for ohmic drop: E_eff = E - I*R_s.
    # Fixed-point iteration: I = f(E - R_s * I)
    I = I_cap + I_w + I_f + I_base
    if params.R_s > 0:
        I_old = I.copy()
        for _ in range(params.max_iter):
            E_eff = E - params.R_s * I_old
            # Update only the faradaic and baseline parts that depend on E (peaks shift with E)
            I_f_eff = _faradaic_peaks(E_eff, params, direction)
            I_new = params.Cdl * scan_rate * np.sign(dE) + I_w + I_f_eff + params.baseline_offset + params.baseline_slope * (E_eff - E_eff[0])
            if np.max(np.abs(I_new - I_old)) < params.tol:
                I_old = I_new
                break
            I_old = 0.5 * I_old + 0.5 * I_new  # damping for stability
        I = I_old

    return I


def simulate_cv(E: np.ndarray, scan_rate: float, params: CVParams | Dict) -> np.ndarray:
    """
    Convenience wrapper: same as simulate_sweep (kept for API symmetry
    if you later pass multiple sweeps).
    """
    return simulate_sweep(E, scan_rate, params)

# Back-compat wrapper: older code imports `simulate`
def simulate(E, scan_rate, params):
    return simulate_cv(E, scan_rate, params)
