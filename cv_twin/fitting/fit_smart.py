# cv_twin/fitting/fit_smart.py
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter, find_peaks
from simulator.model import ModelParams, simulate


# ---------- smoothing ----------
def _smooth(y, win=21, poly=3):
    """Savitzky–Golay smoothing with safe fallbacks."""
    win = max(5, int(win) | 1)   # odd, >=5
    poly = min(poly, win - 2)
    try:
        return savgol_filter(y, win, poly)
    except Exception:
        return y


# ---------- auto region selection ----------
def _select_fit_region(E, I, peak_frac=0.20, margin_pts=12):
    """
    Pick the best region to fit:
      1) smooth -> find dominant |peak|
      2) region where |I| >= peak_frac * |peak|
      3) expand by 'margin_pts'
    Fallbacks for small/noisy arrays.
    Returns: mask (bool array), (lo, hi) indices
    """
    E = np.asarray(E)
    I = np.asarray(I)

    n = len(E)
    if n < 30:
        mask = np.zeros(n, dtype=bool)
        i0, i1 = max(0, n // 4), min(n - 1, 3 * n // 4)
        mask[i0:i1 + 1] = True
        return mask, (i0, i1)

    Is = _smooth(I, win=max(21, (n // 15) * 2 + 1), poly=3)
    mag = np.abs(Is)

    peaks, _ = find_peaks(mag)
    if peaks.size == 0:
        p = int(np.argmax(mag))
    else:
        p = int(peaks[np.argmax(mag[peaks])])

    peak_val = mag[p]
    if peak_val <= 0:
        i0, i1 = max(0, n // 6), min(n - 1, 5 * n // 6)
        mask = np.zeros(n, dtype=bool)
        mask[i0:i1 + 1] = True
        return mask, (i0, i1)

    thresh = peak_frac * peak_val
    i0 = p
    while i0 > 0 and mag[i0] >= thresh:
        i0 -= 1
    i1 = p
    while i1 < n - 1 and mag[i1] >= thresh:
        i1 += 1

    i0 = max(0, i0 - margin_pts)
    i1 = min(n - 1, i1 + margin_pts)

    # ensure reasonable width
    min_span = max(20, n // 10)
    if i1 - i0 < min_span:
        span = max(20, n // 8)
        mid = p
        i0 = max(0, mid - span // 2)
        i1 = min(n - 1, mid + span // 2)

    mask = np.zeros(n, dtype=bool)
    mask[i0:i1 + 1] = True
    return mask, (i0, i1)


# ---------- single-cycle smart reversible fit (D, E0, baseline) ----------
def fit_parameters_smart(E_exp, I_exp, init):
    """
    Smart reversible fit on an auto-selected region.
    Returns: {D, E0, rmse, I_fit(full grid), mask, window_idx}
    """
    E_exp = np.asarray(E_exp)
    I_exp = np.asarray(I_exp)

    # auto window
    mask, (i0, i1) = _select_fit_region(E_exp, I_exp, peak_frac=0.20, margin_pts=12)
    E_fit = E_exp[mask]
    I_fit = I_exp[mask]

    # bounds: [D, E0, b]
    bounds = (
        [1e-11, -0.30, -2e-4],
        [1e-9,   0.60,  2e-4]
    )
    x0 = np.array([
        float(init.get("D", 1e-10)),
        float(init.get("E0", 0.10)),
        0.0   # baseline offset
    ], dtype=float)

    # simulate reversible model on E_fit grid
    def sim_curve(D, E0):
        params = ModelParams(
            A=init["A"], L=init["L"], Nx=int(init["Nx"]), D=D,
            T=init["T"], n=int(init["n"]), E0=E0, alpha=init["alpha"],
            E_start=init.get("E_start", init["E0"] - 0.1),
            E_vertex=init.get("E_vertex", init["E0"] + 0.2),
            E_final=init.get("E_final",  init["E0"] - 0.1),
            scan_rate=init["scan_rate"], mode="reversible", k0=init.get("k0", 5e-6)
        )
        t, E_sim, I_sim = simulate(params, Nt=900)
        return np.interp(E_fit, E_sim, I_sim)

    def residuals(theta):
        D, E0, b = theta
        I_model = sim_curve(D, E0) + b
        eps = 1e-9 + np.maximum(1e-6, np.abs(I_fit))
        return (I_model - I_fit) / eps

    res = least_squares(
        residuals, x0, bounds=bounds,
        loss="soft_l1", f_scale=3e-4, xtol=1e-7, ftol=1e-7, max_nfev=90
    )

    Df, E0f, bf = map(float, res.x)
    I_model_region = sim_curve(Df, E0f) + bf
    rmse = float(np.sqrt(np.mean((I_model_region - I_fit) ** 2)))

    # project model back to full E_exp for plotting
    # regenerate on full sim grid then interpolate to E_exp
    t_full, E_full, I_full = simulate(
        ModelParams(
            A=init["A"], L=init["L"], Nx=int(init["Nx"]), D=Df,
            T=init["T"], n=int(init["n"]), E0=E0f, alpha=init["alpha"],
            E_start=init.get("E_start", init["E0"] - 0.1),
            E_vertex=init.get("E_vertex", init["E0"] + 0.2),
            E_final=init.get("E_final",  init["E0"] - 0.1),
            scan_rate=init["scan_rate"], mode="reversible", k0=init.get("k0", 5e-6)
        ),
        Nt=900
    )
    I_fit_full = np.interp(E_exp, E_full, I_full) + bf

    return {
        "D": Df,
        "E0": E0f,
        "rmse": rmse,
        "I_fit": I_fit_full,
        "mask": mask,
        "window_idx": (i0, i1)
    }


# ---------- multi-cycle wrapper ----------
def fit_parameters_smart_multi(sweeps, init):
    """
    Fit all sweeps independently with Smart reversible fitter.
    sweeps: list of (E_sweep, I_sweep) arrays (already scaled to A).
    Returns:
      results: list of dicts per cycle (D, E0, rmse)
      overlays: list of dicts per cycle (E, I_exp, I_fit, mask)
      summary: dict with aggregates
    """
    results = []
    overlays = []

    for (E_s, I_s) in sweeps:
        out = fit_parameters_smart(E_s, I_s, init)
        results.append({"D": out["D"], "E0": out["E0"], "rmse": out["rmse"]})
        overlays.append({
            "E": E_s,
            "I_exp": I_s,
            "I_fit": out["I_fit"],
            "mask": out.get("mask", np.zeros_like(E_s, dtype=bool))
        })

    # aggregates
    D_vals = np.array([r["D"] for r in results], dtype=float)
    E0_vals = np.array([r["E0"] for r in results], dtype=float)
    RMSE_vals = np.array([r["rmse"] for r in results], dtype=float)

    summary = {
        "D_mean": float(np.mean(D_vals)),
        "D_std":  float(np.std(D_vals)),
        "E0_mean": float(np.mean(E0_vals)),
        "E0_std":  float(np.std(E0_vals)),
        "RMSE_mean": float(np.mean(RMSE_vals)),
        "RMSE_std":  float(np.std(RMSE_vals))
    }

    return results, overlays, summary
