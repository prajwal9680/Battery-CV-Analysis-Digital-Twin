import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter, find_peaks
from simulator.model import ModelParams, simulate


def _smooth(y, win=21, poly=3):
    """Savitzky–Golay smoothing with safe fallbacks."""
    win = max(5, int(win) | 1)          # make odd and >=5
    poly = min(poly, win - 2)
    try:
        return savgol_filter(y, win, poly)
    except Exception:
        return y


def select_fit_region(E, I, peak_frac=0.20, margin_pts=12):
    """
    Auto-pick best region to fit:
      1) smooth -> find dominant |peak|
      2) take where |I| >= peak_frac * |peak|
      3) expand by margin
    Fallback: central 15–85% if peak detection fails.
    Returns: mask (len(E)), (lo, hi) indices
    """
    E = np.asarray(E)
    I = np.asarray(I)

    if len(E) < 30:
        mask = np.zeros_like(E, dtype=bool)
        i0, i1 = max(0, len(E)//4), min(len(E)-1, 3*len(E)//4)
        mask[i0:i1+1] = True
        return mask, (i0, i1)

    Is = _smooth(I, win=max(21, len(I)//15*2+1), poly=3)
    mag = np.abs(Is)

    # find strongest peak by magnitude
    peaks, _ = find_peaks(mag)
    if peaks.size == 0:
        # fall back to global max
        p = int(np.argmax(mag))
    else:
        p = int(peaks[np.argmax(mag[peaks])])

    peak_val = mag[p]
    if peak_val <= 0:
        # last fallback: middle slice
        i0, i1 = max(0, len(E)//6), min(len(E)-1, 5*len(E)//6)
        mask = np.zeros_like(E, dtype=bool); mask[i0:i1+1] = True
        return mask, (i0, i1)

    thresh = peak_frac * peak_val

    # expand from peak until mag falls below threshold
    i0 = p
    while i0 > 0 and mag[i0] >= thresh:
        i0 -= 1
    i1 = p
    while i1 < len(E)-1 and mag[i1] >= thresh:
        i1 += 1

    # margin
    i0 = max(0, i0 - margin_pts)
    i1 = min(len(E)-1, i1 + margin_pts)

    # if window is tiny, broaden
    if i1 - i0 < max(20, len(E)//10):
        span = max(20, len(E)//8)
        mid = p
        i0 = max(0, mid - span//2)
        i1 = min(len(E)-1, mid + span//2)

    mask = np.zeros_like(E, dtype=bool)
    mask[i0:i1+1] = True
    return mask, (i0, i1)


def fit_parameters(E_exp, I_exp, init):
    """
    Fit D, E0, and baseline offset b using robust least squares on an
    automatically selected region of the CV.
    Returns fit dictionary with 'mask' for visualization.
    """
    E_exp = np.asarray(E_exp).copy()
    I_exp = np.asarray(I_exp).copy()

    # ------- auto-select region to fit -------
    mask, (i0, i1) = select_fit_region(E_exp, I_exp, peak_frac=0.20, margin_pts=12)
    E_fit = E_exp[mask]
    I_fit = I_exp[mask]

    # parameter bounds: [D, E0, b]
    bounds = (
        [1e-11, -0.3,  -2e-4],
        [1e-9,   0.6,   2e-4]
    )
    x0 = np.array([
        float(init.get('D', 1e-10)),
        float(init.get('E0', 0.1)),
        0.0
    ])

    # --- simulation helper (interpolate to E_fit grid) ---
    def sim_curve(D, E0):
        params = ModelParams(
            A=init['A'], L=init['L'], Nx=int(init['Nx']), D=D,
            T=init['T'], n=int(init['n']), E0=E0, alpha=init['alpha'],
            E_start=init['E0'] - 0.1, E_vertex=init['E0'] + 0.2, E_final=init['E0'] - 0.1,
            scan_rate=init['scan_rate'], mode='reversible'
        )
        t, E_sim, I_sim = simulate(params, Nt=900)
        return np.interp(E_fit, E_sim, I_sim)

    # --- residuals with robust normalization and baseline offset ---
    def residuals(theta):
        D, E0, b = theta
        I_model = sim_curve(D, E0) + b
        # scale by local magnitude to balance forward/reverse segments
        eps = 1e-9 + np.maximum(1e-6, np.abs(I_fit))
        return (I_model - I_fit) / eps

    res = least_squares(
        residuals, x0, bounds=bounds,
        loss='soft_l1', f_scale=3e-4,
        xtol=1e-7, ftol=1e-7, max_nfev=80
    )

    # soft warning if solution sticks to bounds
    if np.any(np.isclose(res.x, bounds[0])) or np.any(np.isclose(res.x, bounds[1])):
        print("[fit] Warning: solution at parameter bounds. Check units or widen bounds.")

    D_fit, E0_fit, b_fit = map(float, res.x)
    I_fit_model = sim_curve(D_fit, E0_fit) + b_fit

    # compute RMSE on the selected region only
    rmse = float(np.sqrt(np.mean((I_fit_model - I_fit)**2)))

    # also return model prediction on full E_exp for plotting
    I_model_full = np.interp(E_exp, E_fit, I_fit_model)

    return {
        "D": D_fit,
        "E0": E0_fit,
        "b": b_fit,
        "rmse": rmse,
        "I_fit": I_model_full,   # model mapped to full E axis (for overlay)
        "mask": mask,            # boolean mask of chosen window
        "window_idx": (i0, i1)   # for debugging/optional display
    }
