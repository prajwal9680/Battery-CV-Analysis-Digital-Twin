import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter, find_peaks
from simulator.model import ModelParams, simulate

# -------- helpers --------
def _smooth(y, win=21, poly=3):
    win = max(5, int(win) | 1)
    poly = min(poly, win - 2)
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, win, poly)
    except Exception:
        return y

def _select_fit_region(E, I, peak_frac=0.20, margin_pts=12):
    """Auto-pick region around dominant peak to reduce bias from tails/baseline."""
    E = np.asarray(E); I = np.asarray(I)
    if len(E) < 30:
        m = np.zeros_like(E, bool); m[ max(0,len(E)//4) : min(len(E)-1,3*len(E)//4)+1 ] = True
        return m, (np.argmax(m>0), len(E)-1-np.argmax(m[::-1]>0))
    Is = _smooth(I, max(21, (len(I)//15)*2+1), 3)
    mag = np.abs(Is)
    peaks, _ = find_peaks(mag)
    p = int(peaks[np.argmax(mag[peaks])]) if peaks.size else int(np.argmax(mag))
    thresh = peak_frac * max(1e-12, mag[p])
    i0, i1 = p, p
    while i0 > 0 and mag[i0] >= thresh: i0 -= 1
    while i1 < len(E)-1 and mag[i1] >= thresh: i1 += 1
    i0 = max(0, i0 - margin_pts); i1 = min(len(E)-1, i1 + margin_pts)
    if i1 - i0 < max(20, len(E)//10):
        span = max(20, len(E)//8); mid = p
        i0 = max(0, mid - span//2); i1 = min(len(E)-1, mid + span//2)
    mask = np.zeros_like(E, bool); mask[i0:i1+1] = True
    return mask, (i0, i1)

# -------- BV + capacitive fit --------
def fit_parameters_bv(E_exp, I_exp, init):
    """
    Fit [D, E0, k0, CdlA, b] using BV boundary + capacitive current:
        I_total = I_faradaic_BV + CdlA * dE/dt + b
    where CdlA is 'double-layer capacitance * area' in A/(V/s).
    Returns dict with I_fit on the full E grid and the selected window mask.
    """
    E_exp = np.asarray(E_exp); I_exp = np.asarray(I_exp)

    # auto window (same as reversible fitter)
    mask, (i0, i1) = _select_fit_region(E_exp, I_exp, peak_frac=0.20, margin_pts=12)
    E_fit = E_exp[mask]; I_fit = I_exp[mask]

    # bounds:        D        E0      k0        CdlA        b
    bounds = ([1e-11,  -0.3,  1e-7,   0.0,      -2e-4],
              [1e-9,    0.6,  1e-2,   5e-3,      2e-4])

    x0 = np.array([
        float(init.get('D', 1e-10)),
        float(init.get('E0', 0.10)),
        float(init.get('k0', 5e-6)),
        5e-5,      # CdlA  (A/(V/s)) ~ 50 µA per 1 V/s
        0.0        # baseline
    ], dtype=float)

    # simulate BV once per parameter set; add capacitive term from dE/dt
    def simulate_bv(D, E0, k0):
        params = ModelParams(
            A=init['A'], L=init['L'], Nx=int(init['Nx']), D=D,
            T=init['T'], n=int(init['n']), E0=E0, alpha=init['alpha'],
            E_start=init['E_start'] if 'E_start' in init else (init['E0'] - 0.1),
            E_vertex=init['E_vertex'] if 'E_vertex' in init else (init['E0'] + 0.2),
            E_final=init['E_final'] if 'E_final' in init else (init['E0'] - 0.1),
            scan_rate=init['scan_rate'], mode='butler-volmer', k0=k0
        )
        t, E_sim, I_far = simulate(params, Nt=900)   # A
        dEdt = np.gradient(E_sim, t)                 # V/s (signed during forward/reverse)
        return t, E_sim, I_far, dEdt

    # cache to avoid double sim in residuals
    cache = {"theta": None, "E_sim": None, "I_far": None, "dEdt": None}
    def model_on_Efit(D, E0, k0, CdlA, b):
        theta = (D, E0, k0)
        if cache["theta"] != theta:
            _, E_sim, I_far, dEdt = simulate_bv(D, E0, k0)
            cache.update(theta=theta, E_sim=E_sim, I_far=I_far, dEdt=dEdt)
        I_cap = CdlA * cache["dEdt"]
        I_tot = cache["I_far"] + I_cap + b
        return np.interp(E_fit, cache["E_sim"], I_tot)

    def residuals(theta):
        D, E0, k0, CdlA, b = theta
        I_model = model_on_Efit(D, E0, k0, CdlA, b)
        eps = 1e-9 + np.maximum(1e-6, np.abs(I_fit))
        return (I_model - I_fit) / eps

    res = least_squares(
        residuals, x0, bounds=bounds,
        loss='soft_l1', f_scale=3e-4, xtol=1e-7, ftol=1e-7, max_nfev=120
    )

    if np.any(np.isclose(res.x, bounds[0])) or np.any(np.isclose(res.x, bounds[1])):
        print("[fit-bv] Warning: solution at bounds. Consider widening bounds or checking units.")

    Df, E0f, k0f, CdlAf, bf = map(float, res.x)
    # build model on the full experimental E axis for plotting
    _, E_sim, I_far, dEdt = simulate_bv(Df, E0f, k0f)
    I_tot = I_far + CdlAf * dEdt + bf
    I_fit_full = np.interp(E_exp, E_sim, I_tot)
    rmse = float(np.sqrt(np.mean((I_fit - I_fit_full[mask])**2)))

    return {
        "D": Df, "E0": E0f, "k0": k0f, "CdlA": CdlAf, "b": bf,
        "rmse": rmse,
        "I_fit": I_fit_full,
        "mask": mask, "window_idx": (i0, i1)
    }
