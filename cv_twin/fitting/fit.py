import numpy as np
from scipy.optimize import least_squares
from simulator.model import ModelParams, simulate


def fit_parameters(E_exp, I_exp, init):
    """
    Fit D, E0, and baseline offset 'b' by minimizing the difference
    between simulated and experimental current.
    Uses robust least squares to handle noise and outliers.
    """

    # -------- parameter bounds --------
    # [D,     E0,     b]
    bounds = (
        [1e-11, -0.3,  -2e-4],   # lower bounds
        [1e-9,   0.6,   2e-4]    # upper bounds
    )

    # initial guess
    x0 = np.array([
        float(init.get('D', 1e-10)),   # diffusion coefficient
        float(init.get('E0', 0.1)),    # formal potential
        0.0                             # baseline offset
    ])

    # -------- simulation helper --------
    def sim_curve(D, E0):
        params = ModelParams(
            A=init['A'], L=init['L'], Nx=int(init['Nx']), D=D,
            T=init['T'], n=int(init['n']), E0=E0, alpha=init['alpha'],
            E_start=init['E0'] - 0.1, E_vertex=init['E0'] + 0.2, E_final=init['E0'] - 0.1,
            scan_rate=init['scan_rate'], mode='reversible'
        )
        t, E, I_sim = simulate(params, Nt=800)
        return np.interp(E_exp, E, I_sim)  # interpolate to experimental grid

    # -------- residuals for optimizer --------
    def residuals(theta):
        D, E0, b = theta
        I_model = sim_curve(D, E0) + b
        eps = 1e-9 + np.maximum(1e-6, np.abs(I_exp))
        return (I_model - I_exp) / eps

    # -------- fit using robust least squares --------
    res = least_squares(
        residuals, x0, bounds=bounds,
        loss='soft_l1', f_scale=3e-4,
        xtol=1e-7, ftol=1e-7, max_nfev=60
    )

    # warn if solution is at bounds
    if np.any(np.isclose(res.x, bounds[0])) or np.any(np.isclose(res.x, bounds[1])):
        print("[fit] Warning: solution at parameter bounds. Check units or widen bounds.")

    # extract results
    D_fit, E0_fit, b_fit = map(float, res.x)
    I_fit = sim_curve(D_fit, E0_fit) + b_fit

    # RMSE
    rmse = float(np.sqrt(np.mean((I_fit - I_exp)**2)))

    return {
        'D': D_fit,
        'E0': E0_fit,
        'b': b_fit,
        'rmse': rmse,
        'I_fit': I_fit
    }
