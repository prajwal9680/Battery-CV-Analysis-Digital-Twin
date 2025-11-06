import numpy as np
from scipy.optimize import least_squares
from simulator.model import ModelParams, simulate

def fit_parameters(E_exp, I_exp, init):
    """
    Fit D and E0 using the reversible model. Returns dict with D, E0, rmse, I_fit.
    """
    E_exp = np.asarray(E_exp)
    I_exp = np.asarray(I_exp)

    def sim_curve(D, E0):
        p = ModelParams(
            A=float(init['A']), L=float(init['L']), Nx=int(init['Nx']), D=float(D),
            T=float(init['T']), n=int(init['n']), E0=float(E0), alpha=float(init['alpha']),
            CObulk=1e-3, CRbulk=0.0,
            E_start=float(E_exp.min()), E_vertex=float(E_exp.max()), E_final=float(E_exp.min()),
            scan_rate=float(init['scan_rate']), mode='reversible'
        )
        _, E_sim, I_sim = simulate(p, Nt=500)
        return np.interp(E_exp, E_sim, I_sim)

    def residuals(theta):
        D, E0 = theta
        I_on = sim_curve(D, E0)
        eps = 1e-9 + np.maximum(1e-6, np.abs(I_exp))
        return (I_on - I_exp) / eps

    x0 = np.array([float(init['D']), float(init['E0'])])
    bounds = ([1e-11, -0.2], [1e-9, 0.6])
    res = least_squares(residuals, x0, bounds=bounds, xtol=1e-7, ftol=1e-7, max_nfev=40)
    D_fit, E0_fit = map(float, res.x)
    # Warn if solution hits bounds (indicates poor fit or wrong units)
if np.any(np.isclose(res.x, res.bounds[0])) or np.any(np.isclose(res.x, res.bounds[1])):
    print("[fit] Warning: solution is at parameter bounds. Consider widening bounds or checking units/data.")

    I_fit = sim_curve(D_fit, E0_fit)
    rmse = float(np.sqrt(np.mean((I_fit - I_exp)**2)))

    return {'D': D_fit, 'E0': E0_fit, 'rmse': rmse, 'I_fit': I_fit}
