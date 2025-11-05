import numpy as np
from dataclasses import dataclass
from typing import Literal
from .waveform import triangle_wave, triangle_wave_times

# Constants
F = 96485.33212   # Faraday's constant (C/mol e-)
R = 8.314462618   # Gas constant (J/mol K)

@dataclass
class ModelParams:
    A: float = 1.0e-4        # Electrode area (m²)
    L: float = 1.5e-4        # Diffusion layer thickness (m)
    Nx: int = 90             # Number of spatial points
    D: float = 1.0e-10       # Diffusion coefficient (m²/s)
    T: float = 298.15        # Temperature (K)
    n: int = 1               # Number of electrons
    E0: float = 0.10         # Formal potential (V)
    alpha: float = 0.5       # Transfer coefficient
    CObulk: float = 1.0e-3   # Bulk oxidized species (mol/m³)
    CRbulk: float = 0.0      # Bulk reduced species (mol/m³)

    # CV settings
    E_start: float = 0.00
    E_vertex: float = 0.40
    E_final: float = 0.00
    scan_rate: float = 0.05   # V/s

    # Mode selection
    mode: Literal['reversible', 'butler-volmer'] = 'reversible'
    k0: float = 5.0e-6        # Standard rate constant (for BV only)

def build_grid(L, Nx):
    x = np.linspace(0.0, L, Nx)
    dx = x[1] - x[0]
    return x, dx

def diffusion_laplacian(Nx, dx, D):
    """Build Laplacian matrix for diffusion."""
    lap = np.zeros((Nx, Nx))
    for i in range(1, Nx - 1):
        lap[i, i - 1] = 1.0
        lap[i, i] = -2.0
        lap[i, i + 1] = 1.0
    lap[Nx - 1, Nx - 2] = 1.0
    lap[Nx - 1, Nx - 1] = -1.0
    return (D / dx ** 2) * lap

def rhs_reversible(t, y, p, lap, dx):
    """Right-hand side for reversible boundary condition (Nernst equilibrium)."""
    Nx = p.Nx
    CO = y[:Nx].copy()
    CR = y[Nx:].copy()

    E = triangle_wave(t, p.E_start, p.E_vertex, p.E_final, p.scan_rate)

    beta = (p.n * F) / (R * p.T)
    ratio = np.exp(beta * (E - p.E0))
    Csum = max(p.CObulk + p.CRbulk, 1e-12)

    # Nernst at electrode surface
    CO[0] = Csum * ratio / (1 + ratio)
    CR[0] = Csum / (1 + ratio)

    dCOdt = lap @ CO
    dCRdt = lap @ CR

    # Flux and current
    J = -p.D * (CO[1] - CO[0]) / dx
    I = p.n * F * p.A * J

    return np.concatenate([dCOdt, dCRdt]), I, E

def rhs_butler_volmer(t, y, p, lap, dx):
    """Right-hand side for Butler–Volmer kinetic boundary."""
    Nx = p.Nx
    CO = y[:Nx].copy()
    CR = y[Nx:].copy()

    E = triangle_wave(t, p.E_start, p.E_vertex, p.E_final, p.scan_rate)
    Csum = max(p.CObulk + p.CRbulk, 1e-12)

    # Solve nonlinear surface condition
    CO1 = CO[1]
    CO0, CR0, J = _solve_surface_BV(Csum, CO1, p, dx, E)
    CO[0] = CO0
    CR[0] = CR0

    dCOdt = lap @ CO
    dCRdt = lap @ CR

    I = p.n * F * p.A * J
    return np.concatenate([dCOdt, dCRdt]), I, E

def _solve_surface_BV(Csum, CO1, p, dx, E):
    """Nonlinear solve for surface concentrations using charge transfer kinetics."""
    from math import exp
    beta = (p.n * F) / (R * p.T)
    eta = E - p.E0

    em = exp(-p.alpha * beta * eta)
    ep = exp((1 - p.alpha) * beta * eta)

    def f(CO0):
        Jdiff = -p.D * (CO1 - CO0) / dx
        Jbv = p.k0 * (CO0 * em - (Csum - CO0) * ep)
        return Jdiff - Jbv

    a, b = 0.0, Csum
    for _ in range(60):
        m = 0.5 * (a + b)
        if f(a) * f(m) <= 0:
            b = m
        else:
            a = m
    CO0 = 0.5 * (a + b)
    CR0 = Csum - CO0
    J = -p.D * (CO1 - CO0) / dx
    return CO0, CR0, J

def simulate(p: ModelParams, Nt=500):
    """Run the CV simulation and return (time, E(t), I(t))."""
    from scipy.integrate import solve_ivp

    x, dx = build_grid(p.L, p.Nx)
    lap = diffusion_laplacian(p.Nx, dx, p.D)

    t1, t2 = triangle_wave_times(p.E_start, p.E_vertex, p.E_final, p.scan_rate)
    t_end = t2
    t_eval = np.linspace(0, t_end, Nt)

    y0 = np.zeros(2 * p.Nx)
    y0[:p.Nx] = p.CObulk
    y0[p.Nx:] = p.CRbulk

    currents = []
    potentials = []

    if p.mode == 'reversible':
        def _rhs(t, y):
            dydt, I, E = rhs_reversible(t, y, p, lap, dx)
            currents.append(I)
            potentials.append(E)
            return dydt
    else:
        def _rhs(t, y):
            dydt, I, E = rhs_butler_volmer(t, y, p, lap, dx)
            currents.append(I)
            potentials.append(E)
            return dydt

    solve_ivp(_rhs, [0, t_end], y0, t_eval=t_eval, method='BDF', atol=1e-8, rtol=1e-5)

    I = np.array(currents[:len(t_eval)])
    E = np.array(potentials[:len(t_eval)])

    return t_eval, E, I
