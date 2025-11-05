import numpy as np

def triangle_wave_times(E_start, E_vertex, E_final, v):
    """Return the time points for the forward and reverse sweeps."""
    t1 = abs(E_vertex - E_start) / v
    t2 = t1 + abs(E_final - E_vertex) / v
    return t1, t2

def triangle_wave(t, E_start, E_vertex, E_final, v):
    """Triangle wave potential profile for CV."""
    t1, t2 = triangle_wave_times(E_start, E_vertex, E_final, v)
    if t <= t1:
        return E_start + np.sign(E_vertex - E_start) * v * t
    elif t <= t2:
        return E_vertex + np.sign(E_final - E_vertex) * v * (t - t1)
    else:
        return E_final

