from cv_twin.qc.preprocess import PreprocessConfig, load_and_preprocess

cfg = PreprocessConfig(
    smooth_window_pts=31,
    smooth_polyorder=3,
    spike_zscore_thresh=4.5,
    points_per_volt=800,
    baseline_tail_frac=0.12,
    voltage_scale=1.0,
    current_scale=1.0
)

prep = load_and_preprocess("cv_twin/data/sample/sample_cv.csv", cfg)
df_clean = prep["df_clean"]
segments_meta = prep["segments"]

print(df_clean.head())
from cv_twin.simulator.model import simulate_cv, CVParams
import numpy as np

# Detect one sweep to demo (use sweep_id==0)
df0 = df_clean[df_clean["sweep_id"] == 0].reset_index(drop=True)
E = df0["E_V"].values

# Estimate scan rate from preprocessed sweep (approx constant)
# v ≈ (E_end - E_start) / total_time; without time we approximate from grid density.
# Use your experimental scan rate if you know it; else this gives a workable default.
estimated_scan_rate = 0.05  # V/s; replace with your real scan rate if known

params = CVParams(
    Cdl=1e-3,
    R_s=2.0,
    k_w=0.0,          # try 5e-4 if you want diffusion tails
    E0=0.5,
    peak_width=0.035,
    peak_scale=5e-4,
    peak_sep=0.02,
    peak_asym=1.0,
)

I_model = simulate_cv(E, estimated_scan_rate, params)

# quick check
print("Simulated current stats:", float(np.min(I_model)), float(np.max(I_model)))
from cv_twin.fitting.fit_b1 import fit_cv, FitConfig

# 1) set your experimental scan rate here (example: 20 mV/s)
scan_rate = 0.02  # V/s  ← CHANGE to your real value

cfg_fit = FitConfig(
    sweeps=(0, 1),     # fit first forward & reverse
    scan_rate=scan_rate
)

result = fit_cv(df_clean, cfg_fit)

print("Fit success:", result["success"])
print("Message:", result["message"])
print("R2:", round(result["metrics"]["r2"], 4), "  RMSE:", result["metrics"]["rmse"])
print("Per-sweep:", result["per_sweep"])
print("Best params:", result["params"])
