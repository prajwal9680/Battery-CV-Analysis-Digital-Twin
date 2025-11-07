# cv_twin/ui/app.py
import sys
from pathlib import Path

# Ensure imports work regardless of how Streamlit runs
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from simulator.model import ModelParams, simulate
from fitting.preprocess import load_cv_csv, segment_sweeps
from fitting.fit_smart import fit_parameters_smart, fit_parameters_smart_multi
from qc.qc_rules import qc_status

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="CV Digital Twin", layout="wide")
st.title("⚡ CV Digital Twin — Smart Fit (Multi-cycle)")

st.caption(
    "Loads CV CSV, segments cycles, auto-selects the best region per cycle, "
    "fits a reversible model with baseline, and shows D, E0, RMSE per cycle. "
    "Use the unit selector if your CSV is in mA or µA."
)

# -----------------------------
# Defaults
# -----------------------------
default = dict(
    A=1.0e-4, L=1.5e-4, Nx=90, D=1.0e-10, E0=0.10, T=298.15, n=1, alpha=0.5,
    E_start=0.00, E_vertex=0.40, E_final=0.00, scan_rate=0.05, k0=5e-6
)

# -----------------------------
# Helpers
# -----------------------------
def run_sim(params, Nt=500):
    p = ModelParams(
        A=float(params["A"]), L=float(params["L"]), Nx=int(params["Nx"]), D=float(params["D"]),
        T=float(params["T"]), n=int(params["n"]), E0=float(params["E0"]), alpha=float(params["alpha"]),
        E_start=float(params["E_start"]), E_vertex=float(params["E_vertex"]), E_final=float(params["E_final"]),
        scan_rate=float(params["scan_rate"]), mode="reversible", k0=float(params["k0"])
    )
    return simulate(p, Nt=Nt)

def unit_scale(label_key: str):
    choice = st.selectbox("Current units in CSV", ["A", "mA", "µA"], index=0, key=label_key)
    return {"A": 1.0, "mA": 1e-3, "µA": 1e-6}[choice]

def qc_card(status, issues):
    color  = {"PASS":"#E8F5E9","WARN":"#FFF8E1","FAIL":"#FFEBEE"}[status]
    border = {"PASS":"#66BB6A","WARN":"#FFB300","FAIL":"#E53935"}[status]
    items = "".join(f"<li>{it}</li>" for it in issues) if issues else "<li>No issues detected</li>"
    st.markdown(
        "<div style='border:1px solid "+border+"; background:"+color+"; padding:12px; border-radius:10px;'>"
        f"<b>QC Status: {status}</b><br/><ul>{items}</ul></div>",
        unsafe_allow_html=True
    )

# -----------------------------
# Sidebar
# -----------------------------
mode_view = st.sidebar.radio("View", ["Simulation", "Smart Fit (multi-cycle)"])

# -----------------------------
# Simulation
# -----------------------------
if mode_view == "Simulation":
    st.sidebar.subheader("Controls")
    scan = st.sidebar.slider("Scan rate (V/s)", 0.01, 0.2, float(default["scan_rate"]))
    E_range = st.sidebar.slider("Potential range (V)", 0.2, 1.0, float(default["E_vertex"]))

    params = default.copy()
    params["scan_rate"] = float(scan)
    params["E_vertex"]  = float(E_range)

    if st.button("Run Simulation", use_container_width=True):
        t, E, I = run_sim(params, Nt=500)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(E, I * 1e3, linewidth=1.8)
        ax.set_xlabel("Potential / V"); ax.set_ylabel("Current / mA"); ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)

# -----------------------------
# Smart Fit (multi-cycle)
# -----------------------------
else:
    st.subheader("Upload & Smart-Fit")
    file = st.file_uploader("Upload CSV (time_s, potential_V, current_A)", type=["csv"])
    scale = unit_scale("units_smart")

    with st.expander("Model/Init parameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            A  = st.number_input("Area A (m²)", value=float(default["A"]), format="%.6e")
            L  = st.number_input("Diffusion length L (m)", value=float(default["L"]), format="%.6e")
            Nx = st.number_input("Grid points Nx", value=int(default["Nx"]), step=10)
        with col2:
            D0 = st.number_input("Init D (m²/s)", value=float(default["D"]), format="%.3e")
            E00 = st.number_input("Init E0 (V)", value=float(default["E0"]), format="%.3f")
            T  = st.number_input("Temperature (K)", value=float(default["T"]))
        with col3:
            n  = st.number_input("Electrons (n)", value=int(default["n"]), step=1)
            alpha = st.number_input("Alpha", value=float(default["alpha"]))
            scan_rate = st.number_input("Scan rate (V/s)", value=float(default["scan_rate"]))

    if file is not None and st.button("Run Smart Fit (all cycles)", use_container_width=True):
        df = load_cv_csv(file)
        sweeps = segment_sweeps(df)  # list of dataframes per sweep

        if len(sweeps) == 0:
            st.error("No sweeps detected. Check your CSV columns and values.")
        else:
            # build init param pack
            init_common = {
                "A": A, "L": L, "Nx": int(Nx),
                "D": D0, "E0": E00, "T": T, "n": int(n),
                "alpha": alpha, "scan_rate": scan_rate,
                "E_start": default["E_start"], "E_vertex": default["E_vertex"], "E_final": default["E_final"],
                "k0": default["k0"]
            }

            # convert each sweep to (E, I) arrays in Amps
            E_list = []
            I_list = []
            for dfi in sweeps:
                E_list.append(dfi["potential_V"].values)
                I_list.append(dfi["current_A"].values * scale)

            sweeps_tuple = list(zip(E_list, I_list))
            results, overlays, summary = fit_parameters_smart_multi(sweeps_tuple, init_common)

            # Plot: each cycle experimental vs fit, shaded fit window
            fig, ax = plt.subplots(figsize=(7.0, 4.4))
            for i, ov in enumerate(overlays, start=1):
                E = ov["E"]; I_exp = ov["I_exp"]; I_fit = ov["I_fit"]
                ax.plot(E, I_exp * 1e3, linewidth=1.0, alpha=0.85, label=f"Cycle {i} exp")
                ax.plot(E, I_fit * 1e3, linewidth=1.6, alpha=0.95, label=f"Cycle {i} fit")
                # Shade window if lengths match
                try:
                    ylo = min(np.min(I_exp*1e3), np.min(I_fit*1e3)) - 0.1
                    yhi = max(np.max(I_exp*1e3), np.max(I_fit*1e3)) + 0.1
                    ax.fill_between(E, ylo, yhi, where=ov.get("mask", np.zeros_like(E, bool)),
                                    alpha=0.07, step="mid")
                except Exception:
                    pass
            ax.set_xlabel("Potential / V"); ax.set_ylabel("Current / mA")
            ax.grid(True, alpha=0.3); ax.legend(ncols=2, fontsize=8)
            st.pyplot(fig, clear_figure=True)

            # Simple outputs table (D, E0, RMSE) per cycle
            st.subheader("Per-cycle results (simple)")
            rows = []
            for i, r in enumerate(results, start=1):
                rows.append({
                    "Cycle": i,
                    "D (m²/s)": f"{r['D']:.2e}",
                    "E0 (V)": f"{r['E0']:.3f}",
                    "RMSE (A)": f"{r['rmse']:.2e}"
                })
            st.table(rows)

            # QC: use overall RMSE vs peak of the first cycle as a simple check
            try:
                peak = max(1e-9, float(np.max(np.abs(I_list[0]))))
                rmse_pct = 100.0 * float(np.mean([r["rmse"] for r in results])) / peak
                status, issues = qc_status(results[0]["D"], rmse_pct)
                qc_card(status, issues)
                st.caption(
                    f"Summary — D(mean)={summary['D_mean']:.2e} ± {summary['D_std']:.2e} m²/s, "
                    f"E0(mean)={summary['E0_mean']:.3f} ± {summary['E0_std']:.3f} V, "
                    f"RMSE(mean)={summary['RMSE_mean']:.2e} A"
                )
            except Exception:
                pass
