# cv_twin/ui/app.py
import sys
from pathlib import Path
from fitting.fit_bv import fit_parameters_bv

# Make package imports work no matter how Streamlit runs
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from simulator.model import ModelParams, simulate
from fitting.preprocess import load_cv_csv, segment_sweeps
from fitting.fit import fit_parameters
from qc.qc_rules import qc_status


# =========================
# Page & Theme
# =========================
st.set_page_config(page_title="CV Digital Twin", layout="wide")
st.title("⚡ CV Digital Twin – General Electrochemistry")

st.caption(
    "Simulate cyclic voltammetry, fit experimental curves (reversible model), "
    "and assess data quality. Use the unit selector if your CSV is in mA or µA."
)


# =========================
# Defaults
# =========================
default = dict(
    A=1.0e-4,          # m^2 electrode area
    L=1.5e-4,          # m diffusion length
    Nx=90,             # grid points
    D=1.0e-10,         # m^2/s diffusion coefficient
    E0=0.10,           # V formal potential
    T=298.15,          # K temperature
    n=1,               # electrons transferred
    alpha=0.5,         # transfer coefficient
    E_start=0.00,      # V
    E_vertex=0.40,     # V
    E_final=0.00,      # V
    scan_rate=0.05,    # V/s
    k0=5e-6            # m/s (BV only, used for simulation)
)


# =========================
# Helpers
# =========================
def run_sim(params, Nt=500, boundary="reversible"):
    """Run CV simulation with current parameters."""
    p = ModelParams(
        A=float(params["A"]), L=float(params["L"]), Nx=int(params["Nx"]), D=float(params["D"]),
        T=float(params["T"]), n=int(params["n"]), E0=float(params["E0"]), alpha=float(params["alpha"]),
        E_start=float(params["E_start"]), E_vertex=float(params["E_vertex"]), E_final=float(params["E_final"]),
        scan_rate=float(params["scan_rate"]), mode=boundary, k0=float(params["k0"])
    )
    return simulate(p, Nt=Nt)

def annotate_peaks(E, I):
    i_max = int(np.argmax(I)); i_min = int(np.argmin(I))
    return (E[i_max], I[i_max]), (E[i_min], I[i_min])

def qc_card(status, issues):
    color  = {"PASS":"#E8F5E9","WARN":"#FFF8E1","FAIL":"#FFEBEE"}[status]
    border = {"PASS":"#66BB6A","WARN":"#FFB300","FAIL":"#E53935"}[status]
    items = "".join(f"<li>{it}</li>" for it in issues) if issues else "<li>No issues detected</li>"
    html = (
        "<div style='border:1px solid "+border+"; background:"+color+"; padding:12px; border-radius:10px;'>"
        f"<b>QC Status: {status}</b><br/>"
        f"<ul>{items}</ul>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)

def unit_scale(label_key: str):
    """Unit selector widget; returns scale factor to convert CSV current to Amps."""
    choice = st.selectbox("Current units in CSV", ["A", "mA", "µA"], index=0, key=label_key)
    return {"A": 1.0, "mA": 1e-3, "µA": 1e-6}[choice]


# =========================
# Sidebar: View Mode
# =========================
mode_view = st.sidebar.radio("View Mode", ["Basic", "Advanced"])


# =========================
# BASIC VIEW
# =========================
if mode_view == "Basic":
    st.sidebar.subheader("Basic Controls")
    scan = st.sidebar.slider("Scan rate (V/s)", 0.01, 0.2, float(default["scan_rate"]))
    E_range = st.sidebar.slider("Potential range (V)", 0.2, 1.0, float(default["E_vertex"]))

    col1, col2 = st.columns(2)

    # ---------- Left: Quick simulation ----------
    with col1:
        st.subheader("Simulation")
        if st.button("Run Simulation", use_container_width=True):
            params = default.copy()
            params["scan_rate"] = float(scan)
            params["E_vertex"]  = float(E_range)

            t, E, I = run_sim(params, Nt=500, boundary="reversible")
            ox, red = annotate_peaks(E, I)

            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(E, I * 1e3, linewidth=1.8)
            ax.scatter([ox[0], red[0]], [ox[1]*1e3, red[1]*1e3], s=25)
            ax.text(ox[0], ox[1]*1e3, f"Ox: {ox[0]:.2f} V", fontsize=9)
            ax.text(red[0], red[1]*1e3, f"Red: {red[0]:.2f} V", fontsize=9)
            ax.set_xlabel("Potential / V"); ax.set_ylabel("Current / mA"); ax.grid(True, alpha=0.3)
            st.pyplot(fig, clear_figure=True)

    # ---------- Right: Sample load + Fit ----------
    with col2:
        st.subheader("Sample Data Fit")
        if st.button("Load Sample Data & Fit", use_container_width=True):
            sample_path = project_root / "data" / "sample" / "sample_cv.csv"

            # Units
            scale_basic = unit_scale("units_basic")

            # Load & segment
            df = load_cv_csv(sample_path)
            df_sw = segment_sweeps(df)[0]

            # Convert to Amps for fitting; plot in mA later for readability
            E_exp = df_sw["potential_V"].values
            I_exp = df_sw["current_A"].values * scale_basic

            # Fit (reversible model + auto-region selection + baseline)
            init = {
                "A": default["A"], "L": default["L"], "Nx": default["Nx"], "D": default["D"], "T": default["T"],
                "n": default["n"], "E0": default["E0"], "alpha": default["alpha"], "scan_rate": default["scan_rate"]
            }
            res = fit_parameters(E_exp, I_exp, init)

            # Plot overlay
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(E_exp, I_exp * 1e3, label="Experimental", linewidth=1.2)
            ax.plot(E_exp, res["I_fit"] * 1e3, label="Fit", linewidth=1.8)
            # Shade auto-selected region (if present)
            try:
                ylo = min(np.min(I_exp*1e3), np.min(res["I_fit"]*1e3)) - 0.1
                yhi = max(np.max(I_exp*1e3), np.max(res["I_fit"]*1e3)) + 0.1
                ax.fill_between(
                    E_exp, ylo, yhi,
                    where=res.get("mask", np.zeros_like(E_exp, dtype=bool)),
                    alpha=0.10, step="mid", label="Fit window"
                )
            except Exception:
                pass
            ax.set_xlabel("Potential / V"); ax.set_ylabel("Current / mA")
            ax.grid(True, alpha=0.3); ax.legend()
            st.pyplot(fig, clear_figure=True)

            # QC
            peak = max(1e-9, float(np.max(np.abs(I_exp))))
            rmse_pct = 100.0 * float(res["rmse"]) / peak
            qc_card(*qc_status(res["D"], rmse_pct))

            # Diagnostics
            st.write(
                f"**Fit metrics** | "
                f"D = {res['D']:.2e} m²/s, "
                f"E0 = {res['E0']:.3f} V, "
                f"bias = {res.get('b', 0.0):.2e} A, "
                f"RMSE = {res['rmse']:.2e} A ({rmse_pct:.1f}%)"
            )
            residuals = (res["I_fit"] - I_exp)
            fig2, ax2 = plt.subplots(figsize=(6.2, 2.8))
            ax2.axhline(0, linewidth=1)
            ax2.plot(E_exp, residuals * 1e3, linewidth=1)  # residuals in mA
            ax2.set_xlabel("Potential / V"); ax2.set_ylabel("Residual (mA)")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2, clear_figure=True)
            w = res.get("window_idx")
            if w:
                st.caption(f"Fitted region indices: {w[0]}–{w[1]} (auto-selected)")


# =========================
# ADVANCED VIEW
# =========================
else:
    # =========================
# ADVANCED VIEW
# =========================
else:
    st.sidebar.subheader("Advanced Controls")
    A  = st.sidebar.number_input("Area (m²)", value=float(default["A"]), format="%.6e")
    L  = st.sidebar.number_input("Diffusion Length L (m)", value=float(default["L"]), format="%.6e")
    Nx = st.sidebar.number_input("Grid Points Nx", value=int(default["Nx"]), step=10)
    D  = st.sidebar.number_input("Diffusion D (m²/s)", value=float(default["D"]), format="%.3e")
    E0 = st.sidebar.number_input("Formal Potential E0 (V)", value=float(default["E0"]), format="%.3f")
    T  = st.sidebar.number_input("Temperature (K)", value=float(default["T"]))
    n  = st.sidebar.number_input("Electrons (n)", value=int(default["n"]), step=1)
    alpha = st.sidebar.number_input("Alpha", value=float(default["alpha"]))
    E_start  = st.sidebar.number_input("E start (V)", value=float(default["E_start"]))
    E_vertex = st.sidebar.number_input("E vertex (V)", value=float(default["E_vertex"]))
    E_final  = st.sidebar.number_input("E final (V)", value=float(default["E_final"]))
    scan_rate = st.sidebar.number_input("Scan rate (V/s)", value=float(default["scan_rate"]))
    boundary = st.sidebar.selectbox("Boundary Mode (simulation)", ["reversible", "butler-volmer"])
    k0 = st.sidebar.number_input("k0 (m/s) [BV only]", value=float(default["k0"]), format="%.2e")

    params = dict(
        A=A, L=L, Nx=Nx, D=D, E0=E0, T=T, n=n, alpha=alpha,
        E_start=E_start, E_vertex=E_vertex, E_final=E_final,
        scan_rate=scan_rate, k0=k0
    )

    col1, col2 = st.columns(2)

    # ---------- Left: simulate with chosen boundary ----------
    with col1:
        st.subheader("Simulation")
        if st.button("Run Simulation", key="adv_sim", use_container_width=True):
            t, E, I = run_sim(params, Nt=500, boundary=boundary)
            ox, red = annotate_peaks(E, I)

            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(E, I * 1e3, linewidth=1.8, label=f"Simulated ({boundary})")
            ax.scatter([ox[0], red[0]], [ox[1]*1e3, red[1]*1e3], s=25)
            ax.text(ox[0], ox[1]*1e3, f"Ox: {ox[0]:.2f} V", fontsize=9)
            ax.text(red[0], red[1]*1e3, f"Red: {red[0]:.2f} V", fontsize=9)
            ax.set_xlabel("Potential / V"); ax.set_ylabel("Current / mA")
            ax.grid(True, alpha=0.3); ax.legend()
            st.pyplot(fig, clear_figure=True)

    # ---------- Right: upload + fit ----------
    with col2:
        st.subheader("Upload & Fit")
        file = st.file_uploader("Upload CSV (time_s, potential_V, current_A)", type=["csv"])
        scale_adv = unit_scale("units_adv")

        fit_model = st.selectbox("Fit Model", ["Reversible", "Butler–Volmer"])

        if file is not None and st.button("Fit Data", use_container_width=True):
            df = load_cv_csv(file)
            df_sw = segment_sweeps(df)[0]

            E_exp = df_sw["potential_V"].values
            I_exp = df_sw["current_A"].values * scale_adv

            init = {"A":A, "L":L, "Nx":int(Nx), "D":D, "T":T, "n":int(n), "E0":E0, "alpha":alpha, "scan_rate":scan_rate}

            if fit_model == "Reversible":
                res = fit_parameters(E_exp, I_exp, init)
                fit_label = "Fit (reversible)"
            else:
                res = fit_parameters_bv(E_exp, I_exp, init)
                fit_label = "Fit (BV)"

            # Plot overlay
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(E_exp, I_exp * 1e3, label="Experimental", linewidth=1.2)
            ax.plot(E_exp, res["I_fit"] * 1e3, label=fit_label, linewidth=1.8)

            # Shade auto-fit region
            try:
                ylo = min(np.min(I_exp*1e3), np.min(res["I_fit"]*1e3)) - 0.1
                yhi = max(np.max(I_exp*1e3), np.max(res["I_fit"]*1e3)) + 0.1
                ax.fill_between(E_exp, ylo, yhi,
                    where=res.get("mask", np.zeros_like(E_exp, bool)),
                    alpha=0.10, step="mid", label="Fit region")
            except:
                pass

            ax.set_xlabel("Potential / V"); ax.set_ylabel("Current / mA")
            ax.grid(True, alpha=0.3); ax.legend()
            st.pyplot(fig, clear_figure=True)

            # QC + diagnostics
            peak = max(1e-9, float(np.max(np.abs(I_exp))))
            rmse_pct = 100.0 * float(res["rmse"]) / peak
            qc_card(*qc_status(res["D"], rmse_pct))

            st.write(
                f"**Fit metrics** | "
                f"D = {res['D']:.2e} m²/s, "
                f"E0 = {res['E0']:.3f} V, "
                f"bias = {res.get('b', 0):.2e} A, "
                f"RMSE = {res['rmse']:.2e} A ({rmse_pct:.1f}%)"
            )

            residuals = res["I_fit"] - I_exp
            fig2, ax2 = plt.subplots(figsize=(6.2, 2.8))
            ax2.axhline(0, linewidth=1)
            ax2.plot(E_exp, residuals * 1e3, linewidth=1)
            ax2.set_xlabel("Potential / V"); ax2.set_ylabel("Residual (mA)")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2, clear_figure=True)

            if res.get("window_idx"):
                w = res["window_idx"]
                st.caption(f"Fitted region indices: {w[0]}–{w[1]} (auto-selected)")

  
