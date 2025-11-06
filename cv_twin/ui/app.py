import sys
from pathlib import Path
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

st.set_page_config(page_title='CV Digital Twin', layout='wide')
st.title('CV Digital Twin – General Electrochemistry')

mode_view = st.sidebar.radio('View Mode', ['Basic', 'Advanced'])

default = dict(A=1.0e-4, L=1.5e-4, Nx=90, D=1.0e-10, E0=0.10, T=298.15, n=1, alpha=0.5,
               E_start=0.00, E_vertex=0.40, E_final=0.00, scan_rate=0.05, k0=5e-6)

def run_sim(params, Nt=500, boundary='reversible'):
    p = ModelParams(A=params['A'], L=params['L'], Nx=int(params['Nx']), D=params['D'],
                    T=params['T'], n=int(params['n']), E0=params['E0'], alpha=params['alpha'],
                    E_start=params['E_start'], E_vertex=params['E_vertex'], E_final=params['E_final'],
                    scan_rate=params['scan_rate'], mode=boundary, k0=params['k0'])
    return simulate(p, Nt=Nt)

def annotate_peaks(E, I):
    i_max = int(np.argmax(I)); i_min = int(np.argmin(I))
    return (E[i_max], I[i_max]), (E[i_min], I[i_min])

if mode_view == 'Basic':
    st.sidebar.subheader('Basic Controls')
    scan = st.sidebar.slider('Scan rate (V/s)', 0.01, 0.2, float(default['scan_rate']))
    E_range = st.sidebar.slider('Potential range (V)', 0.2, 1.0, 0.4)

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Run Simulation'):
            params = default.copy()
            params['scan_rate'] = scan
            params['E_vertex']  = E_range
            t, E, I = run_sim(params, Nt=500, boundary='reversible')
            ox, red = annotate_peaks(E, I)
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(E, I*1e3, linewidth=1.6)
            ax.scatter([ox[0], red[0]], [ox[1]*1e3, red[1]*1e3], s=25)
            ax.text(ox[0], ox[1]*1e3, f'Ox: {ox[0]:.2f} V', fontsize=9)
            ax.text(red[0], red[1]*1e3, f'Red: {red[0]:.2f} V', fontsize=9)
            ax.set_xlabel('Potential / V'); ax.set_ylabel('Current / mA'); ax.grid(True, alpha=0.3)
            st.pyplot(fig, clear_figure=True)

    with col2:
        if st.button('Load Sample Data & Fit'):
            sample_path = project_root / 'data' / 'sample' / 'sample_cv.csv'
            df = load_cv_csv(sample_path)
            df_sw = segment_sweeps(df)[0]
            E_exp = df_sw['potential_V'].values
            I_exp = df_sw['current_A'].values
            init = {'A':default['A'],'L':default['L'],'Nx':default['Nx'],'D':default['D'],'T':default['T'],
                    'n':default['n'],'E0':default['E0'],'alpha':default['alpha'],'scan_rate':default['scan_rate']}
            res = fit_parameters(E_exp, I_exp, init)
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(E_exp, I_exp*1e3, label='Experimental', linewidth=1.2)
            ax.plot(E_exp, res['I_fit']*1e3, label='Fit', linewidth=1.6)
            ax.set_xlabel('Potential / V'); ax.set_ylabel('Current / mA'); ax.grid(True, alpha=0.3); ax.legend()
            st.pyplot(fig, clear_figure=True)
            peak = max(1e-9, float(np.max(np.abs(I_exp))))
            rmse_pct = 100.0 * float(res['rmse']) / peak
            status, issues = qc_status(res['D'], rmse_pct)
            color  = {'PASS':'#E8F5E9','WARN':'#FFF8E1','FAIL':'#FFEBEE'}[status]
            border = {'PASS':'#66BB6A','WARN':'#FFB300','FAIL':'#E53935'}[status]
            issues_html = ''.join('<li>'+it+'</li>' for it in issues) if issues else '<li>No issues detected</li>'
            html = ("<div style='border:1px solid "+border+"; background:"+color+"; padding:12px; border-radius:8px;'>"
                    "<b>QC Status: "+status+"</b><br/>"
                    "<ul>"+issues_html+"</ul>"
                    "</div>")
            st.markdown(html, unsafe_allow_html=True)
                        # ==== Diagnostics: Show fit metrics + residual plot ====
            st.write(f"**Fit metrics**  |  D = {res['D']:.2e} m²/s,  E0 = {res['E0']:.3f} V,  RMSE = {res['rmse']:.2e} A ({rmse_pct:.1f}%)")

            # Residuals vs Potential Plot
            residuals = (res['I_fit'] - I_exp)
            fig2, ax2 = plt.subplots(figsize=(6.2, 2.8))
            ax2.axhline(0, linewidth=1)
            ax2.plot(E_exp, residuals * 1e3, linewidth=1)  # Convert to mA for readability
            ax2.set_xlabel('Potential / V')
            ax2.set_ylabel('Residual (mA)')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2, clear_figure=True)


else:
    st.sidebar.subheader('Advanced Controls')
    A = st.sidebar.number_input('Area (m²)', value=float(default['A']), format='%.6e')
    L = st.sidebar.number_input('Diffusion Length L (m)', value=float(default['L']), format='%.6e')
    Nx = st.sidebar.number_input('Grid Points Nx', value=int(default['Nx']), step=10)
    D = st.sidebar.number_input('Diffusion D (m²/s)', value=float(default['D']), format='%.3e')
    E0 = st.sidebar.number_input('Formal Potential E0 (V)', value=float(default['E0']), format='%.3f')
    T  = st.sidebar.number_input('Temperature (K)', value=float(default['T']))
    n  = st.sidebar.number_input('Electrons (n)', value=int(default['n']), step=1)
    alpha = st.sidebar.number_input('Alpha', value=float(default['alpha']))
    E_start = st.sidebar.number_input('E start (V)', value=float(default['E_start']))
    E_vertex = st.sidebar.number_input('E vertex (V)', value=float(default['E_vertex']))
    E_final  = st.sidebar.number_input('E final (V)', value=float(default['E_final']))
    scan_rate = st.sidebar.number_input('Scan rate (V/s)', value=float(default['scan_rate']))
    boundary = st.sidebar.selectbox('Boundary Mode', ['reversible','butler-volmer'])
    k0 = st.sidebar.number_input('k0 (m/s) [BV only]', value=float(default['k0']), format='%.2e')

    params = dict(A=A, L=L, Nx=Nx, D=D, E0=E0, T=T, n=n, alpha=alpha,
                  E_start=E_start, E_vertex=E_vertex, E_final=E_final,
                  scan_rate=scan_rate, k0=k0)

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Run Simulation'):
            t, E, I = run_sim(params, Nt=500, boundary=boundary)
            ox, red = annotate_peaks(E, I)
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(E, I*1e3, linewidth=1.6, label='Simulated')
            ax.scatter([ox[0], red[0]], [ox[1]*1e3, red[1]*1e3], s=25)
            ax.text(ox[0], ox[1]*1e3, f'Ox: {ox[0]:.2f} V', fontsize=9)
            ax.text(red[0], red[1]*1e3, f'Red: {red[0]:.2f} V', fontsize=9)
            ax.set_xlabel('Potential / V'); ax.set_ylabel('Current / mA'); ax.grid(True, alpha=0.3); ax.legend()
            st.pyplot(fig, clear_figure=True)

    with col2:
        file = st.file_uploader('Upload CSV (time_s, potential_V, current_A)', type=['csv'])
        if file is not None and st.button('Fit Parameters (D & E0, reversible)'):
            units_basic = st.selectbox('Current units in CSV', ['A','mA','µA'], index=0, key='units_basic')
            scale_basic = {'A':1.0, 'mA':1e-3, 'µA':1e-6}[units_basic]

            df = load_cv_csv(file); df_sw = segment_sweeps(df)[0]
            E_exp, I_exp = df_sw['potential_V'].values, df_sw['current_A'].values
            init = {'A':A,'L':L,'Nx':int(Nx),'D':D,'T':T,'n':int(n),'E0':E0,'alpha':alpha,'scan_rate':scan_rate}
            res = fit_parameters(E_exp, I_exp, init)
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(E_exp, I_exp*1e3, label='Experimental', linewidth=1.2)
            ax.plot(E_exp, res['I_fit']*1e3, label='Fit', linewidth=1.6)
            ax.set_xlabel('Potential / V'); ax.set_ylabel('Current / mA'); ax.grid(True, alpha=0.3); ax.legend()
            st.pyplot(fig, clear_figure=True)
            peak = max(1e-9, float(np.max(np.abs(I_exp))))
            rmse_pct = 100.0 * float(res['rmse']) / peak
            status, issues = qc_status(res['D'], rmse_pct)
            color  = {'PASS':'#E8F5E9','WARN':'#FFF8E1','FAIL':'#FFEBEE'}[status]
            border = {'PASS':'#66BB6A','WARN':'#FFB300','FAIL':'#E53935'}[status]
            issues_html = ''.join('<li>'+it+'</li>' for it in issues) if issues else '<li>No issues detected</li>'
            html = ("<div style='border:1px solid "+border+"; background:"+color+"; padding:12px; border-radius:8px;'>"
                    "<b>QC Status: "+status+"</b><br/>"
                    "<ul>"+issues_html+"</ul>"
                    "</div>")
            st.markdown(html, unsafe_allow_html=True)
