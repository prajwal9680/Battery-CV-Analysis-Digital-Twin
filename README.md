# Battery-CV-Analysis-Digital-Twin

**Author:** Prajwal S  

A general-purpose **Cyclic Voltammetry (CV) Digital Twin** for electrochemical systems.  
It simulates CV curves, fits experimental data (extracts diffusion coefficient `D` and formal potential `E0` in a reversible model), and provides a simple QC status card to evaluate data quality.

---

## 🚀 Run in One Click (GitHub Codespaces)

1. Click the green **Code** button  
2. Select **“Create codespace on main”**  
3. Wait for the environment to build (installs automatically)  
4. In the terminal, run:

```bash
python -m streamlit run cv_twin/ui/app.py --server.port 8501 --server.headless true
```

## 📂 Project Structure
```
cv_twin/
  simulator/        # CV model & numerical solver
  fitting/          # Preprocessing + curve fitting
  qc/               # Quality check rules
  ui/               # Streamlit user interface
  data/sample/      # Sample CSV for testing
```
