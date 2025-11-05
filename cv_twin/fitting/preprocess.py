import pandas as pd, numpy as np

def load_cv_csv(path_or_file):
    """Read a CV CSV and standardize column names to time_s, potential_V, current_A."""
    df = pd.read_csv(path_or_file)
    cols = {c.lower(): c for c in df.columns}
    t = next((v for k, v in cols.items() if 'time' in k), None)
    e = next((v for k, v in cols.items() if 'potential' in k or k in ('e_v', 'e')), None)
    i = next((v for k, v in cols.items() if 'current' in k or k in ('i_a', 'i')), None)
    if not (t and e and i):
        raise ValueError("CSV needs time/potential/current columns")
    df = df.rename(columns={t: 'time_s', e: 'potential_V', i: 'current_A'})
    return df[['time_s', 'potential_V', 'current_A']].dropna()

def segment_sweeps(df):
    """Split a CV into monotonic segments (forward / reverse)."""
    E = df['potential_V'].values
    dE = np.diff(E, prepend=E[0])
    s = np.sign(dE)
    idx = [0]
    for k in range(1, len(s)):
        if s[k] != s[k-1]:
            idx.append(k)
    idx.append(len(E))
    sweeps = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 20:
            sweeps.append(df.iloc[a:b].reset_index(drop=True))
    return sweeps or [df]

