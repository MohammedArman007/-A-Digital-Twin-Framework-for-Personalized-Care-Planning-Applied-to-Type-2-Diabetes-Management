import os
import pandas as pd
import numpy as np

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def create_synthetic_cgm(n_patients=5, days=2, freq_min=15, seed=42):
    """Create a small synthetic CGM-like dataset for prototyping.
       Returns a DataFrame with columns: patient_id, timestamp, glucose, meal, insulin, activity
    """
    rng = np.random.default_rng(seed)
    rows = []
    minutes_per_day = 24*60
    steps = int(minutes_per_day / freq_min) * days
    for pid in range(1, n_patients+1):
        base = 90 + rng.normal(0, 5)
        t0 = pd.Timestamp('2025-01-01')
        for s in range(steps):
            ts = t0 + pd.Timedelta(minutes=s*freq_min)
            # simulate circadian + noise + meal spikes
            circ = 10 * np.sin(2*np.pi*(ts.hour*60 + ts.minute)/1440)
            meal = 0
            if ts.hour in (8,13,20) and rng.random() < 0.2:
                meal = int(rng.integers(30,100))
            insulin = 0
            if rng.random() < 0.01:
                insulin = float(rng.uniform(1.0,4.0))
            activity = rng.choice([0,1,2], p=[0.8,0.15,0.05])
            glucose = base + circ + 0.2*meal - 5*activity + rng.normal(0,8)
            rows.append([pid, ts, max(40, glucose), meal, insulin, activity])
    df = pd.DataFrame(rows, columns=['patient_id','timestamp','glucose','meal','insulin','activity'])
    return df

def aggregate_cgm_features(df):
    """Aggregate CGM to per-patient features (mean, std, time-in-range)"""
    def tir(series, low=70, high=180):
        return ((series >= low) & (series <= high)).mean()
    agg = df.groupby('patient_id').glucose.agg(['mean','std','min','max']).reset_index()
    agg['tir'] = df.groupby('patient_id').glucose.apply(lambda s: tir(s)).values
    agg['cv'] = agg['std'] / agg['mean']
    return agg
