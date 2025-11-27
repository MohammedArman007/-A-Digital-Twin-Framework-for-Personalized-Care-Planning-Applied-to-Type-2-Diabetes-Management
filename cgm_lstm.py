"""CGM Forecasting LSTM/GRU stub.
This file contains code to prepare sliding windows and an optional TensorFlow model.
TensorFlow is optional — the code will skip training if TF is not installed.
"""
import os
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
from sklearn.preprocessing import StandardScaler

def create_windows(values, window_size=12, horizon=1):
    X, y = [], []
    for i in range(len(values) - window_size - horizon + 1):
        X.append(values[i:i+window_size])
        y.append(values[i+window_size:i+window_size+horizon])
    return np.array(X), np.array(y)

def prepare_patient_series(df, patient_id, value_col='glucose', window_size=12, horizon=1):
    s = df[df.patient_id==patient_id].sort_values('timestamp')
    vals = s[value_col].values.astype(float)
    scaler = StandardScaler()
    vals_scaled = scaler.fit_transform(vals.reshape(-1,1)).flatten()
    X, y = create_windows(vals_scaled, window_size, horizon)
    return X, y, scaler

def build_lstm_model(input_shape, horizon=1):
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow not available. Install tensorflow to train deep models.')
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == '__main__':
    # Simple demo with synthetic data if no real CGM file is provided
    from utils import create_synthetic_cgm
    df = create_synthetic_cgm(n_patients=1, days=2, freq_min=15)
    X, y, scaler = prepare_patient_series(df, patient_id=1, window_size=16, horizon=1)
    print('Prepared windows:', X.shape, y.shape)
    if TF_AVAILABLE:
        model = build_lstm_model(input_shape=X.shape[1:], horizon=1)
        model.summary()
        model.fit(X, y, epochs=5, batch_size=32)
        model.save('models/cgm_lstm.h5')
        print('Saved model to models/cgm_lstm.h5')
    else:
        print('TensorFlow not available. Install TensorFlow to train LSTM models.')
