"""Emergency Department (ED) Risk Prediction (ensemble) stub.
Uses features combined from static + aggregated CGM features.
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from utils import aggregate_cgm_features

def prepare_joined_dataset(static_df, cgm_df):
    # static_df must have patient_id
    agg = aggregate_cgm_features(cgm_df)
    df = static_df.merge(agg, on='patient_id', how='left').fillna(0)
    # create synthetic label for ED risk (for prototyping)
    df['ed_risk'] = ((df['mean']>160) | (df['cv']>0.4) | (df['Glucose']>180)).astype(int)
    return df

def train_ed_risk(df):
    features = [c for c in df.columns if c not in ('patient_id','ed_risk')]
    X = df[features].fillna(0)
    y = df['ed_risk']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:,1]
    print('ROC-AUC:', roc_auc_score(y_test, preds))
    print(classification_report(y_test, (preds>0.5).astype(int)))
    return clf

if __name__ == '__main__':
    # simple demo using synthetic data if missing
    static_df = pd.DataFrame({
        'patient_id':[1,2,3],
        'Glucose':[120,200,95],
        'BMI':[28,35,24],
        'Age':[45,60,30]
    })
    from utils import create_synthetic_cgm
    cgm_df = create_synthetic_cgm(n_patients=3, days=2)
    df = prepare_joined_dataset(static_df, cgm_df)
    clf = train_ed_risk(df)
