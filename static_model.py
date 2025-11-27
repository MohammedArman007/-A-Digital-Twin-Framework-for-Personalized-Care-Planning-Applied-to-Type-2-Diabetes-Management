"""Static Risk Prediction Model (sklearn)
Usage:
    python static_model.py --data data/pima.csv --out models/static_model.pkl
If dataset not found, the script will generate a small synthetic dataset.
"""
import argparse
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from utils import load_csv

def prepare_pima(path=None):
    if path and os.path.exists(path):
        df = load_csv(path)
    else:
        # synthetic small dataset
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            'Pregnancies': rng.integers(0,6,size=n),
            'Glucose': rng.normal(120,30,size=n).clip(40,300),
            'BloodPressure': rng.normal(70,12,size=n),
            'SkinThickness': rng.normal(20,10,size=n).clip(0,99),
            'Insulin': rng.normal(80,40,size=n).clip(0,900),
            'BMI': rng.normal(30,6,size=n).clip(10,60),
            'DiabetesPedigreeFunction': rng.random(n),
            'Age': rng.integers(21,80,size=n)
        })
        # simple synthetic rule for outcome
        df['Outcome'] = ((df.Glucose>140) | (df.BMI>32)).astype(int)
    return df

def train_static(df, out_path=None):
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    print('Accuracy:', accuracy_score(y_test, preds))
    print('ROC-AUC:', roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        joblib.dump(clf, out_path)
        print('Model saved to', out_path)
    return clf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/pima.csv')
    parser.add_argument('--out', type=str, default='models/static_model.pkl')
    args = parser.parse_args()
    df = prepare_pima(args.data)
    train_static(df, args.out)
