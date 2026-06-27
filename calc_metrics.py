import pandas as pd
import numpy as np
import onnxruntime as ort
import joblib
import os
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc

DATA_DIR = 'ieee-fraud-detection'
MODEL_DIR = 'models'

print("Loading test data...")
df = pd.read_csv(os.path.join(DATA_DIR, 'eval_sample.csv'))
y = df['isFraud'].values
feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))
X = df[feature_cols].values.astype(np.float32)

mean_val, std_val = joblib.load(os.path.join(MODEL_DIR, 'scaler_params.pkl'))
X_scaled = (X - mean_val) / (std_val + 1e-8)

print("\nEvaluating ONNX models...")
for name in ['standard', 'denoising', 'sparse']:
    try:
        session = ort.InferenceSession(os.path.join(MODEL_DIR, f'{name}_ae.onnx'))
        input_name = session.get_inputs()[0].name
        probs = session.run(None, {input_name: X})[0].flatten()
        
        auroc = roc_auc_score(y, probs)
        p, r, t = precision_recall_curve(y, probs)
        auprc = auc(r, p)
        f1 = 2 * (p * r) / (p + r + 1e-10)
        best_f1 = np.max(f1)
        
        print(f"--- Model: {name.upper()} ---")
        print(f"ROC-AUC: {auroc:.4f}")
        print(f"AUPRC  : {auprc:.4f}")
        print(f"Best F1: {best_f1:.4f}\n")
    except Exception as e:
        print(f"Failed to evaluate {name}: {e}")
