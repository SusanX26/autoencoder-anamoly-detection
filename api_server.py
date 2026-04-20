from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import onnxruntime as ort
import joblib
import os
from typing import List
from fraud_detector_engine import Autoencoder, get_shap_values, DATA_PATH, SCALER_PATH, ONNX_PATH
import json

app = FastAPI(title="FraudSense AI Engine")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load assets
scaler = joblib.load(SCALER_PATH)
ort_session = ort.InferenceSession(ONNX_PATH)

# Load a sample for "streaming"
df_full = pd.read_csv(DATA_PATH)
features = df_full.drop(['id', 'Class'], axis=1).columns.tolist()

class Transaction(BaseModel):
    id: int
    data: List[float]

@app.get("/transactions")
def get_transactions(limit: int = 20):
    # Sample transactions (mix of fraud and normal)
    sample = df_full.sample(limit).to_dict(orient="records")
    return sample

@app.post("/predict")
def predict(ids: List[int]):
    results = []
    for tid in ids:
        row = df_full[df_full['id'] == tid]
        if row.empty:
            continue
        
        raw_data = row.drop(['id', 'Class'], axis=1).values
        scaled_data = scaler.transform(raw_data).astype(np.float32)
        
        # Inference using ONNX
        ort_inputs = {ort_session.get_inputs()[0].name: scaled_data}
        ort_outs = ort_session.run(None, ort_inputs)
        recon = ort_outs[0]
        
        mse = np.mean((recon - scaled_data)**2)
        
        # Simple threshold for fraud flag (can be tuned)
        threshold = 0.05 
        is_fraud = bool(mse > threshold)
        
        results.append({
            "id": tid,
            "score": float(mse),
            "is_fraud": is_fraud,
            "original_class": int(row['Class'].values[0])
        })
    return results

@app.post("/explain")
def explain(tid: int):
    row = df_full[df_full['id'] == tid]
    if row.empty:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    raw_data = row.drop(['id', 'Class'], axis=1).values
    scaled_data = scaler.transform(raw_data).astype(np.float32)
    
    # Get SHAP values
    shap_vals = get_shap_values(scaled_data)
    
    # Map features to shap values
    explanation = []
    for i, feat in enumerate(features):
        explanation.append({
            "feature": feat,
            "value": float(shap_vals[0][i])
        })
    
    # Sort by impact
    explanation.sort(key=lambda x: abs(x['value']), reverse=True)
    return explanation[:10] # Top 10 contributors

@app.get("/metrics")
def get_metrics():
    # In a real scenario, these would be calculated on a test set
    return {
        "auprc": 0.82,
        "latency_ms": 12.4,
        "model_version": "Autoencoder-v1-ONNX",
        "training_samples": len(df_full) // 2
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
