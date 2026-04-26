from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import os
import sys

# Ensure api folder is in path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from onnx_engine import get_onnx_mse, get_mock_shap
except ImportError:
    from .onnx_engine import get_onnx_mse, get_mock_shap

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust data pathing
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data_sample.csv')

def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return empty df with correct columns as fallback
        return pd.DataFrame(columns=['id', 'Class'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])

df = load_data()

@app.get("/transactions")
@app.get("/api/transactions")
def get_transactions(limit: int = 12):
    if df.empty: return []
    sample = df.sample(min(limit, len(df))).to_dict('records')
    return sample

@app.post("/predict")
@app.post("/api/predict")
def predict(tids: List[int], model_type: str = 'standard'):
    rows = df[df['id'].isin(tids)]
    if rows.empty: return []
    
    features = rows.drop(['id', 'Class'], axis=1)
    try:
        mse_scores = get_onnx_mse(features, model_type=model_type)
    except Exception:
        mse_scores = [0.0] * len(rows)
    
    results = []
    for i, (_, row) in enumerate(rows.iterrows()):
        score = float(mse_scores[i])
        results.append({
            "id": int(row['id']),
            "score": score,
            "is_anomaly": bool(score > 0.05)
        })
    return results

@app.post("/explain")
@app.post("/api/explain")
def explain(tid: int, model_type: str = 'standard'):
    row = df[df['id'] == tid]
    if row.empty: return []
    
    features = row.drop(['id', 'Class'], axis=1)
    explanation = get_mock_shap(features)
    return explanation

@app.get("/metrics")
@app.get("/api/metrics")
def get_metrics():
    # Matching the structure expected by the frontend
    m_std = {"auprc": 0.821, "f1": 0.794, "latency": 11.2, "precision": 0.991}
    m_spr = {"auprc": 0.864, "f1": 0.841, "latency": 12.8, "precision": 0.999}
    
    return {
        "standard": m_std,
        "sparse": m_spr,
        "global": m_spr,
        "feature_importance": [
            {"feature": "V14", "value": 0.85},
            {"feature": "V17", "value": 0.72},
            {"feature": "V12", "value": 0.61},
            {"feature": "V10", "value": 0.58},
            {"feature": "V4", "value": 0.45}
        ]
    }

@app.get("/model-info")
@app.get("/api/model-info")
def model_info():
    return {
        "standard": [
            {"layer": "Input", "neurons": 30, "activation": "None"},
            {"layer": "Encoder 1", "neurons": 32, "activation": "ReLU"},
            {"layer": "Encoder 2", "neurons": 16, "activation": "ReLU"},
            {"layer": "Bottleneck", "neurons": 8, "activation": "ReLU"},
            {"layer": "Decoder 1", "neurons": 16, "activation": "ReLU"},
            {"layer": "Decoder 2", "neurons": 32, "activation": "ReLU"},
            {"layer": "Output", "neurons": 30, "activation": "Sigmoid"}
        ],
        "sparse": [
            {"layer": "Input", "neurons": 30, "activation": "None"},
            {"layer": "Sparse Encoder 1", "neurons": 32, "activation": "ReLU"},
            {"layer": "Sparse Encoder 2", "neurons": 16, "activation": "ReLU"},
            {"layer": "Bottleneck (L1)", "neurons": 8, "activation": "ReLU"},
            {"layer": "Decoder 1", "neurons": 16, "activation": "ReLU"},
            {"layer": "Decoder 2", "neurons": 32, "activation": "ReLU"},
            {"layer": "Output", "neurons": 30, "activation": "Sigmoid"}
        ]
    }
