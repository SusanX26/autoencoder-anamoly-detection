import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib
import os

# Robust pathing for Vercel
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Load Scaler
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
scaler = joblib.load(scaler_path)

# Load ONNX sessions with error handling
try:
    std_path = os.path.join(MODEL_DIR, 'standard_ae.onnx')
    sparse_path = os.path.join(MODEL_DIR, 'sparse_ae.onnx')
    std_session = ort.InferenceSession(std_path)
    sparse_session = ort.InferenceSession(sparse_path)
except Exception as e:
    print(f"Error loading ONNX sessions: {e}")
    std_session = None
    sparse_session = None

def get_onnx_mse(data, model_type='standard'):
    session = std_session if model_type == 'standard' else sparse_session
    if session is None:
        return np.zeros(len(data))
    
    # Scale data
    data_scaled = scaler.transform(data).astype(np.float32)
    
    # Run inference
    inputs = {session.get_inputs()[0].name: data_scaled}
    outputs = session.run(None, inputs)
    reconstructed = outputs[0]
    
    # Calculate MSE
    mse = np.mean(np.power(data_scaled - reconstructed, 2), axis=1)
    return mse

def get_mock_shap(data):
    features = [f'V{i}' for i in range(1, 29)] + ['Amount']
    results = []
    for feat in features:
        results.append({
            "feature": feat,
            "value": np.random.uniform(0, 0.5)
        })
    return sorted(results, key=lambda x: x['value'], reverse=True)[:10]
