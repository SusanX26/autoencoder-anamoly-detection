import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib
import os

# Base path for models
MODEL_DIR = os.path.join(os.getcwd(), 'models')

# Load Scaler
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

# Load ONNX sessions
std_session = ort.InferenceSession(os.path.join(MODEL_DIR, 'standard_ae.onnx'))
sparse_session = ort.InferenceSession(os.path.join(MODEL_DIR, 'sparse_ae.onnx'))

def get_onnx_mse(data, model_type='standard'):
    session = std_session if model_type == 'standard' else sparse_session
    
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
    # For cloud demo, we provide mock SHAP values to save size/memory
    # This keeps the UI working without the 500MB SHAP dependency
    features = [f'V{i}' for i in range(1, 29)] + ['Amount']
    results = []
    for feat in features:
        results.append({
            "feature": feat,
            "value": np.random.uniform(0, 0.5)
        })
    return sorted(results, key=lambda x: x['value'], reverse=True)[:10]
