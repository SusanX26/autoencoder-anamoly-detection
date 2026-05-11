import time
import numpy as np
import joblib
import onnxruntime as ort
import os
import json

def benchmark():
    MODEL_DIR = 'models_optimized'
    metadata_path = os.path.join(MODEL_DIR, 'ensemble_metadata.json')
    scaler_path = os.path.join(MODEL_DIR, 'power_scaler.pkl')
    onnx_path = os.path.join(MODEL_DIR, 'extreme_sae.onnx')
    iforest_path = os.path.join(MODEL_DIR, 'iforest_ensemble.pkl')

    if not all([os.path.exists(p) for p in [scaler_path, onnx_path, iforest_path]]):
        print("Error: Models not found. Please train first.")
        return

    # Load components
    scaler = joblib.load(scaler_path)
    session = ort.InferenceSession(onnx_path)
    iforest = joblib.load(iforest_path)
    
    # Dummy data (1 transaction)
    dummy_input = np.random.randn(1, 29)
    
    # Warmup
    for _ in range(10):
        _ = scaler.transform(dummy_input)
        _ = session.run(None, {'input': dummy_input.astype(np.float32)})[0]
        _ = iforest.decision_function(dummy_input)

    # Benchmark Loop
    iterations = 500
    start_time = time.time()
    
    for _ in range(iterations):
        # 1. Scaling
        x_scaled = scaler.transform(dummy_input)
        # 2. AE
        recon = session.run(None, {'input': x_scaled.astype(np.float32)})[0]
        # 3. IF
        _ = iforest.decision_function(x_scaled)
        
    end_time = time.time()
    avg_latency = ((end_time - start_time) / iterations) * 1000 # in ms
    
    print(f"LATENCY_RESULT:{avg_latency:.4f}")

if __name__ == "__main__":
    benchmark()
