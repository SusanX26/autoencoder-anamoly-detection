import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import onnxruntime as ort
import os

def benchmark_fast():
    MODEL_DIR = 'models_optimized'
    onnx_path = os.path.join(MODEL_DIR, 'extreme_sae.onnx')

    # Load ONNX
    session = ort.InferenceSession(onnx_path)
    
    # Fast Scaler (StandardScaler)
    scaler = StandardScaler()
    scaler.fit(np.random.randn(100, 29)) # dummy fit
    
    # Dummy data
    dummy_input = np.random.randn(1, 29)
    
    # Warmup
    for _ in range(50):
        x_scaled = scaler.transform(dummy_input)
        _ = session.run(None, {'input': x_scaled.astype(np.float32)})[0]

    # Benchmark Loop
    iterations = 2000
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        x_scaled = scaler.transform(dummy_input)
        _ = session.run(None, {'input': x_scaled.astype(np.float32)})[0]
        
    end_time = time.perf_counter()
    avg_latency = ((end_time - start_time) / iterations) * 1000 # in ms
    
    print(f"ULTRA_FAST_LATENCY:{avg_latency:.4f} ms")

if __name__ == "__main__":
    benchmark_fast()
