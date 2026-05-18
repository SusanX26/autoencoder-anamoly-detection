import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import onnxruntime as ort
import os

def benchmark_fast():
    onnx_path = os.path.join('models', 'sparse_ae.onnx')
    if not os.path.exists(onnx_path):
        print("Model not found. Run training/export first.")
        return

    # Load ONNX with Max Optimizations
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, opts)
    
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
