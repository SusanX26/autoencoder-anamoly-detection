import onnxruntime as ort
import numpy as np
import time

for name in ['standard', 'denoising', 'sparse']:
    session = ort.InferenceSession(f'models/{name}_ae.onnx')
    input_name = session.get_inputs()[0].name
    lats = []
    # Warmup
    for _ in range(10):
        session.run(None, {input_name: np.random.randn(1, 357).astype(np.float32)})
    # Benchmark
    for _ in range(100):
        x = np.random.randn(1, 357).astype(np.float32)
        t0 = time.perf_counter_ns()
        session.run(None, {input_name: x})
        lats.append(time.perf_counter_ns() - t0)
    print(f"{name.upper()} AE Inference Latency: {np.mean(lats) / 1_000_000:.3f} ms")
