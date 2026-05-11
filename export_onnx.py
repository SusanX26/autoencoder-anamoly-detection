import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch, torch.nn as nn
import onnx, numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
from fraud_detector_engine import (
    StandardAutoencoder, SparseAutoencoder, DenoisingAutoencoder,
    STANDARD_MODEL_PATH, SPARSE_MODEL_PATH, DENOISING_MODEL_PATH,
    STANDARD_ONNX_PATH, SPARSE_ONNX_PATH, DENOISING_ONNX_PATH,
    DATA_PATH
)
import pandas as pd

df = pd.read_csv(DATA_PATH, nrows=1).fillna(0)
X = df.drop(['Class'], axis=1)
if 'id' in X.columns: X = X.drop(['id'], axis=1)
if 'Time' in X.columns: X = X.drop(['Time'], axis=1)
input_dim = X.shape[1]

# Use a batch of 2 so PyTorch detects the batch dim as variable
dummy = torch.randn(2, input_dim)

def export_dynamic(model, path, is_sparse=False):
    model.eval()
    if is_sparse:
        class W(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): r, _ = self.m(x); return r
        model = W(model)
        model.eval()

    with torch.no_grad():
        torch.onnx.export(
            model, dummy, path,
            input_names=['input'], output_names=['output'],
            opset_version=17,
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            dynamo=False   # Use legacy exporter which fully supports dynamic_axes
        )
    print(f"Exported: {path}")

std = StandardAutoencoder(input_dim)
std.load_state_dict(torch.load(STANDARD_MODEL_PATH, weights_only=True))
export_dynamic(std, STANDARD_ONNX_PATH)

spr = SparseAutoencoder(input_dim)
spr.load_state_dict(torch.load(SPARSE_MODEL_PATH, weights_only=True))
export_dynamic(spr, SPARSE_ONNX_PATH, is_sparse=True)

den = DenoisingAutoencoder(input_dim)
den.load_state_dict(torch.load(DENOISING_MODEL_PATH, weights_only=True))
export_dynamic(den, DENOISING_ONNX_PATH)

print("All 3 ONNX models exported with dynamic batch support!")
