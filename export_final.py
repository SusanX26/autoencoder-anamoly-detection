import torch
import torch.nn as nn
import os
import sys

# Set encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

class ExtremeAutoencoder(nn.Module):
    def __init__(self, input_dim, is_sparse=False):
        super(ExtremeAutoencoder, self).__init__()
        self.is_sparse = is_sparse
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        if self.is_sparse:
            return reconstructed, latent
        return reconstructed

def export_all():
    MODEL_DIR = 'models_optimized'
    input_dim = 29
    dummy_input = torch.randn(1, input_dim)
    
    class W(nn.Module):
        def __init__(self, m, s=False): super().__init__(); self.m = m; self.s = s
        def forward(self, x):
            res = self.m(x)
            return res[0] if self.s else res

    configs = [
        ('standard_ae.pth', 'standard_ae.onnx', False),
        ('sparse_ae.pth', 'sparse_ae.onnx', True),
        ('denoising_ae.pth', 'denoising_ae.onnx', False)
    ]
    
    for pth, onnx_name, is_spr in configs:
        pth_path = os.path.join(MODEL_DIR, pth)
        onnx_path = os.path.join(MODEL_DIR, onnx_name)
        
        if not os.path.exists(pth_path):
            print(f"Skipping {pth_path} (not found)")
            continue
            
        print(f"Exporting {pth} to {onnx_name}...")
        model = ExtremeAutoencoder(input_dim, is_sparse=is_spr)
        model.load_state_dict(torch.load(pth_path, map_location='cpu'))
        model.eval()
        
        torch.onnx.export(W(model, s=is_spr), dummy_input, onnx_path,
                          input_names=['input'], output_names=['output'],
                          opset_version=12)
        print(f"Success: {onnx_name}")

if __name__ == "__main__":
    export_all()
