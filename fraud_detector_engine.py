import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap
import onnx
import onnxruntime
import os
import json
import joblib

# --- CONFIG ---
DATA_PATH = 'creditcard.csv'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'temp_data.csv'

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

STANDARD_MODEL_PATH = os.path.join(MODEL_DIR, 'standard_ae.pth')
SPARSE_MODEL_PATH = os.path.join(MODEL_DIR, 'sparse_ae.pth')
DENOISING_MODEL_PATH = os.path.join(MODEL_DIR, 'denoising_ae.pth')

STANDARD_ONNX_PATH = os.path.join(MODEL_DIR, 'standard_ae.onnx')
SPARSE_ONNX_PATH = os.path.join(MODEL_DIR, 'sparse_ae.onnx')
DENOISING_ONNX_PATH = os.path.join(MODEL_DIR, 'denoising_ae.onnx')

SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# --- MODELS ---

class StandardAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(StandardAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- TRAINING ---

def train_models():
    print(f"Loading original imbalanced dataset: {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH).fillna(0)
    
    # Critical for Anomaly Detection: Train ONLY on Normal transactions (Class 0)
    X_normal = df[df['Class'] == 0].drop(['Class'], axis=1)
    if 'id' in X_normal.columns:
        X_normal = X_normal.drop(['id'], axis=1)
    if 'Time' in X_normal.columns:
        X_normal = X_normal.drop(['Time'], axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)
    joblib.dump(scaler, SCALER_PATH)
    
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
    X_train_tensor = torch.FloatTensor(X_train)
    input_dim = X_train.shape[1]
    
    criterion = nn.MSELoss()
    
    # 1. Train Standard Autoencoder
    print("\n[1/3] Training Standard Autoencoder...")
    std_model = StandardAutoencoder(input_dim)
    std_optimizer = optim.Adam(std_model.parameters(), lr=0.001)
    for epoch in range(10):
        std_model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])
        total_loss = 0
        for i in range(0, X_train_tensor.size()[0], 1024):
            indices = permutation[i:i+1024]
            batch_x = X_train_tensor[indices]
            std_optimizer.zero_grad()
            output = std_model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            std_optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/(len(X_train_tensor)/1024):.6f}")
    torch.save(std_model.state_dict(), STANDARD_MODEL_PATH)

    # 2. Train Sparse Autoencoder
    print("\n[2/3] Training Sparse Autoencoder (L1 Regularization)...")
    spr_model = SparseAutoencoder(input_dim)
    spr_optimizer = optim.Adam(spr_model.parameters(), lr=0.001)
    l1_lambda = 1e-4 
    for epoch in range(10):
        spr_model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])
        total_loss = 0
        for i in range(0, X_train_tensor.size()[0], 1024):
            indices = permutation[i:i+1024]
            batch_x = X_train_tensor[indices]
            spr_optimizer.zero_grad()
            reconstructed, latent = spr_model(batch_x)
            mse_loss = criterion(reconstructed, batch_x)
            l1_loss = torch.mean(torch.abs(latent))
            loss = mse_loss + l1_lambda * l1_loss
            loss.backward()
            spr_optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/(len(X_train_tensor)/1024):.6f}")
    torch.save(spr_model.state_dict(), SPARSE_MODEL_PATH)

    # 3. Train Denoising Autoencoder
    print("\n[3/3] Training Denoising Autoencoder (Noise Resilience)...")
    den_model = DenoisingAutoencoder(input_dim)
    den_optimizer = optim.Adam(den_model.parameters(), lr=0.001)
    noise_factor = 0.2
    for epoch in range(10):
        den_model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])
        total_loss = 0
        for i in range(0, X_train_tensor.size()[0], 1024):
            indices = permutation[i:i+1024]
            batch_x = X_train_tensor[indices]
            
            # Add noise to input
            noisy_batch_x = batch_x + noise_factor * torch.randn_like(batch_x)
            
            den_optimizer.zero_grad()
            output = den_model(noisy_batch_x)
            loss = criterion(output, batch_x) # Reconstruct clean from noisy
            loss.backward()
            den_optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/(len(X_train_tensor)/1024):.6f}")
    torch.save(den_model.state_dict(), DENOISING_MODEL_PATH)

# --- EXPORT ---

def export_to_onnx():
    print("\nExporting models to ONNX for Enterprise Deployment...")
    df = pd.read_csv(DATA_PATH, nrows=1).fillna(0)
    X = df.drop(['Class'], axis=1)
    if 'id' in X.columns: X = X.drop(['id'], axis=1)
    if 'Time' in X.columns: X = X.drop(['Time'], axis=1)
    input_dim = X.shape[1]
    dummy_input = torch.randn(1, input_dim)

    # Standard
    std_model = StandardAutoencoder(input_dim)
    std_model.load_state_dict(torch.load(STANDARD_MODEL_PATH))
    std_model.eval()
    torch.onnx.export(std_model, dummy_input, STANDARD_ONNX_PATH, input_names=['input'], output_names=['output'])

    # Sparse
    spr_model = SparseAutoencoder(input_dim)
    spr_model.load_state_dict(torch.load(SPARSE_MODEL_PATH))
    spr_model.eval()
    class SparseExportWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): r, _ = self.m(x); return r
    torch.onnx.export(SparseExportWrapper(spr_model), dummy_input, SPARSE_ONNX_PATH, input_names=['input'], output_names=['output'])

    # Denoising
    den_model = DenoisingAutoencoder(input_dim)
    den_model.load_state_dict(torch.load(DENOISING_MODEL_PATH))
    den_model.eval()
    torch.onnx.export(den_model, dummy_input, DENOISING_ONNX_PATH, input_names=['input'], output_names=['output'])
    
    print("All ONNX Models Ready.")

def optimize_for_production(model):
    # Dynamic Quantization: Reduces model size by 4x and speeds up CPU inference
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

def get_prediction(data, model_type='standard'):
    # Use ONNX Runtime for Enterprise-grade latency
    onnx_path = os.path.join(MODEL_DIR, f'{model_type}_ae.onnx')
    if os.path.exists(onnx_path):
        session = onnxruntime.InferenceSession(onnx_path)
        inputs = {session.get_inputs()[0].name: data.astype(np.float32)}
        reconstructed = session.run(None, inputs)[0]
        mse = np.mean((reconstructed - data)**2, axis=1)
        return mse
    return np.zeros(len(data))

def get_shap_values(sample_data, model_type='standard'):
    df = pd.read_csv(DATA_PATH, nrows=200).fillna(0)
    X = df.drop(['Class'], axis=1)
    if 'id' in X.columns: X = X.drop(['id'], axis=1)
    if 'Time' in X.columns: X = X.drop(['Time'], axis=1)
    
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)
    input_dim = X.shape[1]
    
    # Standard models are required for SHAP (Quantized models don't support gradients)
    if model_type == 'sparse':
        model = SparseAutoencoder(input_dim)
        model.load_state_dict(torch.load(SPARSE_MODEL_PATH, map_location='cpu'))
    elif model_type == 'denoising':
        model = DenoisingAutoencoder(input_dim)
        model.load_state_dict(torch.load(DENOISING_MODEL_PATH, map_location='cpu'))
    else:
        model = StandardAutoencoder(input_dim)
        model.load_state_dict(torch.load(STANDARD_MODEL_PATH, map_location='cpu'))

    model.eval()
    
    # SHAP GradientExplainer is more stable for MSE-based explanations
    class ShapWrapper(nn.Module):
        def __init__(self, m, m_type):
            super().__init__()
            self.m = m
            self.m_type = m_type
        def forward(self, x):
            if self.m_type == 'sparse':
                r, _ = self.m(x)
            else:
                r = self.m(x)
            # Must return (batch, 1) for SHAP
            return torch.mean((r - x)**2, dim=1, keepdim=True)

    wrapped_model = ShapWrapper(model, model_type)
    background = torch.FloatTensor(X_scaled[:30])
    
    # GradientExplainer is very fast for single samples
    explainer = shap.GradientExplainer(wrapped_model, background)
    
    sample_tensor = torch.FloatTensor(sample_data)
    shap_vals = explainer.shap_values(sample_tensor)
    
    # GradientExplainer usually returns (samples, features)
    return np.array(shap_vals).flatten().tolist()

if __name__ == "__main__":
    train_models()
    export_to_onnx()
    print("Engine Optimized & Sync Complete.")

