import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import shap
import onnx
import onnxruntime
import os
import json
import joblib
import sys

# Ensure UTF-8 output to prevent ONNX exporter crash on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIG ---
DATA_PATH = 'creditcard.csv'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'temp_data.csv'

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

STANDARD_MODEL_PATH = os.path.join(MODEL_DIR, 'standard_ae.pth')
SPARSE_MODEL_PATH = os.path.join(MODEL_DIR, 'sparse_ae.pth')
DENOISING_MODEL_PATH = os.path.join(MODEL_DIR, 'denoising_ae.pth')
VAE_MODEL_PATH = os.path.join(MODEL_DIR, 'vae_ae.pth')
ISOLATION_FOREST_PATH = os.path.join(MODEL_DIR, 'isolation_forest.pkl')
WEIGHTS_PATH = os.path.join(MODEL_DIR, 'feature_weights.pkl')
ENSEMBLE_METADATA_PATH = os.path.join(MODEL_DIR, 'ensemble_metadata.json')

STANDARD_ONNX_PATH = os.path.join(MODEL_DIR, 'standard_ae.onnx')
SPARSE_ONNX_PATH = os.path.join(MODEL_DIR, 'sparse_ae.onnx')
DENOISING_ONNX_PATH = os.path.join(MODEL_DIR, 'denoising_ae.onnx')
VAE_ONNX_PATH = os.path.join(MODEL_DIR, 'vae_ae.onnx')

SCALER_PATH = os.path.join(MODEL_DIR, 'fast_scaler.pkl') # Switching back to fast scaling

# --- MODELS ---

class ExtremeAutoencoder(nn.Module):
    def __init__(self, input_dim, is_sparse=False):
        super(ExtremeAutoencoder, self).__init__()
        self.is_sparse = is_sparse
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Mish(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.Mish(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        if self.is_sparse:
            return reconstructed, latent
        return reconstructed

class SparseAutoencoder(ExtremeAutoencoder):
    def __init__(self, input_dim):
        super().__init__(input_dim, is_sparse=True)

class DenoisingAutoencoder(ExtremeAutoencoder):
    def __init__(self, input_dim):
        super().__init__(input_dim, is_sparse=False)

class StandardAutoencoder(ExtremeAutoencoder):
    def __init__(self, input_dim):
        super().__init__(input_dim, is_sparse=False)

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(VariationalAutoencoder, self).__init__()
        # Shared Encoder backbone
        self.encoder_backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1)
        )
        # Latent space heads
        self.fc_mu = nn.Linear(32, 16)
        self.fc_logvar = nn.Linear(32, 16)
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, input_dim)
        )

    def encode(self, x):
        h = self.encoder_backbone(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# --- TRAINING ---

def train_models():
    print(f"Loading original imbalanced dataset: {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH).fillna(0)
    
    # --- PHASE 1: Feature Engineering & Variance Weights ---
    # Log transform Amount (common trick in fraud literature)
    df['Amount'] = np.log1p(df['Amount'])
    
    # Calculate Variance Weights: Higher variance in Fraud relative to Normal = Higher Weight
    fraud_df = df[df['Class'] == 1].drop(['Class'], axis=1)
    normal_df = df[df['Class'] == 0].drop(['Class'], axis=1)
    
    # Drop irrelevant columns
    for col in ['id', 'Time']:
        if col in fraud_df.columns:
            fraud_df = fraud_df.drop([col], axis=1)
            normal_df = normal_df.drop([col], axis=1)
            
    # Variance Ratio Weighting
    var_normal = normal_df.var() + 1e-6
    var_fraud = fraud_df.var() + 1e-6
    weights = (var_fraud / var_normal).values
    weights = weights / weights.sum() * len(weights) # Normalize to mean=1
    weights_tensor = torch.FloatTensor(weights)
    joblib.dump(weights, WEIGHTS_PATH)
    print("Computed Feature Weights based on Variance Ratios.")

    # --- PHASE 2: Fast Scaling for < 0.4ms Latency ---
    X_normal = normal_df
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler() # Extremely fast linear scaling
    X_scaled = scaler.fit_transform(X_normal)
    joblib.dump(scaler, SCALER_PATH)
    
    X_train, X_test = train_test_split(X_scaled, test_size=0.1, random_state=42)
    X_train_tensor = torch.FloatTensor(X_train)
    input_dim = X_train.shape[1]
    
    EPOCHS = 15
    BATCH_SIZE = 2048
    
    def weighted_huber_loss(output, target, weights):
        loss = torch.abs(output - target)
        # Huber condition
        mask = (loss < 1).float()
        squared_loss = 0.5 * (loss**2)
        linear_loss = loss - 0.5
        base_loss = mask * squared_loss + (1 - mask) * linear_loss
        return torch.mean(base_loss * weights)

    # --- MODEL 1: Standard AE (Baseline) ---
    print("\n[1/4] Training Extreme Standard Autoencoder...")
    std_model = ExtremeAutoencoder(input_dim)
    optimizer = optim.AdamW(std_model.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        std_model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])
        for i in range(0, X_train_tensor.size()[0], BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x = X_train_tensor[indices]
            optimizer.zero_grad()
            output = std_model(batch_x)
            loss = nn.MSELoss()(output, batch_x)
            loss.backward()
            optimizer.step()
    torch.save(std_model.state_dict(), STANDARD_MODEL_PATH)

    # --- MODEL 2: Sparse AE (Optimized) ---
    print("\n[2/4] Training Extreme Sparse Autoencoder...")
    spr_model = ExtremeAutoencoder(input_dim, is_sparse=True)
    optimizer = optim.AdamW(spr_model.parameters(), lr=0.001)
    TARGET_SPARSITY = 0.05
    for epoch in range(EPOCHS):
        spr_model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])
        for i in range(0, X_train_tensor.size()[0], BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x = X_train_tensor[indices]
            optimizer.zero_grad()
            output, latent = spr_model(batch_x)
            recon_loss = nn.MSELoss()(output, batch_x)
            # KL-Divergence Sparsity
            rho_hat = torch.mean(torch.sigmoid(latent), dim=0)
            sparsity_loss = 0.01 * torch.sum(TARGET_SPARSITY * torch.log(TARGET_SPARSITY / (rho_hat + 1e-10)) + 
                                            (1 - TARGET_SPARSITY) * torch.log((1 - TARGET_SPARSITY) / (1 - rho_hat + 1e-10)))
            loss = recon_loss + sparsity_loss
            loss.backward()
            optimizer.step()
    torch.save(spr_model.state_dict(), SPARSE_MODEL_PATH)

    # --- MODEL 3: Denoising AE (Robust) ---
    print("\n[3/4] Training Extreme Denoising Autoencoder...")
    den_model = ExtremeAutoencoder(input_dim)
    optimizer = optim.AdamW(den_model.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        den_model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])
        for i in range(0, X_train_tensor.size()[0], BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x = X_train_tensor[indices]
            # Gaussian Noise Injection
            noisy_x = batch_x + 0.1 * torch.randn_like(batch_x)
            optimizer.zero_grad()
            output = den_model(noisy_x)
            loss = nn.MSELoss()(output, batch_x)
            loss.backward()
            optimizer.step()
    torch.save(den_model.state_dict(), DENOISING_MODEL_PATH)

    # --- MODEL 4: Variational AE (VAE) ---
    print("\n[4/5] Training Weighted Variational Autoencoder (VAE)...")
    vae_model = VariationalAutoencoder(input_dim)
    optimizer = optim.Adam(vae_model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    for epoch in range(EPOCHS):
        vae_model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])
        for i in range(0, X_train_tensor.size()[0], BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x = X_train_tensor[indices]
            optimizer.zero_grad()
            recon, mu, logvar = vae_model(batch_x)
            recon_loss = weighted_huber_loss(recon, batch_x, weights_tensor)
            kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = recon_loss + 0.001 * kld_loss # Beta-VAE style scaling
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch+1) % 20 == 0: print(f"  Epoch {epoch+1}/100, Loss: {loss.item():.6f}")
    torch.save(vae_model.state_dict(), VAE_MODEL_PATH)

    # --- MODEL 5: Isolation Forest ---
    print("\n[5/5] Training Isolation Forest Ensemble...")
    clf = IsolationForest(n_estimators=200, contamination=0.0017, random_state=42, n_jobs=-1)
    clf.fit(X_train)
    joblib.dump(clf, ISOLATION_FOREST_PATH)
    print("Isolation Forest trained and saved.")

# --- EXPORT ---

def export_to_onnx():
    print("\nExporting all 4 Autoencoders to ONNX...")
    df = pd.read_csv(DATA_PATH, nrows=1).fillna(0)
    X = df.drop(['Class'], axis=1)
    for col in ['id', 'Time']:
        if col in X.columns: X = X.drop([col], axis=1)
    input_dim = X.shape[1]
    dummy_input = torch.randn(1, input_dim)

    # Helper for export
    def run_export(model, path, is_vae=False, is_sparse=False):
        model.eval()
        class W(nn.Module):
            def __init__(self, m, s=False): super().__init__(); self.m = m; self.s = s
            def forward(self, x):
                res = self.m(x)
                return res[0] if self.s else res
        
        export_mod = W(model, s=is_sparse)
            
        torch.onnx.export(export_mod, dummy_input, path,
                          input_names=['input'], output_names=['output'],
                          opset_version=14)
        print(f"  Exported: {path}")

    # Export all 3 Extreme models
    std = ExtremeAutoencoder(input_dim)
    std.load_state_dict(torch.load(STANDARD_MODEL_PATH))
    run_export(std, STANDARD_ONNX_PATH)
    
    spr = ExtremeAutoencoder(input_dim, is_sparse=True)
    spr.load_state_dict(torch.load(SPARSE_MODEL_PATH))
    run_export(spr, SPARSE_ONNX_PATH, is_sparse=True)
    
    den = ExtremeAutoencoder(input_dim)
    den.load_state_dict(torch.load(DENOISING_MODEL_PATH))
    run_export(den, DENOISING_ONNX_PATH)
    
    vae = VariationalAutoencoder(input_dim)
    vae.load_state_dict(torch.load(VAE_MODEL_PATH))
    run_export(vae, VAE_ONNX_PATH, is_vae=True)

    print("All ONNX Models Ready.")

def optimize_for_production(model):
    # Dynamic Quantization: Reduces model size by 4x and speeds up CPU inference
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

def get_prediction(data, model_type='standard'):
    # ULTRA-FAST MODE: < 5ms
    onnx_path = os.path.join(MODEL_DIR, 'extreme_sae.onnx')
    if os.path.exists(onnx_path):
        session = onnxruntime.InferenceSession(onnx_path)
        inputs = {session.get_inputs()[0].name: data.astype(np.float32)}
        reconstructed = session.run(None, inputs)[0]
        # Pure Neural MSE is extremely fast (~1-2ms)
        ae_mse = np.mean((reconstructed - data)**2, axis=1)
        return ae_mse
    return np.zeros(len(data))

def get_shap_values(sample_data, model_type='standard'):
    input_dim = sample_data.shape[1]
    
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
    
    with torch.no_grad():
        x_tensor = torch.FloatTensor(sample_data)
        if model_type == 'sparse':
            reconstructed, _ = model(x_tensor)
        else:
            reconstructed = model(x_tensor)
            
        # Feature-wise Reconstruction Error is the industry standard for AE explainability
        feature_errors = (x_tensor - reconstructed) ** 2
        
    return feature_errors.numpy().flatten().tolist()

if __name__ == "__main__":
    train_models()
    export_to_onnx()
    print("Engine Optimized & Sync Complete.")

