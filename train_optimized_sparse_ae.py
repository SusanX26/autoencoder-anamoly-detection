import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, f1_score
import joblib
import os
import sys
import onnx
import onnxruntime

# Set encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIG ---
DATA_PATH = 'creditcard.csv'
MODEL_DIR = 'models_optimized'
os.makedirs(MODEL_DIR, exist_ok=True)

SCALER_PATH = os.path.join(MODEL_DIR, 'fast_scaler.pkl')
ENSEMBLE_PATH = os.path.join(MODEL_DIR, 'ensemble_metadata.json')

# --- DATA PREPARATION ---

def prepare_data():
    print(f"Loading full dataset from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH).fillna(0)
    
    # Feature Engineering
    df['Amount'] = np.log1p(df['Amount'])
    
    # Drop irrelevant columns
    X = df.drop(['Class'], axis=1)
    if 'Time' in X.columns: X = X.drop(['Time'], axis=1)
    if 'id' in X.columns: X = X.drop(['id'], axis=1)
    
    y = df['Class'].values
    
    # Use StandardScaler for Ultra-Fast Inference (< 1ms)
    print("Applying StandardScaler (Fast Mode)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    
    # Split into Normal (for training) and Full (for validation)
    X_normal = X_scaled[y == 0]
    X_fraud = X_scaled[y == 1]
    
    X_train, X_val_normal = train_test_split(X_normal, test_size=0.1, random_state=42)
    
    # Create a balanced validation set for threshold optimization
    # We use all fraud cases and an equal number of normal cases
    X_val = np.vstack([X_val_normal[:len(X_fraud)], X_fraud])
    y_val = np.array([0]*len(X_fraud) + [1]*len(X_fraud))
    
    return X_train, X_val, y_val, X.shape[1]

# --- HYBRID AUTOENCODER ---

class ExtremeSparseAE(nn.Module):
    def __init__(self, input_dim):
        super(ExtremeSparseAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Linear(64, 32) # Bottleneck
        )
        # Decoder
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
        return reconstructed, latent

# --- LOSS FUNCTIONS ---

def kl_divergence(rho, rho_hat):
    """rho is target sparsity, rho_hat is actual average activation"""
    rho_hat = torch.mean(rho_hat, dim=0) # average over batch
    # Add small epsilon to avoid log(0)
    rho_hat = torch.clamp(rho_hat, 1e-10, 1.0 - 1e-10)
    return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))

def contractive_loss(model, x, latent):
    """Frobenius norm of the Jacobian matrix of the encoder"""
    # This is a simplified version: penalty on gradients of latent w.r.t input
    # In a full Jacobian it would be sum of squared partial derivatives
    # For efficiency in PyTorch, we can use a penalty on the weights of the encoder layers
    penalty = 0
    for param in model.encoder.parameters():
        if len(param.shape) > 1: # Only weights, not biases
            penalty += torch.sum(param**2)
    return 1e-4 * penalty

# --- TRAINING ENGINE ---

def train_ensemble():
    X_train, X_val, y_val, input_dim = prepare_data()
    X_train_tensor = torch.FloatTensor(X_train)
    
    # 1. Train Sparse-Contractive AE
    print("\n--- Training Extreme Sparse-Contractive AE ---")
    model = ExtremeSparseAE(input_dim)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    BATCH_SIZE = 1024
    EPOCHS = 50
    TARGET_SPARSITY = 0.05
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        permutation = torch.randperm(X_train_tensor.size()[0])
        
        for i in range(0, X_train_tensor.size()[0], BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x = X_train_tensor[indices]
            
            optimizer.zero_grad()
            recon, latent = model(batch_x)
            
            # Hybrid Loss
            recon_loss = nn.MSELoss()(recon, batch_x)
            sparsity_loss = 0.01 * kl_divergence(TARGET_SPARSITY, torch.sigmoid(latent))
            c_loss = contractive_loss(model, batch_x, latent)
            
            loss = recon_loss + sparsity_loss + c_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/(X_train_tensor.size()[0]/BATCH_SIZE):.6f}")

    # 2. Train Isolation Forest as part of ensemble
    print("\n--- Training Isolation Forest Component ---")
    iforest = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)
    iforest.fit(X_train)
    
    # 3. Ensemble Evaluation & Threshold Optimization
    print("\n--- Optimizing Ensemble Consensus ---")
    model.eval()
    with torch.no_grad():
        val_recon, _ = model(torch.FloatTensor(X_val))
        ae_errors = torch.mean((val_recon - torch.FloatTensor(X_val))**2, dim=1).numpy()
    
    if_scores = -iforest.decision_function(X_val) # Negative because decision_function returns higher for normal
    
    # Normalize scores for consensus
    ae_errors_norm = (ae_errors - ae_errors.min()) / (ae_errors.max() - ae_errors.min())
    if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
    
    # Consensus Score (Weighted average)
    consensus_score = 0.7 * ae_errors_norm + 0.3 * if_scores_norm
    
    # Find best threshold on validation set
    precision, recall, thresholds = precision_recall_curve(y_val, consensus_score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    auprc = auc(recall, precision)
    
    print(f"Optimization Results:")
    print(f"  Best Threshold: {best_threshold:.4f}")
    print(f"  Best F1-Score: {best_f1:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    
    # Save everything
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'extreme_sae.pth'))
    joblib.dump(iforest, os.path.join(MODEL_DIR, 'iforest_ensemble.pkl'))
    
    metadata = {
        "best_threshold": float(best_threshold),
        "auprc": float(auprc),
        "f1_score": float(best_f1),
        "ae_weight": 0.7,
        "if_weight": 0.3,
        "input_dim": input_dim
    }
    import json
    with open(ENSEMBLE_PATH, 'w') as f:
        json.dump(metadata, f)
        
    print("\nTraining Complete. Metadata saved.")
    
    # Export AE to ONNX for fast inference
    export_onnx(model, input_dim)

def export_onnx(model, input_dim):
    print("Exporting Sparse AE to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, input_dim)
    
    # Wrap to return only reconstruction
    class ExportWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): r, _ = self.m(x); return r
    
    onnx_path = os.path.join(MODEL_DIR, 'extreme_sae.onnx')
    torch.onnx.export(ExportWrapper(model), dummy_input, onnx_path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                      opset_version=17)
    print(f"ONNX Model saved to {onnx_path}")

if __name__ == "__main__":
    train_ensemble()
