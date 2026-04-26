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
DATA_PATH = 'temp_data.csv'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'creditcard_2023.csv'
    if not os.path.exists(DATA_PATH):
        alternative_path = os.path.join('creditcard_2023.csv', 'creditcard_2023.csv')
        if os.path.exists(alternative_path):
            DATA_PATH = alternative_path

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

STANDARD_MODEL_PATH = os.path.join(MODEL_DIR, 'standard_ae.pth')
SPARSE_MODEL_PATH = os.path.join(MODEL_DIR, 'sparse_ae.pth')
STANDARD_ONNX_PATH = os.path.join(MODEL_DIR, 'standard_ae.onnx')
SPARSE_ONNX_PATH = os.path.join(MODEL_DIR, 'sparse_ae.onnx')
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
        # We need to return encoded for sparsity penalty during training
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# --- TRAINING ---

def train_models():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH).fillna(0)
    X_normal = df[df['Class'] == 0].drop(['id', 'Class'], axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)
    joblib.dump(scaler, SCALER_PATH)
    
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
    X_train_tensor = torch.FloatTensor(X_train)
    input_dim = X_train.shape[1]
    
    # 1. Train Standard Autoencoder
    print("\n[1/2] Training Standard Autoencoder...")
    std_model = StandardAutoencoder(input_dim)
    std_optimizer = optim.Adam(std_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
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
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train_tensor):.6f}")
    torch.save(std_model.state_dict(), STANDARD_MODEL_PATH)

    # 2. Train Sparse Autoencoder
    print("\n[2/2] Training Sparse Autoencoder (with L1 Sparsity)...")
    spr_model = SparseAutoencoder(input_dim)
    spr_optimizer = optim.Adam(spr_model.parameters(), lr=0.001)
    
    # Sparsity parameters
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
            
            # Loss = Reconstruction Error + Sparsity Penalty
            mse_loss = criterion(reconstructed, batch_x)
            l1_loss = torch.mean(torch.abs(latent)) # L1 penalty for sparsity
            loss = mse_loss + l1_lambda * l1_loss
            
            loss.backward()
            spr_optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train_tensor):.6f}")
    torch.save(spr_model.state_dict(), SPARSE_MODEL_PATH)

# --- EXPORT ---

def export_to_onnx():
    print("\nExporting models to ONNX...")
    df = pd.read_csv(DATA_PATH, nrows=1).fillna(0)
    input_dim = df.drop(['id', 'Class'], axis=1).shape[1]
    dummy_input = torch.randn(1, input_dim)

    # Standard
    std_model = StandardAutoencoder(input_dim)
    std_model.load_state_dict(torch.load(STANDARD_MODEL_PATH))
    std_model.eval()
    torch.onnx.export(std_model, dummy_input, STANDARD_ONNX_PATH, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    # Sparse
    spr_model = SparseAutoencoder(input_dim)
    spr_model.load_state_dict(torch.load(SPARSE_MODEL_PATH))
    spr_model.eval()
    # Note: Onnx export for sparse needs to handle the multi-output, but for inference we only need reconstructed
    # So we wrap it or just use the reconstructed output
    class ExportWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            recon, _ = self.model(x)
            return recon
    
    torch.onnx.export(ExportWrapper(spr_model), dummy_input, SPARSE_ONNX_PATH, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print("ONNX Models Exported.")

# --- SHAP ---

def get_shap_values(sample_data, model_type='standard'):
    df = pd.read_csv(DATA_PATH, nrows=200).fillna(0)
    X = df.drop(['id', 'Class'], axis=1)
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)
    input_dim = X.shape[1]
    
    if model_type == 'sparse':
        model = SparseAutoencoder(input_dim)
        model.load_state_dict(torch.load(SPARSE_MODEL_PATH))
        def model_predict(data):
            data_tensor = torch.FloatTensor(data)
            with torch.no_grad():
                recon, _ = model(data_tensor)
                mse = torch.mean((recon - data_tensor)**2, dim=1)
            return mse.numpy()
    else:
        model = StandardAutoencoder(input_dim)
        model.load_state_dict(torch.load(STANDARD_MODEL_PATH))
        def model_predict(data):
            data_tensor = torch.FloatTensor(data)
            with torch.no_grad():
                recon = model(data_tensor)
                mse = torch.mean((recon - data_tensor)**2, dim=1)
            return mse.numpy()

    model.eval()
    explainer = shap.KernelExplainer(model_predict, shap.sample(X_scaled, 20))
    shap_values = explainer.shap_values(sample_data)
    return shap_values

if __name__ == "__main__":
    train_models()
    export_to_onnx()
    print("Engine Ready.")
