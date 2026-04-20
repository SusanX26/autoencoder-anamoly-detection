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

# --- CONFIG ---
DATA_PATH = r'd:\deadshot file\computer_ER_FIles\susan shrestha sir mit aus\creditcard_2023.csv\creditcard_2023.csv'
MODEL_PATH = 'autoencoder_fraud.pth'
ONNX_PATH = 'autoencoder_fraud.onnx'
SCALER_PATH = 'scaler.pkl'

# --- AUTOENCODER MODEL ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8) 
        )
        # Decoder
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

def train_model():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Drop id and Class for training (unsupervised)
    # Even though it's balanced, we train on Class 0 to learn "Normal" patterns
    X_normal = df[df['Class'] == 0].drop(['id', 'Class'], axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, SCALER_PATH)
    
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
    
    X_train_tensor = torch.FloatTensor(X_train)
    
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training...")
    epochs = 10
    batch_size = 1024
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])
        epoch_loss = 0
        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_train_tensor[indices]
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_train_tensor):.6f}")

import joblib

def export_to_onnx():
    print("Loading model for ONNX export...")
    # Get input dim from data
    df = pd.read_csv(DATA_PATH, nrows=1)
    input_dim = df.drop(['id', 'Class'], axis=1).shape[1]
    
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    dummy_input = torch.randn(1, input_dim)
    torch.onnx.export(model, dummy_input, ONNX_PATH, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"ONNX exported to {ONNX_PATH}")

def get_shap_values(sample_data):
    # Load data for SHAP background (medoids or sample)
    df = pd.read_csv(DATA_PATH, nrows=500)
    X = df.drop(['id', 'Class'], axis=1)
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)
    
    input_dim = X.shape[1]
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    def model_predict(data):
        data_tensor = torch.FloatTensor(data)
        with torch.no_grad():
            recon = model(data_tensor)
            # Use sum of squared errors per row
            mse = torch.mean((recon - data_tensor)**2, dim=1)
        return mse.numpy()

    # Use a smaller background for speed in real-time
    explainer = shap.KernelExplainer(model_predict, shap.sample(X_scaled, 20))
    shap_values = explainer.shap_values(sample_data)
    return shap_values

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model()
    
    if not os.path.exists(ONNX_PATH):
        export_to_onnx()
    
    print("Engine Ready.")
