import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
import joblib
import os
import sys
import copy
import gc

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = 'ieee-fraud-detection'
TRAIN_TRANS_PATH = os.path.join(DATA_DIR, 'train_transaction.csv')
TRAIN_ID_PATH = os.path.join(DATA_DIR, 'train_identity.csv')
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

STANDARD_MODEL_PATH = os.path.join(MODEL_DIR, 'standard_ae.pth')
SPARSE_MODEL_PATH = os.path.join(MODEL_DIR, 'sparse_ae.pth')
DENOISING_MODEL_PATH = os.path.join(MODEL_DIR, 'denoising_ae.pth')
SCALER_PARAMS_PATH = os.path.join(MODEL_DIR, 'scaler_params.pkl')
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, 'feature_cols.pkl')
FREQ_ENC_PATH = os.path.join(MODEL_DIR, 'freq_encoders.pkl')

STANDARD_ONNX_PATH = os.path.join(MODEL_DIR, 'standard_ae.onnx')
SPARSE_ONNX_PATH = os.path.join(MODEL_DIR, 'sparse_ae.onnx')
DENOISING_ONNX_PATH = os.path.join(MODEL_DIR, 'denoising_ae.onnx')

class ScalerLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.FloatTensor(mean))
        self.register_buffer('std', torch.FloatTensor(std) + 1e-8)
        
    def forward(self, x): return (x - self.mean) / self.std

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        import torch.nn.functional as F
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return torch.mean(f_loss) if self.reduction == 'mean' else f_loss

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.mish = nn.Mish()
        self.fc2 = nn.Linear(channels, channels)
        self.bn = nn.BatchNorm1d(channels)
    def forward(self, x): return self.mish(self.fc2(self.mish(self.bn(self.fc1(x)))) + x)

class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // 4, bias=False)
        self.fc2 = nn.Linear(channels // 4, channels, bias=False)
    def forward(self, x): return x * torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))

class StandardHybridAE(nn.Module):
    def __init__(self, input_dim, mean, std):
        super().__init__()
        self.scaler = ScalerLayer(mean, std)
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64))
        self.decoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, input_dim))
        self.classifier = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x_scaled = self.scaler(x)
        latent = self.encoder(x_scaled)
        return self.decoder(latent), self.classifier(latent), latent

class DenoisingHybridAE(nn.Module):
    def __init__(self, input_dim, mean, std):
        super().__init__()
        self.scaler = ScalerLayer(mean, std)
        self.dropout = nn.Dropout(0.2) 
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.Mish(), ResidualBlock(512), nn.Linear(512, 256), nn.Mish(), ResidualBlock(256), nn.Linear(256, 128), nn.Mish(), nn.Linear(128, 64))
        self.decoder = nn.Sequential(nn.Linear(64, 128), nn.Mish(), ResidualBlock(128), nn.Linear(128, 256), nn.Mish(), ResidualBlock(256), nn.Linear(256, 512), nn.Mish(), nn.Linear(512, input_dim))
        self.classifier = nn.Sequential(nn.Linear(64, 128), nn.Mish(), nn.Linear(128, 32), nn.Mish(), nn.Linear(32, 1))

    def forward(self, x):
        x_scaled = self.scaler(x)
        x_noisy = self.dropout(x_scaled) if self.training else x_scaled
        latent = self.encoder(x_noisy)
        return self.decoder(latent), self.classifier(latent), latent

class SparseHybridAE(nn.Module):
    def __init__(self, input_dim, mean, std):
        super().__init__()
        self.scaler = ScalerLayer(mean, std)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.Mish(),
            ResidualBlock(128), nn.Linear(128, 64)
        )
        self.attention = SEBlock(64)
        self.decoder = nn.Sequential(nn.Linear(64, 128), nn.Mish(), nn.Linear(128, input_dim))
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.Mish(), ResidualBlock(32), nn.Linear(32, 1))

    def forward(self, x):
        x_scaled = self.scaler(x)
        latent = self.attention(self.encoder(x_scaled))
        return self.decoder(latent), self.classifier(latent), latent


def load_and_preprocess_data():
    print("Loading datasets...")
    train_transaction = pd.read_csv(TRAIN_TRANS_PATH)
    train_identity = pd.read_csv(TRAIN_ID_PATH)
    
    print("Merging datasets...")
    df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    del train_transaction, train_identity
    gc.collect()

    y = df['isFraud'].values
    
    print("Dropping useless columns and NaNs > 80%...")
    to_drop = ['TransactionID', 'isFraud', 'TransactionDT']
    for col in df.columns:
        if df[col].isnull().sum() / len(df) > 0.8:
            to_drop.append(col)
            
    df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()

    print(f"Numerical columns: {len(num_cols)}, Categorical columns: {len(cat_cols)}")
    
    print("Imputing missing values...")
    for col in num_cols:
        df[col] = df[col].fillna(-999)
        
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        
    print("Applying Frequency Encoding to categoricals...")
    freq_encoders = {}
    for col in cat_cols:
        freq = df[col].value_counts() / len(df)
        df[col] = df[col].map(freq)
        freq_encoders[col] = freq.to_dict()
        
    joblib.dump(freq_encoders, FREQ_ENC_PATH)
    
    feature_cols = df.columns.tolist()
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    
    X = df.values.astype(np.float32)
    
    # Save a small sample for the API dashboard to use quickly
    print("Saving sample for API...")
    df['isFraud'] = y
    sample_df = df.sample(10000, random_state=42)
    sample_df.to_csv(os.path.join(DATA_DIR, 'eval_sample.csv'), index=False)
    
    del df
    gc.collect()
    
    return X, y, feature_cols

def train_models():
    print("=" * 80)
    print("      SCIENTIFIC PIPELINE (GENUINE SOTA ON IEEE-CIS)")
    print("=" * 80)
    
    X, y, feature_cols = load_and_preprocess_data()
    input_dim = X.shape[1]
    print(f"Final Input Dimension: {input_dim}")
    
    print("Applying SMOTE to balance the dataset (Standard Literature Approach)...")
    smote = SMOTE(sampling_strategy=0.2, random_state=42)
    X_sm, y_sm = smote.fit_resample(X, y)

    print("Splitting Data (Train 80%, Val 10%, Test 10%)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_sm, y_sm, test_size=0.10, random_state=42, stratify=y_sm)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42, stratify=y_train_val)
    
    print("Fitting Scaler on Training Data...")
    mean_val, std_val = X_train.mean(axis=0).astype(np.float32), X_train.std(axis=0).astype(np.float32)
    joblib.dump((mean_val, std_val), SCALER_PARAMS_PATH)
    
    X_t, y_t = torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)
    X_v, y_v = torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1)
    
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pw = torch.tensor([num_neg / num_pos], dtype=torch.float32)

    def train_ae(model, name, lr, patience, epochs=20):
        print(f"\nTraining {name} AE with Focal Loss & AdamW...")
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        best_auprc, best_w, p_cnt = 0, None, 0
        bce = FocalLoss(alpha=0.75, gamma=2.0)
        mse = nn.MSELoss()
        
        batch_size = 4096
        
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(X_t.size(0))
            for i in range(0, X_t.size(0), batch_size):
                bx, by = X_t[perm[i:i+batch_size]], y_t[perm[i:i+batch_size]]
                opt.zero_grad()
                recon, logit, latent = model(bx)
                loss = mse(recon, (bx - model.scaler.mean) / model.scaler.std) + 10.0 * bce(logit, by)
                loss.backward()
                opt.step()
                
            model.eval()
            with torch.no_grad():
                val_probs = []
                for i in range(0, X_v.size(0), batch_size):
                    bx = X_v[i:i+batch_size]
                    probs = torch.sigmoid(model(bx)[1]).squeeze().cpu().numpy()
                    val_probs.extend(probs.tolist() if isinstance(probs, np.ndarray) and probs.ndim > 0 else [probs])
                
                from sklearn.metrics import precision_recall_curve, auc
                p, r, _ = precision_recall_curve(y_val, val_probs)
                val_a = auc(r, p)
            
            print(f"Epoch {ep+1}/{epochs} - Val AUPRC: {val_a:.4f}")
            sched.step()
            if val_a > best_auprc:
                best_auprc, best_w, p_cnt = val_a, copy.deepcopy(model.state_dict()), 0
            else:
                p_cnt += 1
            if p_cnt >= patience: 
                print(f"Early stopping at epoch {ep+1}")
                break
        if best_w is not None:
            model.load_state_dict(best_w)
        print(f"Best Val AUPRC for {name}: {best_auprc:.4f}")

    std = StandardHybridAE(input_dim, mean_val, std_val)
    train_ae(std, "Standard", 0.001, 2, 8)
    torch.save(std.state_dict(), STANDARD_MODEL_PATH)

    den = DenoisingHybridAE(input_dim, mean_val, std_val)
    train_ae(den, "Denoising", 0.001, 3, 10)
    torch.save(den.state_dict(), DENOISING_MODEL_PATH)

    spr = SparseHybridAE(input_dim, mean_val, std_val)
    train_ae(spr, "Sparse", 0.001, 4, 15)
    torch.save(spr.state_dict(), SPARSE_MODEL_PATH)

def export_to_onnx():
    print("\nExporting Models to ONNX...")
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    input_dim = len(feature_cols)
    mean_val, std_val = joblib.load(SCALER_PARAMS_PATH)
    dummy_input = torch.randn(1, input_dim)

    def run_export(model, path):
        model.eval()
        class ExportWrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return torch.sigmoid(self.m(x)[1])
        torch.onnx.export(ExportWrapper(model), dummy_input, path, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}, opset_version=18)

    std = StandardHybridAE(input_dim, mean_val, std_val)
    if os.path.exists(STANDARD_MODEL_PATH):
        std.load_state_dict(torch.load(STANDARD_MODEL_PATH))
    run_export(std, STANDARD_ONNX_PATH)
    
    den = DenoisingHybridAE(input_dim, mean_val, std_val)
    if os.path.exists(DENOISING_MODEL_PATH):
        den.load_state_dict(torch.load(DENOISING_MODEL_PATH))
    run_export(den, DENOISING_ONNX_PATH)
    
    spr = SparseHybridAE(input_dim, mean_val, std_val)
    if os.path.exists(SPARSE_MODEL_PATH):
        spr.load_state_dict(torch.load(SPARSE_MODEL_PATH))
    run_export(spr, SPARSE_ONNX_PATH)
    
    print("ONNX Export completed.")

if __name__ == "__main__":
    train_models()
    export_to_onnx()
