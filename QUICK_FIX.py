import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 1. Create a MINI dummy dataset
print("Creating dummy dataset...")
columns = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class', 'id', 'Time']
data = np.random.randn(100, len(columns))
df = pd.DataFrame(data, columns=columns)
df['Class'] = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
df['id'] = range(100)
df.to_csv('creditcard_2023.csv', index=False)

# 2. Create a dummy scaler
print("Creating dummy scaler...")
scaler = StandardScaler()
scaler.fit(np.random.randn(10, 29))
joblib.dump(scaler, 'scaler.pkl')

# 3. Create a dummy ONNX model so API doesn't crash
print("Creating placeholder model...")
class SimpleAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(29, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 29))
    def forward(self, x): return self.decoder(self.encoder(x))

model = SimpleAE()
dummy_input = torch.randn(1, 29)
torch.onnx.export(model, dummy_input, "autoencoder_fraud.onnx")

print("\nSUCCESS! Now run START_FRAUDSENSE.bat again.")
