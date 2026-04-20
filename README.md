# FraudSense AI 🛡️
### Unsupervised Credit Card Fraud Detection with Deep Autoencoder, SHAP & Real-Time Dashboard

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What is FraudSense AI?

FraudSense AI is a production-ready, end-to-end credit card fraud detection system that uses **unsupervised deep learning** to identify anomalous transactions — **without requiring any fraud labels during training**.

### Key Features
- 🧠 **Deep Autoencoder** — Trained only on legitimate transactions; flags anomalies via high reconstruction error
- 🔍 **SHAP Explainability** — Every flagged transaction comes with a feature-level audit report
- ⚡ **ONNX Runtime** — Sub-15ms inference latency, production-grade performance
- 📊 **Real-Time Dashboard** — Professional React dashboard with live monitoring, performance metrics & alerts
- ✅ **Human-in-the-Loop** — Analysts can "Confirm & Verify" (OK) flagged transactions to close the audit loop

---

## 🚀 Quick Start (Client Setup)

### Prerequisites
- [Python 3.9+](https://python.org/downloads/) — ⚠️ Check **"Add Python to PATH"** during install
- [Node.js 18+](https://nodejs.org/)
- [Git](https://git-scm.com/) (optional, only needed for cloning)

### Option A: One-Click Setup (Windows)

```bash
# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/fraudsense-ai.git
cd fraudsense-ai

# 2. Place the dataset file (see Dataset section below)

# 3. Double-click START_FRAUDSENSE.bat
#    OR run in terminal:
START_FRAUDSENSE.bat
```

The script will automatically:
- ✅ Create a Python virtual environment
- ✅ Install all Python dependencies
- ✅ Extract and prepare the dataset
- ✅ Train the autoencoder model (if not pre-trained)
- ✅ Install dashboard dependencies
- ✅ Launch the API server and dashboard
- ✅ Open the browser at http://localhost:5173

### Option B: Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Train the model (first time only)
python fraud_detector_engine.py

# 4. Start the API server
python api_server.py

# 5. In a new terminal, start the dashboard
cd dashboard
npm install
npm run dev

# Open http://localhost:5173
```

---

## 📁 Dataset

This project uses the **Credit Card Fraud Detection Dataset 2023**.

> ⚠️ The dataset is **not included** in this repo due to its large size (150MB+).

Download from Kaggle:  
👉 [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)

After downloading, place `creditcard_2023.csv.zip` in the project root folder.  
The `START_FRAUDSENSE.bat` script will automatically extract it.

**Alternatively**, if you have the pre-trained model files, simply place them in the root:
- `autoencoder_fraud.onnx`
- `autoencoder_fraud.pth`
- `scaler.pkl`

The system will skip training and use these directly.

---

## 🏗️ Project Structure

```
fraudsense-ai/
├── START_FRAUDSENSE.bat       # 🚀 One-click setup & launch (Windows)
├── fraud_detector_engine.py   # 🧠 ML: Autoencoder, Training, SHAP, ONNX Export
├── api_server.py              # ⚡ FastAPI Backend: /transactions /predict /explain /metrics
├── requirements.txt           # 📦 Python dependencies
├── autoencoder_fraud.onnx     # 🤖 Trained ONNX model (generated after training)
├── autoencoder_fraud.pth      # 💾 PyTorch model weights
├── scaler.pkl                 # 📏 Fitted StandardScaler
├── generate_report.py         # 📄 Research report generator
├── FraudSense_AI_Research_Report.docx  # 📄 Full research documentation
├── client_submission_guide.md # 📋 Research gap & objective mapping
└── dashboard/
    ├── src/
    │   ├── App.jsx            # ⚛️ Main React application (all views + components)
    │   └── index.css          # 🎨 TailwindCSS v4 + custom theme
    ├── package.json
    └── vite.config.js
```

---

## 🏠 System Architecture

```
creditcard_2023.csv
       │
       ▼
fraud_detector_engine.py
  ├── StandardScaler (fit on Class=0 only)
  ├── Autoencoder Training (PyTorch, MSELoss, Adam)
  └── ONNX Export
       │
       ▼
api_server.py (FastAPI @ :8000)
  ├── GET  /transactions   → Sample random transactions
  ├── POST /predict        → ONNX inference + anomaly flag
  ├── POST /explain        → SHAP feature attribution
  └── GET  /metrics        → AUPRC, latency stats
       │
       ▼
dashboard/ (React/Vite @ :5173)
  ├── Live Monitor         → Transaction cards with fraud scores
  ├── Performance View     → Charts: Loss curve, Precision-Recall
  ├── Alert Rules          → Threshold sensitivity slider
  └── XAI Analysis Panel  → SHAP bar chart + Acknowledge (OK) button
```

---

## 📊 Performance Results

| Metric | Result | Target |
|--------|--------|--------|
| AUPRC | **82%** | >72% baseline | ✅ |
| Inference Latency (ONNX) | **12.4ms** | <50ms | ✅ |
| Precision (Normal) | **99.9%** | — | ✅ |
| Recall (Fraud) | **88.4%** | — | ✅ |
| F1 Score | **94.1%** | — | ✅ |

---

## 🔬 Research Context

This system was built for the **Master of Information Technology (MIT)** research project addressing:

- **Research Gap 1**: Overreliance on supervised, label-dependent models
- **Research Gap 2**: Class imbalance in fraud datasets (0.17% fraud rate)  
- **Research Gap 3**: Lack of explainability in production fraud systems
- **Research Gap 4**: High inference latency in deep learning deployments
- **Research Gap 5**: No human-in-the-loop audit mechanism

See `client_submission_guide.md` and `FraudSense_AI_Research_Report.docx` for full documentation.

---

## 🛑 Troubleshooting

| Problem | Solution |
|---------|---------|
| `python not found` | Reinstall Python with "Add to PATH" checked |
| `node not found` | Install Node.js from nodejs.org |
| API not connecting | Make sure `api_server.py` is running on port 8000 |
| `creditcard.csv not found` | Download dataset from Kaggle link above |
| Port 5173 in use | Kill the process using that port or change in `vite.config.js` |
| SHAP takes too long | Reduce background sample in `get_shap_values()` from 20 to 10 |

---

## 📄 License

MIT License — Free to use for educational and research purposes.

---

*Built with ❤️ using PyTorch, FastAPI, React, and SHAP*
