# FraudSense AI - Client Setup Guide

## The Antigravity Prompt (Copy this EXACTLY into Antigravity)

---

I want to set up and run the FraudSense AI project locally on my computer. Here is exactly what I need you to do, step by step:

**Project:** FraudSense AI - Credit Card Fraud Detection Dashboard
**GitHub Repo:** https://github.com/SusanX26/autoencoder-anamoly-detection
**Dataset Required:** creditcard_2023.csv (downloaded from Kaggle)

**My Setup:**
- I have already downloaded the project folder from GitHub
- I have placed my creditcard_2023.csv file inside the project folder
- The project folder is at: [REPLACE WITH YOUR FOLDER PATH e.g. C:\Users\YourName\fraudsense]

**Please do all of the following:**

1. First, check that creditcard_2023.csv exists in the project folder. If it exists, rename it to temp_data.csv (this is the name the engine expects).

2. Check that Python 3.9+ is installed. If not, tell me to install it from python.org.

3. Check that Node.js 18+ is installed. If not, tell me to install it from nodejs.org.

4. Create a Python virtual environment inside the project folder:
   - Run: python -m venv venv

5. Activate the virtual environment and install ALL Python dependencies:
   - Run: venv\Scripts\activate (on Windows)
   - Run: pip install torch torchvision fastapi uvicorn pandas numpy scikit-learn onnx onnxruntime shap joblib python-multipart

6. Train the AI models by running:
   - python fraud_detector_engine.py
   - This will take about 3-5 minutes and will create the models/ folder with the trained files.

7. Install the dashboard (frontend) dependencies:
   - Navigate into the dashboard/ folder
   - Run: npm install

8. Now start BOTH servers at the same time:
   - Server 1 (API): In one terminal run: venv\Scripts\activate && python api_server.py
   - Server 2 (Dashboard): In another terminal run: cd dashboard && npm run dev

9. Open the browser at http://localhost:5173

10. Verify everything is working:
    - Transactions should load on the main screen
    - Clicking a transaction should show the XAI/SHAP chart
    - The Performance tab should show all 4 charts with data
    - The Compare Models tab should show the comparison table

If anything fails, show me the exact error message and fix it.

---

## Alternative: One-Click Setup (Windows Only)

If you are on Windows, simply:
1. Place creditcard_2023.csv in the project folder
2. Rename it to temp_data.csv
3. Double-click START_FRAUDSENSE.bat

The bat file will do everything automatically.

---

## What the System Does

- **Live Monitor Tab**: Shows 12 real transactions sampled from your dataset with fraud/safe labels
- **Clicking a Transaction**: Opens the forensic panel with SHAP feature attribution chart (real XAI)
- **Performance Tab**: Shows AUPRC, F1, Latency metrics + Learning Curve + Reconstruction Distribution + Top Features
- **Compare Models Tab**: Side-by-side table comparing Standard vs Sparse Autoencoder
- **Alert Rules Tab**: Sensitivity slider to adjust the fraud detection threshold

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `torch` install fails | Run: pip install torch --index-url https://download.pytorch.org/whl/cpu |
| API server crash on start | Make sure models/ folder exists - run fraud_detector_engine.py first |
| SHAP chart shows nothing | Click the transaction and wait 3-5 seconds (SHAP computation takes time) |
| Port 5173 already in use | Kill the process or change the port in dashboard/vite.config.js |
| creditcard_2023.csv not found | Make sure the file is in the ROOT project folder (same level as api_server.py) |
