# FraudSense AI: Client Setup & Execution Guide

This guide is designed for the client to set up the FraudSense AI project on their local machine and run it using the **Antigravity AI Coding Assistant**.

## 1. Prerequisites
- **Python 3.9+** installed.
- **Node.js (LTS)** installed.
- **Antigravity AI Assistant** access.

## 2. Local Setup Instructions
1.  **Clone/Download the Code**: Download the project folder from GitHub.
2.  **Add the Dataset**: 
    - Copy your `creditcard_2023.csv` file.
    - Create a folder named `creditcard_2023.csv` in the root directory (if it doesn't exist).
    - Place the `creditcard_2023.csv` file inside that folder: `creditcard_2023.csv/creditcard_2023.csv`.
    - Alternatively, just place the `creditcard_2023.csv` file directly in the root directory.

## 3. The Antigravity Launch Prompt
Once the folder is open in your IDE with Antigravity, copy and paste the following prompt exactly:

> **"I have successfully downloaded the FraudSense AI project from GitHub and placed the 'creditcard_2023.csv' dataset in the root directory. Please perform a full system initialization: 1) Set up the Python virtual environment and install all dependencies from requirements.txt, 2) Verify the dataset and run 'fraud_detector_engine.py' to train and export the Standard and Sparse models, 3) Install the dashboard dependencies in the 'dashboard' folder, and 4) Launch the API server and the real-time Dashboard so I can begin monitoring transactions. Ensure the XAI (SHAP) charts and model architecture comparisons are fully functional."**

---

## 4. Manual Launch (Alternative)
If you prefer to run it manually without the prompt:
1.  Double-click `START_FRAUDSENSE.bat`.
2.  Wait for the automated setup to complete.
3.  The dashboard will open automatically in your browser at `http://localhost:5173`.
