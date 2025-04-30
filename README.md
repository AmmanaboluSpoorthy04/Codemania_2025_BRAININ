# Codemania_2025_BRAININ

# Credit Card Fraud Detection 💳🚫

This project is a **machine learning-based web application** that predicts whether a credit card transaction is fraudulent or not based on provided input features.

## 🔍 Overview

- Backend: Python (Google Colab + Flask)
- Frontend: HTML (Flask Templates)
- Model: Trained on anonymized credit card transaction data
- Deployment: Localhost (can be extended to cloud platforms)
- Source Control: Git & GitHub

## ⚙️ How It Works

1. User provides transaction `Time` and `Amount`.
2. The app generates the remaining 28 anonymized features randomly (for simulation).
3. The trained model predicts whether the transaction is **Fraudulent (1)** or **Non-Fraudulent (0)**.
4. The result is displayed on the webpage.

## 🧪 Sample Inputs

You can test with values like:

- **Time**: `50000`
- **Amount**: `120.50`

## 🚀 Getting Started

### Requirements

- Python 3.x
- Flask
- scikit-learn
- joblib
- numpy
