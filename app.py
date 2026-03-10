# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

# ===============================
# DEVICE AND MODEL SETUP
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IDSModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ===============================
# LOAD MODEL AND SCALER
# ===============================
scaler = joblib.load("feature_scaler.pkl")
input_dim = scaler.mean_.shape[0]

model = IDSModel(input_dim)
model.load_state_dict(torch.load("global_federated_ids_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Privacy epsilon
epsilon = 345387.76

# ===============================
# STREAMLIT PAGE SETUP
# ===============================
st.set_page_config(page_title="Federated IDS Dashboard", layout="wide")

st.title("🔐 Federated Learning IDS Dashboard")
st.write("Intrusion Detection System using Federated Learning with Differential Privacy")

# Sidebar
st.sidebar.header("Model Information")
st.sidebar.write("Model: Federated Neural Network")
st.sidebar.write("Dataset: CICIDS2017")
st.sidebar.write(f"Privacy Budget (ε): {epsilon:.2f}")

# ===============================
# FILE UPLOAD
# ===============================
st.header("Network Traffic Input")

uploaded_file = st.file_uploader("Upload CSV file with network flows", type="csv")

if uploaded_file:

    df_input = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df_input.head())

    # ===============================
    # PREPROCESS DATA
    # ===============================

    # Extract label if it exists
    if "Label" in df_input.columns:
        y_true = df_input["Label"].values
        df_features = df_input.drop(columns=["Label"])
    else:
        y_true = None
        df_features = df_input.copy()

    # Keep only numeric columns
    df_features = df_features.select_dtypes(include=[np.number])

    # Replace infinite values
    df_features = df_features.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values
    df_features = df_features.fillna(0)

    # Convert to numpy
    X_input = df_features.values.astype(np.float32)

    # ===============================
    # FEATURE VALIDATION
    # ===============================
    if X_input.shape[1] != input_dim:

        st.error(
            f"""
Feature mismatch detected.

Model expects **{input_dim} features**  
Uploaded file contains **{X_input.shape[1]} numeric columns**

Ensure the CSV contains the same feature set used during model training.
"""
        )
        st.stop()

    # ===============================
    # SCALE FEATURES
    # ===============================
    X_scaled = scaler.transform(X_input)

    # ===============================
    # MODEL PREDICTION
    # ===============================
    features_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(model(features_tensor)).cpu().numpy().flatten()

    preds = (probs > 0.5).astype(int)

    # ===============================
    # ATTACK PROBABILITY GAUGE
    # ===============================
    st.subheader("Attack Probability (First Flow)")

    attack_prob = float(probs[0])

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=attack_prob * 100,
        title={'text': "Attack Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # TRAFFIC CLASSIFICATION
    # ===============================
    st.subheader("Traffic Classification (First Flow)")

    if attack_prob > 0.5:
        st.error("⚠️ Attack Detected")
    else:
        st.success("✅ Normal Traffic")

    # ===============================
    # MODEL EVALUATION
    # ===============================
    if y_true is not None:

        st.header("Model Evaluation")

        y_pred = preds
        y_scores = probs

        # ===============================
        # CONFUSION MATRIX
        # ===============================
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig_cm, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig_cm)

        st.caption(f"Confusion Matrix (ε ≈ {epsilon:.2f})")

        # ===============================
        # ROC CURVE
        # ===============================
        st.subheader("ROC Curve")

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax = plt.subplots()

        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

        st.pyplot(fig_roc)

        st.caption(f"ROC Curve with Differential Privacy (ε ≈ {epsilon:.2f})")

        # ===============================
        # PERFORMANCE METRICS
        # ===============================
        st.subheader("Model Performance Summary")

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        metrics = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1 Score", "ROC-AUC"],
            "Value": [precision, recall, f1, roc_auc]
        })

        st.table(metrics)

    # ===============================
    # SHOW PREDICTIONS TABLE
    # ===============================
    st.header("Flow Predictions")

    results = df_input.copy()
    results["Attack_Probability"] = probs
    results["Prediction"] = preds

    st.dataframe(results.head(20))