
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Classification App", layout="centered")

st.title("Breast Cancer Classification System")
st.write("Upload test dataset to evaluate trained ML models.")

model_option = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

uploaded_file = st.file_uploader("Upload Test CSV (must include target column)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    if 'target' not in data.columns:
        st.error("Dataset must contain 'target' column.")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]
        
        model = joblib.load(f"model/{model_option}.pkl")
        scaler = joblib.load("model/scaler.pkl")
        
        if model_option in ["Logistic Regression", "KNN"]:
            X = scaler.transform(X)
        
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:,1]
        
        st.subheader("Evaluation Metrics")
        st.write("Accuracy:", accuracy_score(y, y_pred))
        st.write("AUC Score:", roc_auc_score(y, y_prob))
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d")
        st.pyplot(fig)
        
        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))
