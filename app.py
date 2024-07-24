import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

def plot_roc_curve(y_true, y_pred_prob):
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {roc_auc:.2f})')
    st.pyplot(plt)

def main():
    st.title("Credit Card Fraud Detection Application")
    st.header("Introduction")
    st.write("""
        Welcome to the Credit Card Fraud Detection Application. This application allows you to:
        - Upload a CSV file with transaction data.
        - Run inference using a trained XGBoost model.
        - View the results including confusion matrix, ROC curve, and other relevant metrics.
    """)

if __name__ == "__main__":
    main()
