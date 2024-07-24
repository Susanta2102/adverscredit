import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

# Paths to the pre-trained models
model_paths = {
    "Normal XGBoost": "/Users/aryankargwal/adverscredit/code/weights/xgboost.pkl",
    "XGBoost with FGSA Training": "/Users/aryankargwal/adverscredit/code/weights/xgboost_fgsa.pkl",
    "XGBoost with PGD Training": "/Users/aryankargwal/adverscredit/code/weights/xgboost_pgd.pkl"
}

# Function to load model from pkl file
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Preload the models
models = {name: load_model(path) for name, path in model_paths.items()}

# Function to display performance metrics
def display_metrics(y_true, y_pred, y_prob, model_name):
    st.write(f"### {model_name} Performance Metrics")
    
    st.write("#### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.write(pd.DataFrame(cm, index=['Actual Neg', 'Actual Pos'], columns=['Predicted Neg', 'Predicted Pos']))
    
    st.write("#### Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())
    
    st.write("#### ROC AUC Score")
    roc_auc = roc_auc_score(y_true, y_prob)
    st.write(f"ROC AUC Score: {roc_auc:.4f}")
    
    st.write("#### ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    st.pyplot(plt)
    
    st.write("#### Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    st.pyplot(plt)

# Streamlit application
st.title("XGBoost with Grid Search🌲")

st.write("""
In the following inference and comparison UI, you can upload your own dataset and choose among the different models that have been finetuned using the adversarial samples.
""")
st.write("""
The models have been trained on a simple implmentation of XGBoost with Grid Search for Optimization to make the classification more robust even when faced with Adversarial dataset. We have also used Regularization and Normalization to aid the training.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Assuming the target column is known
    target_col = 'Class'
    feature_cols = [col for col in data.columns if col != target_col]

    X = data[feature_cols].values
    y = data[target_col].values

    st.write("### Model Comparison")

    model_options = list(models.keys())
    
    selected_model_1 = st.selectbox("Select Model 1", model_options, index=0)
    selected_model_2 = st.selectbox("Select Model 2", model_options, index=1)

    if selected_model_1 and selected_model_2:
        model_1 = models[selected_model_1]
        model_2 = models[selected_model_2]

        y_pred_1 = model_1.predict(X)
        y_prob_1 = model_1.predict_proba(X)[:, 1]
        
        y_pred_2 = model_2.predict(X)
        y_prob_2 = model_2.predict_proba(X)[:, 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            display_metrics(y, y_pred_1, y_prob_1, selected_model_1)
        
        with col2:
            display_metrics(y, y_pred_2, y_prob_2, selected_model_2)