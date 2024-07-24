import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from torch.utils.data import DataLoader, TensorDataset

# Check if GPU is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define model and attention classes
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, hidden_states):
        scores = torch.tanh(self.W(hidden_states))
        scores = torch.matmul(scores, self.v.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context

class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        out = self.fc(attn_out)
        return out

@st.cache_data
def load_model(model_path, input_dim, hidden_dim, output_dim, num_layers=1):
    model = LSTMWithAttention(input_dim, hidden_dim, output_dim, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
    
def evaluate_model(model, x_tensor, y_tensor, model_name):
    model.eval()
    with torch.no_grad():
        outputs = model(x_tensor)
        y_prob = torch.sigmoid(outputs.squeeze()).cpu().numpy()
        y_pred = (y_prob > 0.5).astype(int)

        # Compute metrics
        conf_matrix = confusion_matrix(y_tensor.cpu().numpy(), y_pred)
        class_report = classification_report(y_tensor.cpu().numpy(), y_pred)
        roc_auc = roc_auc_score(y_tensor.cpu().numpy(), y_prob)
        fpr, tpr, _ = roc_curve(y_tensor.cpu().numpy(), y_prob)

        return y_tensor.cpu().numpy(), y_pred, y_prob, roc_auc

def display_metrics(y_true, y_pred, y_prob, roc_auc, model_name):
    st.write(f"### {model_name} Performance Metrics")
    
    st.write("#### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.write(pd.DataFrame(cm, index=['Actual Neg', 'Actual Pos'], columns=['Predicted Neg', 'Predicted Pos']))
    
    st.write("#### Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())
    
    st.write("#### ROC AUC Score")
    st.write(f"ROC AUC Score: {roc_auc:.4f}")
    
    st.write("#### ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    st.pyplot(plt)
    
    st.write("#### Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    st.pyplot(plt)

# Streamlit application
st.title("LSTM With Attention🔮")

st.write("""
In the following inference and comparison UI, you can upload your own dataset and choose among the different models that have been finetuned using the adversarial samples.
""")
st.write("""
The model that has been trained on adversarial samples has been empowered with a Attention Module to make the classification more robust even when faced with Adversarial dataset. We have also used Regularization and Normalization to aid the training.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    target_col = 'Class'
    feature_cols = [col for col in data.columns if col != target_col]

    X = data[feature_cols].values
    y = data[target_col].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM: (num_samples, seq_length, num_features)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Convert numpy arrays to PyTorch tensors
    x_test_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    # Define model paths
    model_paths = {
        "Normal LSTM": "/Users/aryankargwal/adverscredit/code/weights/lstm.pth",
        "LSTM with FGSA Training": "/Users/aryankargwal/adverscredit/code/weights/lstm_fgsa.pth",
        "LSTM with PGD Training": "/Users/aryankargwal/adverscredit/code/weights/lst_pgd.pth"
    }

    # Preload the models
    models = {name: load_model(path, x_test_tensor.shape[2], 128, 1, 1) for name, path in model_paths.items()}

    st.write("### Model Comparison")

    model_options = list(models.keys())
    
    selected_model_1 = st.selectbox("Select Model 1", model_options, index=0)
    selected_model_2 = st.selectbox("Select Model 2", model_options, index=1)

    if selected_model_1 and selected_model_2:
        model_1 = models[selected_model_1]
        model_2 = models[selected_model_2]

        # Perform inference
        def predict_and_evaluate(model, x_tensor, y_tensor, model_name):
            y_true, y_pred, y_prob, roc_auc = evaluate_model(model, x_tensor, y_tensor, model_name)
            display_metrics(y_true, y_pred, y_prob, roc_auc, model_name)

        col1, col2 = st.columns(2)
        
        with col1:
            predict_and_evaluate(model_1, x_test_tensor, y_test_tensor, selected_model_1)

        with col2:
            predict_and_evaluate(model_2, x_test_tensor, y_test_tensor, selected_model_2)
