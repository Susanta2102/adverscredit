import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

def main():
    st.title("AdversCredit💰")
    st.header("Introduction")
    st.write("""
        The project investigates the performances of XGBoost and LSTM with attention mechanisms through a web interface. This project aims to implement adversarial attack strategies on a Card Fraud Detection model and note the performance of a normally trained model on such samples.
    """)
    st.write("""
        However, other models trained on the Adversarial Samples are trying to showcase the ability of machine learning algorithms to perform robustly against adversarial datasets,
        and in the age of GenAI and the rise of synthetic datasets and prompt abilities to empower fraud. Our findings reveal the performance of such models, shedding light on the existing frameworks against attacks.
    """)
    
    st.subheader("Dataset")
    st.markdown("""
        For the dataset we have used the Credit Card Fraud Detection Dataset link to which can be found [here](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023),
        which comes containing over 550,000 samples with the classes divided exactly in half.
    """)

if __name__ == "__main__":
    main()
