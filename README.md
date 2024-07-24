# AdversCredit💰

## Introduction
The AdversCredit project investigates the performance of XGBoost and LSTM with attention mechanisms through a web interface. This project aims to implement adversarial attack strategies on a Card Fraud Detection model and note the performance of a normally trained model on such samples. 

However, other models trained on the Adversarial Samples are trying to showcase the ability of machine learning algorithms to perform robustly against adversarial datasets, in the age of GenAI and the rise of synthetic datasets and prompt abilities to empower fraud. Our findings reveal the performance of such models, shedding light on the existing frameworks against attacks.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/adverscredit.git
    cd adverscredit
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To run the Streamlit application, execute the following command:
```bash
streamlit run app.py
```

## Exploring the Code

### Training Models

If you want to check out the code for training the models, you can find the relevant Jupyter notebooks in the `code/models` directory. Each model has been saved with its respective name:

- [xgboost.ipynb](code/models/xgboost.ipynb)
- [xgboost_fgsa.ipynb](code/models/xgboost_fgsa.ipynb)
- [xgboost_pgd.ipynb](code/models/xgboost_pgd.ipynb)
- [lstm.ipynb](code/models/lstm.ipynb)
- [lstm_fgsa.ipynb](code/models/lstm_fgsa.ipynb)
- [lstm_pgd.ipynb](code/models/lstm_pgd.ipynb)

### Creating Adversarial Samples

There are two scripts available in the `preparing_data` directory for creating adversarial samples:

- [fgsa.ipynb](preparing_data/fgsa.ipynb): Script for creating adversarial samples using the FGSA method.
- [pgd.ipynb](preparing_data/pgd.ipynb): Script for creating adversarial samples using the PGD method.

You can explore these notebooks to create adversarial samples on your own dataset.

## Dataset

For the dataset, we have used the Credit Card Fraud Detection Dataset, which can be found [here](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023). The dataset contains over 550,000 samples with the classes divided exactly in half.

If you're interested in a detailed Pandas Profiling report for the original dataset, please refer to the dataset section in the Streamlit application.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.