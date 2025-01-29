import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import random

# Sidebar image
st.sidebar.image('plots/payment_fraud.jpg', use_container_width=True)

# Load the trained models
svm = joblib.load("weights/linear_svm_model.pkl")
logistic_regression_model = joblib.load("weights/logistic_regression_model.pkl")
xgboost_model = joblib.load("weights/xgboost_model.pkl")

# Intro Page
def intro():
    st.title('Online Payments Fraud Detection')
    st.write(
        'Welcome to the Online Payments Fraud Detection app! This app showcases machine learning models '
        'for detecting fraud in online payment transactions.'
    )
    st.markdown(
        """
        ### Dataset Columns:
        - **step**: Represents a unit of time where 1 step equals 1 hour.
        - **type**: Type of online transaction.
        - **amount**: The amount of the transaction.
        - **nameOrig**: Customer starting the transaction.
        - **oldbalanceOrg**: Balance before the transaction.
        - **newbalanceOrig**: Balance after the transaction.
        - **nameDest**: Recipient of the transaction.
        - **oldbalanceDest**: Initial balance of recipient before the transaction.
        - **newbalanceDest**: The new balance of recipient after the transaction.
        - **isFlaggedFraud**: Transaction marked potentially fraudulent by the system.
        - **isFraud**: Indicates whether the transaction is fraudulent (1) or not (0).
        """
    )

# Generate random inference data
def generate_inference_data():
    step_choices = [1, 743, 204500]
    amount_min, amount_max = 0, 92445520
    balance_min, balance_max = 0, 59585040

    inference_data = {
        'step': random.choice(step_choices),
        'amount': round(random.uniform(amount_min, amount_max), 2),
        'oldbalanceOrg': round(random.uniform(balance_min, balance_max), 2),
        'newbalanceOrig': round(random.uniform(balance_min, balance_max), 2),
        'oldbalanceDest': round(random.uniform(balance_min, balance_max), 2),
        'newbalanceDest': round(random.uniform(balance_min, balance_max), 2),
        'isFlaggedFraud': random.choice([0, 1]),
        'type_CASH_IN': 0,
        'type_CASH_OUT': 0,
        'type_DEBIT': 0,
        'type_PAYMENT': 0,
        'type_TRANSFER': 0
    }

    # Randomly select a type
    selected_type = random.choice(['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'])
    inference_data[selected_type] = 1

    return pd.DataFrame([inference_data])

# Predict Fraud
def predict(model, data):
    prediction = model.predict(data)[0]
    result = "Fraud" if prediction == 1 else "Legitimate"
    color = "red" if prediction == 1 else "green"
    return f"<span style='color:{color}; font-weight:bold;'>{result}</span>"

# Inference Page
def inference():
    st.title('Run Inference')
    st.write('Generate a random transaction and check if it is fraudulent using different models.')

    model_selection = st.selectbox(
        '**Select Model**', 
        ['Logistic Regression', 'XGBoost', 'Support Vector Machine (SVM)']
    )

    if st.button('Generate and Run'):
        model = None
        if model_selection == 'Support Vector Machine (SVM)':
            model = svm
        elif model_selection == 'XGBoost':
            model = xgboost_model
        elif model_selection == 'Logistic Regression':
            model = logistic_regression_model

        if model is not None:
            inference_data = generate_inference_data()
            st.write('Generated Inference Data:')
            st.dataframe(inference_data)
            
            prediction = predict(model, inference_data)
            st.markdown(f"### **Prediction**: {prediction}", unsafe_allow_html=True)

# Data Visualization Page
def data_plots():
    st.title('Data Visualization')

    plot_selection = st.selectbox(
        'Select a plot:',
        ['Correlation Matrix', 'Fraud vs. Flagged Fraud', 'Fraudulent Transactions by Type', 'Transaction Types']
    )

    if plot_selection == 'Correlation Matrix':
        st.image('plots/confusion_matrix.png', use_container_width=True)
    elif plot_selection == 'Fraud vs. Flagged Fraud':
        st.image('plots/fraud_flagged_counts.png', use_container_width=True)
    elif plot_selection == 'Fraudulent Transactions by Type':
        st.image('plots/fraud_transactiontype.png', use_container_width=True)
    elif plot_selection == 'Transaction Types':
        st.image('plots/transaction_types.png', use_container_width=True)

# Main App
def main():
    st.sidebar.title('Navigation')
    pages = {
        "Home": intro,
        "Models": inference,
        "Data Visualization": data_plots
    }

    selection = st.sidebar.radio("Go to:", list(pages.keys()))
    page = pages[selection]
    page()

if __name__ == "__main__":
    main()

