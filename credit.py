import streamlit as st
import pickle
import numpy as np
from sklearn.impute import SimpleImputer

model = pickle.load(open('credit_model8.pkl', 'rb'))

def run():
    st.title("Credit Card Default Prediction")
 
    # Load dataset
    column_names = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'Industry', 'Ethnicity',
                    'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore', 'DriversLicense',
                    'Citizen', 'ZipCode', 'Income', 'Approved']
    features = []

    # Prompt user for input
    for column in column_names:
        if column != 'Approved':
            feature = st.text_input(column)
            features.append(feature)

    # Prepare features for prediction
    features_cleaned = []
    for feature in features:
        try:
            cleaned_value = float(feature)
        except ValueError:
            cleaned_value = np.nan
        features_cleaned.append(cleaned_value)

    # Handle missing values using an imputer
    imputer = SimpleImputer(strategy='mean')
    features_cleaned_imputed = imputer.fit_transform([features_cleaned])


    print("Cleaned Features:", features_cleaned)
    features_cleaned_imputed = imputer.fit_transform([features_cleaned])
    print("Shape of features_cleaned_imputed:", features_cleaned_imputed.shape)


    # Make prediction
    prediction = model.predict(features_cleaned_imputed)

    if prediction == 0:
        st.error("This account is not eligible for a credit card.")
    else:
        st.success("This account is eligible for a credit card.")

run()
