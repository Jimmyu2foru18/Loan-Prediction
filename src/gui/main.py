import streamlit as st
import pandas as pd
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import handle_missing_values, encode_categorical_variables

def load_model():
    """Load the trained model."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    model_path = os.path.join(project_root, 'models', 'random_forest_model.joblib')
    return joblib.load(model_path)

def create_prediction_input():
    """Create input fields for loan prediction."""
    st.sidebar.header('Loan Application Details')
    
    # Personal Information
    st.sidebar.subheader('Personal Information')
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    married = st.sidebar.selectbox('Married', ['Yes', 'No'])
    dependents = st.sidebar.selectbox('Dependents', ['0', '1', '2', '3+'])
    education = st.sidebar.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.sidebar.selectbox('Self Employed', ['Yes', 'No'])
    
    # Financial Information
    st.sidebar.subheader('Financial Information')
    applicant_income = st.sidebar.number_input('Applicant Income', min_value=0)
    coapplicant_income = st.sidebar.number_input('Coapplicant Income', min_value=0)
    loan_amount = st.sidebar.number_input('Loan Amount (in thousands)', min_value=0)
    loan_term = st.sidebar.selectbox('Loan Term (months)', [360, 180, 480, 240, 120, 60, 300, 36, 84])
    credit_history = st.sidebar.selectbox('Credit History', [1, 0])
    property_area = st.sidebar.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])
    
    # Create a dictionary of inputs
    data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }
    
    return pd.DataFrame([data])

def predict_loan_status(model, input_df):
    """Make prediction using the trained model."""
    # Preprocess the input data
    input_df = handle_missing_values(input_df)
    input_df = encode_categorical_variables(input_df)
    
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    return prediction[0], prediction_proba[0]

def main():
    st.title('Loan Prediction System')
    st.write("""
    This application predicts the likelihood of loan approval based on applicant information.
    Please fill in the details in the sidebar to get a prediction.
    """)
    
    # Load the trained model
    try:
        model = load_model()
    except Exception as e:
        st.error("Error loading the model. Please make sure the model is trained and saved correctly.")
        return
    
    # Get input values
    input_df = create_prediction_input()
    
    # Add a prediction button
    if st.sidebar.button('Predict Loan Status'):
        # Make prediction
        prediction, prediction_proba = predict_loan_status(model, input_df)
        
        # Display results
        st.header('Prediction Results')
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Loan Status')
            if prediction == 1:
                st.success('✅ Loan Approved')
            else:
                st.error('❌ Loan Not Approved')
        
        with col2:
            st.subheader('Confidence Score')
            approval_probability = prediction_proba[1] * 100
            st.write(f'Probability of approval: {approval_probability:.2f}%')
        
        # Display feature importance or additional insights
        st.subheader('Important Factors')
        st.write("""
        Key factors that influence loan approval:
        - Credit History
        - Income Levels
        - Loan Amount to Income Ratio
        - Employment Status
        """)

if __name__ == '__main__':
    main()