import sys
import os
import pandas as pd
import numpy as np
import pytest

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.preprocess import (
    handle_missing_values,
    encode_categorical_variables,
    preprocess_data
)

@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    data = {
        'Loan_ID': ['LP001', 'LP002', 'LP003'],
        'Gender': ['Male', 'Female', np.nan],
        'Married': ['Yes', 'No', 'Yes'],
        'Dependents': ['0', '1', np.nan],
        'Education': ['Graduate', 'Not Graduate', 'Graduate'],
        'Self_Employed': ['No', np.nan, 'Yes'],
        'ApplicantIncome': [5000, 3000, np.nan],
        'CoapplicantIncome': [0, 2000, 1500],
        'LoanAmount': [100, np.nan, 150],
        'Loan_Amount_Term': [360, 180, np.nan],
        'Credit_History': [1, np.nan, 0],
        'Property_Area': ['Urban', 'Rural', 'Semiurban']
    }
    return pd.DataFrame(data)

def test_handle_missing_values(sample_data):
    """Test if missing values are handled correctly."""
    df = handle_missing_values(sample_data)
    
    # Check if there are any missing values after processing
    assert df.isnull().sum().sum() == 0, "Missing values still exist after handling"
    
    # Check if numerical values are reasonable
    assert df['ApplicantIncome'].mean() > 0, "Invalid mean for ApplicantIncome"
    assert df['LoanAmount'].mean() > 0, "Invalid mean for LoanAmount"

def test_encode_categorical_variables(sample_data):
    """Test if categorical variables are encoded correctly."""
    # First handle missing values to avoid encoding errors
    df = handle_missing_values(sample_data)
    df = encode_categorical_variables(df)
    
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 
                          'Self_Employed', 'Property_Area']
    
    # Check if categorical columns are encoded as numbers
    for col in categorical_columns:
        assert df[col].dtype in ['int32', 'int64'], f"{col} is not properly encoded"

def test_preprocess_data(sample_data):
    """Test the complete preprocessing pipeline."""
    df, loan_ids = preprocess_data(sample_data)
    
    # Check if Loan_ID is removed from features
    assert 'Loan_ID' not in df.columns, "Loan_ID should be removed from features"
    
    # Check if loan_ids are preserved correctly
    assert len(loan_ids) == len(sample_data), "Loan_IDs not preserved correctly"
    
    # Check if all features are numeric
    assert all(df.dtypes != 'object'), "Some columns are not numeric after preprocessing"
    
    # Check if there are no missing values
    assert df.isnull().sum().sum() == 0, "Missing values in preprocessed data"

def test_data_shape_consistency(sample_data):
    """Test if data shape is maintained after preprocessing."""
    df, _ = preprocess_data(sample_data)
    
    # Check if number of rows is maintained
    assert len(df) == len(sample_data), "Number of rows changed during preprocessing"
    
    # Check if number of columns is correct (original - Loan_ID)
    assert len(df.columns) == len(sample_data.columns) - 1, "Incorrect number of columns after preprocessing"

if __name__ == '__main__':
    pytest.main([__file__])