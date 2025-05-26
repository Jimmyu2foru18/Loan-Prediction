import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def load_data(file_path):
    """Load the loan prediction dataset."""
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Numerical columns
    numerical_imputer = SimpleImputer(strategy='mean')
    numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])
    
    # Categorical columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History']
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    
    return df

def encode_categorical_variables(df):
    """Encode categorical variables using LabelEncoder."""
    label_encoder = LabelEncoder()
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    
    return df

def preprocess_data(df):
    """Main preprocessing function."""
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical variables
    df = encode_categorical_variables(df)
    
    # Save Loan_ID separately as it's not used for modeling
    loan_ids = df['Loan_ID']
    df = df.drop('Loan_ID', axis=1)
    
    return df, loan_ids

def main():
    # Define paths using absolute paths
    input_file = os.path.join(PROJECT_ROOT, 'data', 'raw', 'test_Y3wMUE5_7gLdaTN.csv')
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading data from {input_file}")
    df = load_data(input_file)
    
    # Preprocess the data
    print("Preprocessing data...")
    processed_df, loan_ids = preprocess_data(df)
    
    # Save processed data
    output_file = os.path.join(output_dir, 'processed_loan_data.csv')
    processed_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()