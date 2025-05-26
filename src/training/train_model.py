import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os

def load_processed_data(file_path):
    """Load the preprocessed dataset."""
    return pd.read_csv(file_path)

def prepare_features_target(df):
    """Prepare features for training by adding a mock target variable."""
    # Since we're working with test data, we'll create a mock target variable
    # This is for demonstration purposes only
    np.random.seed(42)
    df['Loan_Status'] = np.random.choice(['Y', 'N'], size=len(df), p=[0.7, 0.3])
    
    # Prepare features
    X = df.drop(['Loan_Status'], axis=1) if 'Loan_Status' in df.columns else df
    y = pd.Series(df['Loan_Status'].map({'Y': 1, 'N': 0}))
    return X, y

def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print evaluation metrics
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, roc_auc

def perform_cross_validation(model, X, y):
    """Perform k-fold cross-validation."""
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("\nCross-validation Scores:")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

def save_model(model, file_path):
    """Save the trained model to disk."""
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"\nModel saved to {file_path}")

def main():
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Load preprocessed data
    data_path = os.path.join(project_root, 'data', 'processed', 'processed_loan_data.csv')
    print(f"Loading data from {data_path}")
    df = load_processed_data(data_path)
    
    # Prepare features and target
    X, y = prepare_features_target(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    print("Training Random Forest model...")
    model = train_random_forest(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Perform cross-validation
    perform_cross_validation(model, X, y)
    
    # Save the model
    model_path = os.path.join(project_root, 'models', 'random_forest_model.joblib')
    save_model(model, model_path)

if __name__ == "__main__":
    main()