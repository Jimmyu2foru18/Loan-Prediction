import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix

def set_style():
    """Set the style for all visualizations."""
    plt.style.use('seaborn')
    sns.set_palette('husl')

def plot_feature_distributions(df, save_path=None):
    """Plot distribution of numerical features."""
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_categorical_counts(df, save_path=None):
    """Plot count plots for categorical variables."""
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(2, 3, i)
        sns.countplot(data=df, x=col)
        plt.title(f'Count of {col}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_correlation_matrix(df, save_path=None):
    """Plot correlation matrix of numerical features."""
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names, save_path=None):
    """Plot feature importance from the model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def generate_all_plots(df, model, y_true, y_pred, y_pred_proba, feature_names):
    """Generate and save all visualization plots."""
    output_dir = '../visualization/plots/'
    
    # Set the style for all plots
    set_style()
    
    # Generate all plots
    plot_feature_distributions(df, f'{output_dir}feature_distributions.png')
    plot_categorical_counts(df, f'{output_dir}categorical_counts.png')
    plot_correlation_matrix(df, f'{output_dir}correlation_matrix.png')
    plot_roc_curve(y_true, y_pred_proba, f'{output_dir}roc_curve.png')
    plot_confusion_matrix(y_true, y_pred, f'{output_dir}confusion_matrix.png')
    plot_feature_importance(model, feature_names, f'{output_dir}feature_importance.png')