# Loan Prediction System

## Overview
An automated loan prediction system that uses machine learning to predict loan approval probability. This project implements a binary classification model with a user-friendly GUI for real-time predictions.

## Features
- Data preprocessing and analysis
- Machine learning model for loan approval prediction
- Interactive GUI for user input
- Visualization of prediction results
- Model performance metrics

## Project Structure
```
loan_prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── preprocessing/ 
│   ├── training/   
│   ├── visualization/
│   └── gui/
├── tests/
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jimmyu2foru18/loan-prediction.git
cd loan-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
```bash
python src/preprocessing/preprocess.py
```

2. Train Model:
```bash
python src/training/train_model.py
```

3. Launch GUI:
```bash
python src/gui/main.py
```
---
