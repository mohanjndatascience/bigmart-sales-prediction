# BigMart Sales Prediction

## Problem Statement

Predict sales for BigMart outlets using historical 2013 sales data for 1559 products across 10 stores in different cities. Each product and store has defined attributes. The goal is to build a predictive model to estimate product sales per outlet.

---

## Project Structure

```
bigmart-sales-prediction/
├── Code/
│   ├── Notebooks/                 # Jupyter notebooks for exploration and EDA
│   └── Scripts/                   # Scripts for preprocessing, training, and predictions
├── Data/
│   ├── Input_Data/                # Raw train/test CSVs
│   ├── Preprocessed_Data/         # Preprocessed train/test CSVs
│   ├── Models/                    # Trained model files (.joblib)
│   ├── Output_Data/               # Final predictions/output files
│   └── Validation_Results/        # Model validation/metrics CSVs
├── Documents/                     # Reports, README, or reference docs
├── src/
│   ├── config/
│   │   └── config.py              # Configuration variables and paths
│   └── utils/
│       └── packages.py            # Preprocessing and modeling functions
└── requirements.txt               # Project dependencies
```

---

## Features

- **Data Preprocessing**
  - Handle missing values
  - Standardize categorical variables
  - Create combined features
  - One-hot encoding for categorical variables
- **Model Training & Validation**
  - Linear Regression, Ridge, Lasso
  - Random Forest, Gradient Boosting
  - Support Vector Regressor (SVR), XGBoost
  - Hyperparameter tuning with GridSearchCV
  - Evaluation metrics: MAE, MSE, RMSE, R², Adjusted R²
- **Test Prediction**
  - Generate predictions for unseen data
  - Aggregate top-N model predictions
  - Store results in CSV

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/mohanjndatascience/bigmart-sales-prediction.git
cd bigmart-sales-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. **Run Preprocessing**

```bash
python Code/Scripts/run_preprocessing.py
```

2. **Train & Validate Models**

```bash
python Code/Scripts/train_models.py
```

3. **Generate Test Predictions**

```bash
python Code/Scripts/predict_test.py
```

4. **Optional: Run Full Pipeline**

```bash
python Code/Scripts/pipeline.py
```

---

## Outputs

- Preprocessed datasets: `Data/Preprocessed_Data/`
- Trained models: `Data/Models/`
- Validation metrics: `Data/Validation_Results/`
- Test predictions: `Data/Output_Data/`

---

## Author

**Mohan J N**


