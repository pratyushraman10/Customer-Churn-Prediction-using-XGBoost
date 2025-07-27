# Customer Churn Prediction using XGBoost

This project applies the **Telco Customer Churn** dataset to build a predictive model for identifying whether a customer will leave the service. The solution uses XGBoost in conjunction with pandas, NumPy, and scikit-learn.

---

## Dataset

**Source**: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset includes customer demographics, service usage patterns, billing information, and a binary target column indicating churn.

---

## Technologies Used

- Python (NumPy, pandas)
- scikit-learn for preprocessing, model evaluation, and hyperparameter tuning
- XGBoost for classification

---

## Features Used

- Gender, SeniorCitizen, Partner, Dependents
- Tenure, PhoneService, InternetService
- StreamingTV, Contract, PaperlessBilling
- PaymentMethod, MonthlyCharges, TotalCharges

**Target column**: `Churn` (encoded as 1 for Yes, 0 for No)

---

## Workflow

1. Load and clean the dataset
2. Encode binary and categorical variables
3. Handle missing values
4. Feature scaling using `StandardScaler`
5. Train-test split (80-20)
6. Train XGBoost model
7. Perform hyperparameter tuning using `GridSearchCV`
8. Evaluate model with:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

---

## Results

- **Accuracy**: ~84%
- Model shows strong performance in identifying customer churn
- GridSearchCV used for optimal parameter selection

---

## File Structure

- `churn_prediction.py`: Complete training and evaluation code
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Dataset file
- `README.md`: Project overview

---

This project was created for educational and professional demonstration in customer analytics and machine learning.
