# Heart Failure Clinical Records Prediction

## Project Overview
A machine learning classification pipeline designed to predict patient survival outcomes (`DEATH_EVENT`) based on clinical records and physiological measurements, supporting early risk assessment in medical environments.

## Dataset
- **Source:** Kaggle - Heart Failure Clinical Records Dataset
- **Dataset Link:** [https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data]

## Pipeline Architecture & Methodology
1. **Data Exploration & Preparation:**
   - Verified dataset integrity (zero missing values across 299 clinical records).
   - Retained critical medical anomalies and high-value biomarkers (such as creatinine phosphokinase and platelets) to preserve vital diagnostic signals.
2. **Feature Scaling & Splitting:**
   - Applied `StandardScaler` to normalize numerical features with varying scales.
   - Performed stratified train-test splitting to maintain target class distribution balance.
3. **Validation Strategy:**
   - Implemented K-Fold Cross-Validation (5-Folds) to ensure model stability and prevent evaluation bias caused by small dataset size constraints.
   - Trained classification models including Logistic Regression, Random Forest, and XGBoost.

## Results & Performance
- **Best Performing Model:** Random Forest and Logistic Regression
- **Key Metrics:** Achieved robust cross-validation accuracy and high ROC-AUC scores, validating the model's reliability on unseen medical data.

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
