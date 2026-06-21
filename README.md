# Telco Customer Churn Prediction Engine

An end-to-end data science workspace evaluating multiple machine learning classification paradigms to forecast subscriber churn. This repository steps through standard feature preparation pipelines and directly benchmarks parametric baselines against deep decision trees and stabilized Random Forest ensembles to capture customer friction signatures before subscription drop-off.

## Dataset Description
The model utilizes the standard `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset, evaluating 7,043 distinct subscriber records across 21 demographic, financial, and product parameters:
* **Target Variable:** `Churn` (Binary outcome: `Yes` or `No`).
* **Demographics & Account Metadata:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, and unique account properties.
* **Core Product Profiles:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, and `StreamingMovies`.
* **Financial Metrics:** `Contract` type, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, and `TotalCharges`.

## Methodology
1. **Data Ingestion & Cleaning:** Loaded structural dependencies into Pandas dataframes, dropping arbitrary account keys (`customerID`) to eliminate localized feature drift.
2. **Feature Encoding:** Applied Scikit-learn's `LabelEncoder` across categorical multi-class strings to normalize text-based matrices into machine-readable continuous vectors.
3. **Data Splitting:** Divided structured records into an 80% training matrix and a 20% test split, anchoring reproducible validation checks under a fixed random state (`random_state=42`).
4. **Algorithmic Benchmarking:** Built and optimized three separate predictive classification models:
   * **Logistic Regression:** Fitted as a baseline linear parametric optimizer.
   * **Decision Tree Classifier:** Evaluated as an unconstrained branch boundary model.
   * **Random Forest Classifier:** Configured as a 100-tree voting ensemble with controlled tree depth boundaries (`max_depth=7`).

## Results & Comparative Evaluation
Model optimization checks across validation data yielded the following key performance fields:

* **Logistic Regression:** Delivered robust generalization capacity on unseen test structures, securing a peak Test Accuracy of **`81.41%`** and a balanced Test F1-Score of **`61.70%`**.
* **Decision Tree Classifier:** Exhibited classic unconstrained overfitting behavior, memorizing training fields to hit a Train Accuracy of `99.85%` but dropping significantly to a Test Accuracy of **`72.75%`** (Test F1: `46.96%`) due to structural high-variance boundaries.
* **Random Forest Classifier (Ensemble):** Corrected single-tree variance drift effectively. By limiting maximum node splits, the ensemble successfully reached a stabilized Test Accuracy of **`80.84%`** alongside a Test F1-Score of **`58.07%`**.

## How to Run Locally
1. Clone this repository to your machine workspace setup.
2. Install standard missing package dependencies via pip:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
