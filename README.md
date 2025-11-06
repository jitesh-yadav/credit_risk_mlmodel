**Credit Risk Prediction Model**

1. Project Overview
This project is an end-to-end machine learning model designed to predict the likelihood of a loan applicant defaulting on their loan. By analyzing various personal and financial attributes, the model classifies applicants as high-risk (Default) or low-risk (Non-Default).
The primary business goal is to help financial institutions make more informed lending decisions. This model can be used to minimize financial losses from defaults while maximizing the approval of creditworthy applicants.
Final Model: XGBoost Classifier
Key Result: 93% Accuracy & 73% Recall (on defaults)

2. Dataset
The data was sourced from the Credit Risk Dataset on Kaggle.
It contains 32,581 loan applications with 12 features, including:
•	person_age
•	person_income
•	person_home_ownership
•	loan_intent
•	loan_grade
•	loan_amnt
•	loan_int_rate
•	loan_status (The target variable: 1 = Default, 0 = Non-Default)

3. Project Workflow
The project was completed following a standard data science workflow:
Step 1: Data Cleaning & Exploratory Data Analysis (EDA)
•	Identified Outliers: Found and removed illogical data points (e.g., person_age > 100, person_emp_length > 60).
•	Imputed Missing Values:
o	person_emp_length: Filled missing values using the dataset's median.
o	loan_int_rate: Used a "smart" imputation, filling missing rates with the median interest rate specific to that loan's loan_grade (e.g., all missing 'A' grade loans were filled with the median 'A' rate).
•	Identified Class Imbalance: The target variable loan_status was found to be highly imbalanced, with only 21.8% of the records being defaults (1s). This imbalance was a critical problem to solve.
Step 2: Feature Engineering & Preprocessing
•	Feature Creation: Engineered a new feature, loan_to_income_ratio, by dividing loan_amnt by person_income. This feature proved to be highly predictive.
•	One-Hot Encoding: Converted all categorical (text) columns (loan_intent, loan_grade, etc.) into numerical "dummy" variables that the model could understand.
Step 3: Handling Class Imbalance (SMOTE)
•	To prevent the model from just "guessing" the majority class (Non-Default), I applied the SMOTE (Synthetic Minority Over-sampling Technique).
•	Crucially, SMOTE was applied only to the training dataset to create a balanced 50/50 set for the model to learn from. The test set was left in its original, imbalanced state to provide a realistic evaluation.
Step 4: Modeling
•	The data was split 80% for training and 20% for testing.
•	An XGBoost Classifier was trained on the balanced (SMOTE) training data. XGBoost was chosen for its high performance and interpretability.

4. Key Results & Performance
The trained model was evaluated on the unseen, imbalanced test set:
Metric	Score	Description
Overall Accuracy	93%	The model correctly predicted the outcome 93% of the time.
Recall (for "Will Default")	73%	(Key Risk Metric) The model successfully "caught" 73% of all actual defaulters.
Precision (for "Will Default")	94%	When the model predicted a default, it was correct 94% of the time (low false positives).

5. Key Insights & Feature Importance
The XGBoost model provided clear insights into why it was making its decisions:
•	Top Predictors: The most important features for predicting default were loan_grade_D, person_home_ownership_RENT, and other high-risk loan grades.
•	Feature Engineering Success: Our custom-built feature, loan_to_income_ratio, was the #4 most important feature used by the model, confirming its high predictive value.

6. Technologies Used
•	Python 3
•	Google Colab (Development Environment)
•	Pandas: For data manipulation and cleaning.
•	Scikit-learn (sklearn): For data splitting (train_test_split) and evaluation metrics.
•	Imbalanced-learn (imblearn): For the SMOTE balancing technique.
•	XGBoost: For the classification model (XGBClassifier).
•	Matplotlib & Seaborn: For data visualization and plotting results.
<img width="468" height="627" alt="image" src="https://github.com/user-attachments/assets/afe93f1e-a035-4468-ac2e-6e0d146ca08b" />
