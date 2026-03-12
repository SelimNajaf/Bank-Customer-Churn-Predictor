"""
Bank Customer Churn Predictor
This script processes bank customer data, applies custom feature engineering 
and preprocessing pipelines, evaluates multiple algorithms handling class 
imbalance, and exports the best performing model.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ==========================================
# 0. SYSTEM PATH CONFIGURATION
# ==========================================
# Dynamically add the project root to sys.path so the 'src' package 
# can be discovered regardless of where the script is executed from.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import FeatureEngineer
from src.preprosessor_pipeline import Preprocessor

# ==========================================
# 1. DATA LOADING & EXPLORATION
# ==========================================
FILE_PATH = 'dataset/Bank_Churn.csv'

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: Dataset '{FILE_PATH}' not found. Please ensure it is in the correct directory.")
    sys.exit(1)

print("--- DataFrame Head ---")
print(df.head())

# Drop identifiable columns that hold no predictive value
columns_to_drop = ['CustomerId', 'Surname']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Display basic data quality metrics
print("\n--- Data Quality Check ---")
print(f"Maximum Null Values in any column: {df.isnull().sum().max()}")

# Display the class imbalance ratio
churn_percentage = df['Exited'].value_counts(normalize=True) * 100
print(f"\n--- Class Distribution (Churn Percentage) ---\n{churn_percentage.to_string()}\n")

print('\n--- DataFrame Information ---')
df.info()

# ==========================================
# 2. DATA SPLITTING
# ==========================================
X = df.drop(columns=['Exited'])
y = df['Exited']

# Stratified split ensures the train and test sets have the same proportion of churn cases
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ==========================================
# 3. MODEL CONFIGURATION
# ==========================================
# Calculate ratio for XGBoost to handle class imbalance dynamically
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced", 
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_weight,
        random_state=42
    )
}

# Hyperparameter grids for each respective model
params = {
    "LogisticRegression": {
        "model__C":[0.01, 0.1, 1, 10],
        "model__penalty": ["l2"]
    },
    "RandomForest": {
        "model__n_estimators":[100, 200],
        "model__max_depth": [5, 10, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf":[1, 2]
    },
    "XGBoost": {
        "model__n_estimators": [100, 200],
        "model__max_depth":[3, 5],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1]
    }
}

# ==========================================
# 4. TRAINING & EVALUATION LOOP
# ==========================================
best_score = 0
best_model_name = None
trained_model = None

print("\nStarting Model Training & Hyperparameter Tuning...")

for name, model in models.items():
    print(f'\n{"="*40}')
    print(f'--- Training: {name} ---')
    print(f'{"="*40}')

    # Construct the end-to-end pipeline using custom transformers
    pipeline = Pipeline(
        steps=[
            ('feature_engineer', FeatureEngineer()),
            ('preprocessor', Preprocessor()),
            ('model', model)
        ]
    )

    # Initialize GridSearchCV optimizing for ROC-AUC
    grid_search = GridSearchCV(
        pipeline,
        params[name],
        scoring='roc_auc',
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    current_best_estimator = grid_search.best_estimator_

    # Generate predictions
    y_pred = current_best_estimator.predict(X_test)
    y_prob = current_best_estimator.predict_proba(X_test)[:, 1]

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_prob) 
    f1_s = f1_score(y_test, y_pred)
    cl_report = classification_report(y_test, y_pred)

    print(f'Score (ROC-AUC): {roc_auc:.4f}') 
    print(f'Score (F1): {f1_s:.2f}')
    print(f'Classification Report:\n{cl_report}')

    # Track the highest performing model based on ROC-AUC
    if roc_auc > best_score:
        best_score = roc_auc
        trained_model = current_best_estimator
        best_model_name = name

print(f'\nThe Winning Model: {best_model_name}')
print(f'Best ROC-AUC Score: {best_score:.4f}')


# ==========================================
# 5. CUSTOM BUSINESS THRESHOLD
# ==========================================
# Company Decision: Willing to accept a Precision of ~0.60 to achieve higher Recall.
# Adjusting the prediction threshold down to 0.30 to capture more potential churners.
print("\n" + "="*50)
print(f" BUSINESS EVALUATION: {best_model_name} (Threshold = 0.30)")
print("="*50)

prob = trained_model.predict_proba(X_test)[:, 1]
prob_custom_threshold = (prob > 0.30).astype(int)

print(classification_report(y_test, prob_custom_threshold))
print(f"Final ROC-AUC Score: {roc_auc_score(y_test, prob):.4f}")


# ==========================================
# 6. EXPORTING THE PIPELINE
# ==========================================
# Ensure the directory exists before attempting to save
SAVE_DIR = 'trained_model'
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, 'trained_model.joblib')
joblib.dump(trained_model, MODEL_PATH)

print(f"\nSuccess! Pipeline saved to '{MODEL_PATH}'")