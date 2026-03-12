# 🏦 Bank Customer Churn Predictor & API

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-orange?style=for-the-badge&logo=xgboost)
![Domain](https://img.shields.io/badge/Domain-FinTech_%7C_Banking-success?style=for-the-badge)

## 📖 Project Overview
The **Bank Customer Churn Predictor** is a production-ready machine learning system designed to identify retail banking customers at high risk of leaving. Customer acquisition costs are significantly higher than retention costs, making proactive churn prediction a critical asset for modern financial institutions.

This project showcases an advanced, **Object-Oriented Data Science workflow**. Instead of relying on messy scripts, it utilizes completely custom Scikit-Learn Transformers (`BaseEstimator`, `TransformerMixin`) to encapsulate complex feature engineering and dynamic preprocessing. The winning **XGBoost** model is then optimized via `GridSearchCV`, tuned for specific business objectives, and deployed via a high-performance **FastAPI** REST endpoint.

## ✨ Key Features
*   **Custom Scikit-Learn Transformers:** Modular, reusable `FeatureEngineer` and `Preprocessor` classes ensure that exact same transformations are applied during training and real-time API inference, completely eliminating data leakage.
*   **Behavioral Feature Engineering:** Dynamically derives intelligent features such as `loyalty_score`, `salary_balance_ratio`, `balance_per_product`, and dynamic `age_bucket`s directly within the pipeline.
*   **Algorithmic Imbalance Handling:** Automatically calculates the churn ratio (80/20 split) and injects it into XGBoost's `scale_pos_weight` to aggressively penalize the model for missing actual churners.
*   **Business-Driven Threshold Tuning:** Instead of settling for the default 0.50 probability cutoff, the pipeline automatically evaluates a custom **0.30 business threshold**. This prioritizes **Recall**, ensuring the bank catches 90% of potential churners for intervention campaigns.
*   **FastAPI Microservice:** Exposes the `.joblib` pipeline via an asynchronous REST API, utilizing `Pydantic` to strictly validate incoming JSON customer profiles.

## 📊 Data Description
The model is trained on a localized financial dataset containing 10,000 customer records. 

**Target Variable:** 
*   `Exited` (1 = Churned[20.37%], 0 = Stayed [79.63%])

**Input Features:**
*   **Demographics:** `Age`, `Geography`, `Gender`
*   **Account Metrics:** `CreditScore`, `Tenure`, `Balance`, `EstimatedSalary`
*   **Engagement:** `NumOfProducts`, `HasCrCard`, `IsActiveMember`

## 🛠️ Project Architecture

```text
├── src/
│   ├── feature_engineering.py               # Custom transformer (Behavioral metric generation)
│   └── preprocessor_pipeline.py             # Custom transformer (Dynamic scaling & encoding)
├── dataset/
│   └── Bank_Churn_Data_Dictionary.csv       # Raw dataset [Not included, download required]
├── trained_model/
│   └── trained_model.joblib                 # Serialized end-to-end XGBoost pipeline (Output)
├── train/
│   └── train_model.py                       # GridSearch training, threshold tuning, and evaluation
├── api/
│   └── main.py                              # FastAPI application and prediction endpoint
└── README.md                                # Project documentation
```

## 🚀 Installation & Prerequisites

To run this pipeline and API locally, ensure you have Python 3.8+ installed.

1. **Clone the repository:**
   ```bash
   git clone [Insert Repository Link Here]
   cd [Insert Repository Directory Name]
   ```

2. **Install the required dependencies:**
   It is highly recommended to use a virtual Python environment.
   ```bash
   pip install pandas numpy scikit-learn xgboost fastapi uvicorn joblib pydantic
   ```

3. **Add the Dataset:**
   Ensure the `Bank_Churn.csv` file is placed inside the `dataset/` directory.

## 💻 Usage / How to Run

### Step 1: Train the Model & Export Pipeline
Execute the training script. This will trigger the custom transformers, run the Grid Search across Logistic Regression, Random Forest, and XGBoost, and export the winning pipeline.

```bash
python train_model.py
```

### Step 2: Launch the FastAPI Server
Once `trained_model.joblib` is generated in the `trained_model/` folder, spin up the API using Uvicorn.

```bash
uvicorn main:app --reload
```

### Step 3: Test the API Endpoint
Navigate to `http://127.0.0.1:8000/docs` to use the interactive Swagger UI, or simulate a live customer evaluation via cURL:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0.0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88
}'
```

## 📈 Results & Business Evaluation

During cross-validation, the algorithms were optimized for **ROC-AUC** to measure their ability to successfully separate churners from loyal customers. **XGBoost** emerged as the winning algorithm with a top ROC-AUC score of **0.8712**.

### 📉 The Business Threshold (0.30)
In standard machine learning, the default probability threshold is `0.50`. However, in churn prediction, missing a churner (False Negative) is far more expensive than accidentally sending a retention email to a loyal customer (False Positive). 

By lowering the decision threshold to `0.30`, the model's **Recall jumps to 90%** for the minority class.
*   **What this means:** The bank successfully identifies 90% of all customers who are about to leave, allowing the marketing team to launch highly effective, targeted retention campaigns (e.g., offering waived fees or premium support).

**Example API Response:**
```json
{
  "prediction": 1,
  "probability": 0.6845,
  "status": "Churn"
}
```

## 🤝 Contributing
Contributions are highly encouraged! To further optimize this project:
1. Fork the repository
2. Create your Feature Branch (`git checkout -b feature/AddSHAPExplanations`)
3. Commit your Changes (`git commit -m 'Add SHAP values to API response for explainability'`)
4. Push to the Branch (`git push origin feature/AddSHAPExplanations`)
5. Open a Pull Request

## 📜 License
This project is open-source and available under the MIT License. See `LICENSE` for more information.

## 📬 Contact
**Selim Najaf**

*   **LinkedIn:** [linkedin.com/in/selimnajaf](https://www.linkedin.com/in/selimnajaf/)
*   **GitHub:** [github.com/SelimNajaf](https://github.com/SelimNajaf)

*Developed as a continuous learning initiative in advanced Data Science and ML Engineering.*
