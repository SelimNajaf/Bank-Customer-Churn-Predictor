# 🏦 Bank Customer Churn Predictor

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-FinTech_%7C_Banking-success?style=for-the-badge)

## 📖 Project Overview
Customer retention is significantly cheaper than customer acquisition. The **Bank Customer Churn Predictor** is a machine learning pipeline designed to identify retail banking customers who are at a high risk of leaving (churning). 

This project tackles common real-world data science challenges: handling highly imbalanced datasets, engineering business-logic features from raw data, and tuning classification thresholds to balance the trade-off between identifying churners (Recall) and minimizing false alarms (Precision). By leveraging a unified `scikit-learn` pipeline and Logistic Regression, this model provides a highly interpretable and production-ready solution.

## ✨ Key Features
*   **Strategic Feature Engineering:** Programmatically generates new predictive features such as `tenure_to_age` (loyalty relative to age) and `balance_salary_ratio` (financial utilization) to give the model deeper behavioral context.
*   **Imbalanced Class Handling:** Utilizes `class_weight='balanced'` within the Logistic Regression model to penalize the algorithm heavier when it misclassifies the minority class (churners).
*   **Unified ML Pipeline:** Packages `StandardScaler` (for numeric features), `OneHotEncoder` (for categorical features), and the predictive model into a single `Pipeline` object, ensuring robust cross-validation and preventing data leakage.
*   **Custom Threshold Tuning:** Adjusts the default probability decision boundary (from `0.50` to `0.65`) to reduce "False Positives" (predicting a loyal customer will churn), optimizing the model for business intervention campaigns.

## 📊 Data Description
The model is trained on a localized financial dataset (`Bank_Churn.csv`) detailing customer demographics and account behaviors. Personally Identifiable Information (PII) like `CustomerId` and `Surname` are dropped prior to modeling.
*🔗 **Dataset Link:** [Insert Dataset Link Here]*

**Input Features:**
*   **Demographics:** `Age`, `Geography` (Country), `Gender`
*   **Account History:** `CreditScore`, `Tenure`, `EstimatedSalary`
*   **Product Usage:** `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`
*   **Engineered Features:** `tenure_to_age`, `balance_salary_ratio`

**Target Variable:**
*   `Exited`: Binary indicator (1 = Customer Churned, 0 = Customer Retained)

## 🛠️ Project Architecture

```text
├── churn_predictor.py                         # Main ML script: feature engineering, training, and evaluation
├── Bank_Churn.csv                             # Raw customer dataset[Not included, download required]
├── Bank_Churn_Data_Dictionary.csv             # Description of Columns
└── README.md                                  # Project documentation
```

## 🚀 Installation & Prerequisites

To run this project locally, ensure you have Python 3.8+ installed. 

1. **Clone the repository:**
   ```bash
   git clone [Insert Repository Link Here]
   cd[Insert Repository Directory Name]
   ```

2. **Install the required dependencies:**
   It is highly recommended to use a virtual Python environment.
   ```bash
   pip install pandas scikit-learn
   ```

3. **Add the Dataset:**
   Ensure the `Bank_Churn.csv` file is downloaded and placed in the root directory of the project before running the training script.

## 💻 Usage / How to Run

Execute the main Python script. The script will automatically load the data, apply the custom feature engineering function independently to the train and test sets, fit the preprocessing pipeline, and output the evaluation metrics.

```bash
python churn_predictor.py
```

## 📈 Results / Model Evaluation

The model achieved an overall **ROC-AUC Score of 0.7754**, indicating strong discriminative ability between churners and retained customers. 

Because the dataset is imbalanced, accuracy is not a reliable metric. Instead, the project evaluates performance across two different probability thresholds to align with business objectives:

### Evaluation 1: Standard Threshold (0.50)
*   **Recall (Churn):** `71%` (Successfully identifies the majority of actual churners)
*   **Precision (Churn):** `39%` (High rate of false alarms)
*   *Business Impact:* Best used if the retention campaign is very cheap (e.g., sending an automated email). We catch almost everyone, but we annoy many loyal customers in the process.

### Evaluation 2: Strict Threshold (0.65)
*   **Recall (Churn):** `47%` (Catches fewer churners)
*   **Precision (Churn):** `49%` (Significant improvement in targeting accuracy)
*   **Overall Accuracy:** `79%`
*   *Business Impact:* Best used if the retention campaign is expensive (e.g., offering a $100 bonus or a personal phone call). By increasing the threshold, we ensure the bank only spends money on customers who are highly likely to leave.

## 🤝 Contributing
Contributions are welcome! If you'd like to improve the model (e.g., trying tree-based algorithms like Random Forest or XGBoost, or deploying this via Streamlit):
1. Fork the repository
2. Create your Feature Branch (`git checkout -b feature/AdvancedModeling`)
3. Commit your Changes (`git commit -m 'Add XGBoost Classifier'`)
4. Push to the Branch (`git push origin feature/AdvancedModeling`)
5. Open a Pull Request

## 📜 License
This project is open-source and available under the MIT License. See `LICENSE` for more information.

---

## 📬 Contact
**Selim Najaf**

*   **LinkedIn:** [linkedin.com/in/selimnajaf-data-analyst](https://www.linkedin.com/in/selimnajaf/)
*   **GitHub:** [github.com/SelimNajaf](https://github.com/SelimNajaf)

*Developed as a continuous learning initiative in advanced Data Science and ML Engineering.*
