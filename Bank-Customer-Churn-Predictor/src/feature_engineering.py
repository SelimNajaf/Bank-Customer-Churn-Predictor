from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.salary_median = None 
        
    def fit(self, X, y=None):
        self.salary_median = X['EstimatedSalary'].median()
        return self
        
    def transform(self, X):
        df = X.copy()
        
        df['loyalty_score'] = df['Tenure'] * df['IsActiveMember']
        df['balance_per_product'] = df['Balance'] / (df['NumOfProducts'] + 1)
        df['salary_balance_ratio'] = df['EstimatedSalary'] / (df['Balance'] + 1)
        df['products_active_interaction'] = df['NumOfProducts'] * df['IsActiveMember']
        
        df['age_bucket'] = pd.cut(df['Age'], bins=[18,30,50,70,100], labels=["young","mid","senior","old"])
        df['age_bucket'] = df['age_bucket'].astype(str)
        
        df['credit_age_ratio'] = df['CreditScore'] / (df['Age'] + 1)
        df['zero_balance_flag'] = (df['Balance'] == 0).astype(int)
        
        df['high_salary_low_product'] = ((df['EstimatedSalary'] > self.salary_median) & (df['NumOfProducts'] <= 1)).astype(int)
        
        return df