from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.preprocessor = None

    def fit(self, X, y=None):

        # avtomatik column selection
        categorical_columns = X.select_dtypes(include=['object','category']).columns.tolist()
        numeric_columns = X.select_dtypes(include=['int64','float64']).columns.tolist()

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

        self.preprocessor.fit(X)

        return self

    def transform(self, X):
        return self.preprocessor.transform(X)