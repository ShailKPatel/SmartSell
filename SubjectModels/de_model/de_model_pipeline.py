# math3_model_pipeline.py

import pandas as pd
import numpy as np
import warnings
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV

# Suppress warnings
warnings.filterwarnings("ignore")


### 1. Custom Transformer for Feature Engineering
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        drop_cols = [
            "Student ID",
            "Mentor-1",
            "Mentor-2",
            "Mentor-3",
            "Roll-2",
            "Roll-3",
            "Math-3 Theory",
            "DE Practical",
            "FSD Theory",
            "FSD Practical",
            "Python Theory",
            "Python Practical",
            "Communication Theory",
            "Law Theory",
        ]
        X = X.drop(columns=[col for col in drop_cols if col in X.columns], errors="ignore")

        # Semester 1 and 2 Percentages
        sem1_cols = [
            "Math-1 Theory",
            "Physics Theory",
            "Java-1 Theory",
            "Software Engineering Theory",
        ]
        sem2_cols = [
            "Math-2 Theory",
            "Data Structures using Java Theory",
            "DBMS Theory",
            "Fundamental of Electronics and Electrical Theory",
            "Java-2 Theory",
        ]

        if all(col in X.columns for col in sem1_cols):
            X["Sem 1 Percentage"] = X[sem1_cols].mean(axis=1).round(2)
        if all(col in X.columns for col in sem2_cols):
            X["Sem 2 Percentage"] = X[sem2_cols].mean(axis=1).round(2)

        # Rename sections
        X = X.rename(
            columns={"Div-1": "Section-1", "Div-2": "Section-2", "Div-3": "Section-3"}
        )

        # Extract department initials
        for section in ["Section-1", "Section-2", "Section-3"]:
            if section in X.columns:
                X[section] = X[section].astype(str).str[0]

        return X


### 2. Load Raw Data
df = pd.read_csv("../train_dataset.csv")

target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

### 3. Define Categorical Columns and Preprocessor
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), make_column_selector(dtype_include=np.number)),
    ]
)

### 4. Setup Models and Hyperparameter Spaces
kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

param_spaces = {
    "ridge": {"regressor__alpha": (1e-3, 1e3, "log-uniform")},
    "lasso": {"regressor__alpha": (1e-3, 1e3, "log-uniform")},
    "elastic": {
        "regressor__alpha": (1e-3, 1e3, "log-uniform"),
        "regressor__l1_ratio": (0.05, 1.0, "uniform"),
    },
}


def tune_model(estimator, search_space):
    pipeline = Pipeline(
        [
            ("custom", CustomPreprocessor()),
            ("pre", preprocessor),
            ("regressor", estimator),
        ]
    )
    opt = BayesSearchCV(
        pipeline,
        search_spaces=search_space,
        cv=kf,
        n_iter=50,
        scoring="neg_mean_absolute_error",
        random_state=42,
    )
    opt.fit(X, y)
    return opt.best_estimator_


### 5. Tune All 3 Models
ridge_best = tune_model(Ridge(), param_spaces["ridge"])
lasso_best = tune_model(Lasso(max_iter=10000), param_spaces["lasso"])
elastic_best = tune_model(ElasticNet(max_iter=10000), param_spaces["elastic"])

### 6. Final Ensemble Pipeline
ensemble = VotingRegressor(
    [("ridge", ridge_best), ("lasso", lasso_best), ("elastic", elastic_best)]
)

# Fit final ensemble model
ensemble.fit(X, y)

# Evaluate
mae = np.mean(np.abs(y - ensemble.predict(X)))
print("Model: Voting Ensemble (Ridge + Lasso + ElasticNet)")
print("MAE:", round(mae, 4))

### 7. Save Model
joblib.dump(ensemble, "de_predictor.joblib")
