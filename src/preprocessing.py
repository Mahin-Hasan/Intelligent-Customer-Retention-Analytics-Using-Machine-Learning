import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.config import RANDOM_STATE, TEST_SIZE, TARGET_COLUMN, DROP_COLUMNS


def clean_data(df):
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def split_features_target(df):
    X = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    return X, y


def identify_column_types(X):
    X = X.drop(columns=DROP_COLUMNS, errors="ignore")

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    return categorical_cols, numerical_cols


def build_preprocessor(categorical_cols, numerical_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    return preprocessor


def split_data(X, y):
    X = X.drop(columns=DROP_COLUMNS, errors="ignore")

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )