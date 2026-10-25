"""Carga y preparación del dataset de Hugging Face."""

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_ID = "scikit-learn/auto-mpg"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
DATASET_SPLIT = "train"
SEED = 42


def load_dataframe():
    """Descarga el split configurado y lo convierte en un DataFrame."""
    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    return dataset.to_pandas()


def prepare_data(df_raw):
    """Limpia, transforma y divide los datos sin fuga de información."""
    df = df_raw.replace("?", np.nan).copy()
    missing_before = df.isna().sum()
    duplicates_before = int(df.duplicated().sum())
    df = df.drop(columns=[column for column in ["car_name", "car name", "name"] if column in df])
    df = df.drop_duplicates().copy()

    median_mpg = float(df["mpg"].median())
    df["efficiency"] = np.where(df["mpg"] >= median_mpg, "High", "Low")
    X = df.drop(columns=["mpg", "efficiency"]).apply(pd.to_numeric, errors="coerce")
    y_reg = df["mpg"]
    y_cls = df["efficiency"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=SEED
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_cls, test_size=0.2, random_state=SEED, stratify=y_cls
    )

    imputer_r = SimpleImputer(strategy="median")
    imputer_c = SimpleImputer(strategy="median")
    scaler_r = StandardScaler()
    scaler_c = StandardScaler()
    X_train_r_prepared = scaler_r.fit_transform(imputer_r.fit_transform(X_train_r))
    X_test_r_prepared = scaler_r.transform(imputer_r.transform(X_test_r))
    X_train_c_prepared = scaler_c.fit_transform(imputer_c.fit_transform(X_train_c))
    X_test_c_prepared = scaler_c.transform(imputer_c.transform(X_test_c))

    return {
        "df": df,
        "missing_before": missing_before,
        "duplicates_before": duplicates_before,
        "median_mpg": median_mpg,
        "X": X,
        "y_reg": y_reg,
        "y_cls": y_cls,
        "X_train_r_prepared": X_train_r_prepared,
        "X_test_r_prepared": X_test_r_prepared,
        "y_train_r": y_train_r,
        "y_test_r": y_test_r,
        "X_train_c_prepared": X_train_c_prepared,
        "X_test_c_prepared": X_test_c_prepared,
        "y_train_c": y_train_c,
        "y_test_c": y_test_c,
    }
