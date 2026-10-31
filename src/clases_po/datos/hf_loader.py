"""Carga y preparación del dataset de Hugging Face (Student Alcohol Consumption)."""

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATASET_ID = "scikit-learn/student-alcohol-consumption"
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
    
    # Limpiar columnas innecesarias o identificadores si los hay
    df = df.drop(columns=[col for col in ["id", "student_id"] if col in df])
    df = df.drop_duplicates().copy()

    # Variables objetivo
    median_g3 = float(df["G3"].median())
    df["performance"] = np.where(df["G3"] >= median_g3, "High", "Low")
    
    # Predecimos con todas las variables excepto G3 y performance
    X = df.drop(columns=["G3", "performance"])
    y_reg = df["G3"]
    y_cls = df["performance"]

    # Separar tipos de datos
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Splits
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=SEED
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_cls, test_size=0.2, random_state=SEED, stratify=y_cls
    )

    # Imputadores
    imputer_num = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    # Transformaciones numéricas
    X_train_r_num = scaler.fit_transform(imputer_num.fit_transform(X_train_r[num_cols]))
    X_test_r_num = scaler.transform(imputer_num.transform(X_test_r[num_cols]))
    
    X_train_c_num = scaler.fit_transform(imputer_num.fit_transform(X_train_c[num_cols]))
    X_test_c_num = scaler.transform(imputer_num.transform(X_test_c[num_cols]))
    
    feature_names = num_cols.copy()

    # Transformaciones categóricas (si existen)
    if cat_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        
        # Regresión categóricas
        X_train_r_cat = encoder.fit_transform(imputer_cat.fit_transform(X_train_r[cat_cols]))
        X_test_r_cat = encoder.transform(imputer_cat.transform(X_test_r[cat_cols]))
        
        # Clasificación categóricas
        X_train_c_cat = encoder.fit_transform(imputer_cat.fit_transform(X_train_c[cat_cols]))
        X_test_c_cat = encoder.transform(imputer_cat.transform(X_test_c[cat_cols]))
        
        # Ensamblar finales
        X_train_r_prepared = np.hstack([X_train_r_num, X_train_r_cat])
        X_test_r_prepared = np.hstack([X_test_r_num, X_test_r_cat])
        X_train_c_prepared = np.hstack([X_train_c_num, X_train_c_cat])
        X_test_c_prepared = np.hstack([X_test_c_num, X_test_c_cat])
        
        # Nombres de variables codificadas
        cat_names = encoder.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(cat_names)
    else:
        X_train_r_prepared, X_test_r_prepared = X_train_r_num, X_test_r_num
        X_train_c_prepared, X_test_c_prepared = X_train_c_num, X_test_c_num

    return {
        "df": df,
        "missing_before": missing_before,
        "duplicates_before": duplicates_before,
        "median_mpg": median_g3,  # Se deja este nombre para no romper tus reportes
        "X": X,
        "feature_names": feature_names,
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