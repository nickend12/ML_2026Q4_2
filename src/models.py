"""Entrenamiento y evaluación de los modelos de la actividad."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
)


SEED = 42


def train_and_evaluate_regression(data):
    """Entrena la regresión lineal y devuelve el modelo y sus resultados."""
    model = LinearRegression()
    model.fit(data["X_train_r_prepared"], data["y_train_r"])
    predictions = model.predict(data["X_test_r_prepared"])
    coefficients = pd.Series(model.coef_, index=data["X"].columns)
    return {
        "model": model,
        "predictions": predictions,
        "mae": mean_absolute_error(data["y_test_r"], predictions),
        "rmse": float(np.sqrt(mean_squared_error(data["y_test_r"], predictions))),
        "r2": r2_score(data["y_test_r"], predictions),
        "coefficients": coefficients,
    }


def train_and_evaluate_classification(data):
    """Entrena Random Forest y devuelve el modelo y sus resultados."""
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    model.fit(data["X_train_c_prepared"], data["y_train_c"])
    predictions = model.predict(data["X_test_c_prepared"])
    labels = ["Low", "High"]
    confusion = confusion_matrix(data["y_test_c"], predictions, labels=labels)
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        data["y_test_c"], predictions, labels=labels, zero_division=0
    )
    class_metrics = pd.DataFrame(
        {
            "precision": class_precision,
            "recall": class_recall,
            "f1": class_f1,
            "support": class_support,
        },
        index=labels,
    )
    return {
        "model": model,
        "predictions": predictions,
        "accuracy": accuracy_score(data["y_test_c"], predictions),
        "precision": precision_score(data["y_test_c"], predictions, pos_label="High"),
        "recall": recall_score(data["y_test_c"], predictions, pos_label="High"),
        "f1": f1_score(data["y_test_c"], predictions, pos_label="High"),
        "confusion": confusion,
        "class_metrics": class_metrics,
    }
