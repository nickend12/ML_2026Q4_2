"""Entrenamiento y evaluación de la regresión lineal."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate(data):
    """Entrena la regresión y calcula sus métricas de prueba."""
    model = LinearRegression()
    model.fit(data["X_train_r_prepared"], data["y_train_r"])
    predictions = model.predict(data["X_test_r_prepared"])
    
    # Manejo de coeficientes con las columnas codificadas correctas
    feature_names = data.get("feature_names", [f"feat_{i}" for i in range(len(model.coef_))])
    coefficients = pd.Series(model.coef_, index=feature_names)
    
    return {
        "model": model,
        "predictions": predictions,
        "mae": mean_absolute_error(data["y_test_r"], predictions),
        "rmse": float(np.sqrt(mean_squared_error(data["y_test_r"], predictions))),
        "r2": r2_score(data["y_test_r"], predictions),
        "coefficients": coefficients,
    }