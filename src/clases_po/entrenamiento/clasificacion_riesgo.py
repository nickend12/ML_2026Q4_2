"""Entrenamiento y evaluación del modelo de clasificación."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

SEED = 42

def train_and_evaluate(data):
    """Entrena Random Forest y calcula sus métricas de prueba."""
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    model.fit(data["X_train_c_prepared"], data["y_train_c"])
    predictions = model.predict(data["X_test_c_prepared"])
    
    # Las etiquetas deben coincidir con las creadas en hf_loader.py ("Low", "High")
    labels = ["Low", "High"]
    
    # Matriz de confusión
    confusion = confusion_matrix(data["y_test_c"], predictions, labels=labels)
    
    # Métricas detalladas por clase
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
        "precision": precision_score(data["y_test_c"], predictions, pos_label="High", zero_division=0),
        "recall": recall_score(data["y_test_c"], predictions, pos_label="High", zero_division=0),
        "f1": f1_score(data["y_test_c"], predictions, pos_label="High", zero_division=0),
        "confusion": confusion,
        "class_metrics": class_metrics,
    }