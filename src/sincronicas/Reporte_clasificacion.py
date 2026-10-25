"""Presentación de resultados de clasificación."""


def report_classification(results):
    """Devuelve una presentación textual de las métricas de clasificación."""
    return (
        f"Accuracy: {results['accuracy']:.4f}\n"
        f"Precision High: {results['precision']:.4f}\n"
        f"Recall High: {results['recall']:.4f}\n"
        f"F1 High: {results['f1']:.4f}\n"
        f"Matriz de confusión [Low, High]:\n{results['confusion']}"
    )
