"""Presentación de resultados de regresión."""


def report_regression(results):
    """Devuelve una presentación textual de las métricas de regresión."""
    return (
        f"MAE: {results['mae']:.4f}\n"
        f"RMSE: {results['rmse']:.4f}\n"
        f"R²: {results['r2']:.4f}\n"
        f"Intercepto: {results['model'].intercept_:.4f}\n"
        f"Coeficientes:\n{results['coefficients'].sort_values(ascending=False).to_string()}"
    )
