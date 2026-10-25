"""Generación del resumen automático de la actividad."""

from datetime import datetime, timezone

from .Reporte_clasificacion import report_classification
from .Reporte_regresion import report_regression
from ..datos.hf_loader import DATASET_ID, DATASET_SPLIT, DATASET_URL, SEED


def build_summary(df_raw, data, regression, classification):
    """Construye el contenido textual con resultados e interpretaciones."""
    coefficients = regression["coefficients"]
    confusion = classification["confusion"]
    negative_feature = coefficients.idxmin()
    positive_feature = coefficients.idxmax()
    return f"""RESUMEN AUTOMÁTICO DE MODELOS DE MACHINE LEARNING
============================================================
Fecha de ejecución UTC: {datetime.now(timezone.utc).isoformat()}
Dataset: {DATASET_ID}
URL: {DATASET_URL}
Split: {DATASET_SPLIT}
Dimensiones iniciales: {df_raw.shape}
Dimensiones después de limpiar: {data["df"].shape}

PREPROCESAMIENTO
----------------
Nulos iniciales: {data["missing_before"][data["missing_before"] > 0].to_dict()}
Duplicados iniciales: {data["duplicates_before"]}
Se reemplazaron '?' por NaN, se eliminaron nombres de vehículos y duplicados.
Se convirtieron los predictores a numéricos y se imputó la mediana.
Imputadores y escaladores se ajustaron únicamente con train.
La etiqueta efficiency usa la mediana de mpg ({data["median_mpg"]:.2f}): High >= mediana, Low < mediana.
Predictoras: {list(data["X"].columns)}
División: 80% train / 20% test; random_state={SEED}.

REGRESIÓN LINEAL
----------------
Objetivo: predecir mpg a partir de características físicas y mecánicas.
Algoritmo: LinearRegression con parámetros predeterminados.
{report_regression(regression)}
Interpretación: el modelo explica {regression["r2"] * 100:.2f}% de la varianza de mpg.
El coeficiente más negativo es {negative_feature}; el más positivo es {positive_feature}.
Cada coeficiente representa el cambio en mpg ante una desviación estándar adicional,
manteniendo las demás variables constantes. El intercepto corresponde a predictores promedio.
Limitación: una relación lineal puede no capturar interacciones o no linealidades.

CLASIFICACIÓN
-------------
Objetivo: clasificar efficiency como High o Low.
Algoritmo: RandomForestClassifier(n_estimators=100, max_depth=5, random_state={SEED}).
{report_classification(classification)}
Interpretación: aciertos Low={confusion[0, 0]}, aciertos High={confusion[1, 1]},
Low predicho como High={confusion[0, 1]}, High predicho como Low={confusion[1, 0]}.
Limitación: el resultado depende del umbral de la mediana usado para crear la etiqueta.

CONCLUSIÓN
----------
Ambos modelos se evaluaron sobre datos de prueba no vistos durante el ajuste.
Los resultados no implican causalidad y podrían cambiar con otros datos o particiones.
"""


def write_summary(content, path="resumen_modelos.txt"):
    """Escribe el resumen en la ruta indicada."""
    with open(path, "w", encoding="utf-8") as summary_file:
        summary_file.write(content)
