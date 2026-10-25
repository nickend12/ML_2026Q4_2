# EA2 Unidad 1: Modelos de predicción y clasificación

Proyecto de Machine Learning para predecir el consumo de combustible de
vehículos mediante regresión lineal y clasificar su nivel de eficiencia
mediante Random Forest.

## Integrantes

- Rober Andrés Castillo
- Oscar Javier García

## Alcance de esta entrega

La entrega actual corresponde exclusivamente a estos archivos:

- [`actividad_modelos.ipynb`](./src/clases_po/reportes/actividad_modelos.ipynb):
  punto de entrada y evidencia
  reproducible del flujo completo.
- [`src/`](./src/): módulos de carga, preprocesamiento, modelos y generación del
  resumen.
- `resumen_modelos.txt`: resultado generado durante la ejecución y publicado
  únicamente como artefacto de GitHub Actions; no se versiona en el repositorio.
- [`requirements.txt`](./requirements.txt): dependencias del proyecto.
- [`.github/workflows/ml_pipeline.yml`](./.github/workflows/ml_pipeline.yml): ejecución automática.
- `README.md`: documentación y guía de ejecución.

Los archivos de ejercicios anteriores fueron retirados del estado final del
repositorio en el commit `bf47e9c`. El historial anterior se conserva para
mostrar la evolución del repositorio, pero no forma parte de esta entrega.

## Dataset y objetivo

- **Dataset:** [`scikit-learn/auto-mpg`](https://huggingface.co/datasets/scikit-learn/auto-mpg)
- **Split utilizado:** `train`
- **Carga:** automática mediante `datasets.load_dataset`.
- **Registros iniciales:** 398.
- **Columnas iniciales:** 9.

El dataset contiene información física y mecánica de automóviles. Se
responden dos preguntas utilizando los mismos datos:

1. ¿Se puede predecir el consumo continuo `mpg` a partir de las características
   del vehículo?
2. ¿Se puede clasificar el vehículo como de eficiencia `High` o `Low`?

### Variables principales

| Columna | Tipo | Uso |
|---|---|---|
| `mpg` | Numérico continuo | Objetivo de regresión. |
| `cylinders` | Numérico entero | Variable predictora. |
| `displacement` | Numérico continuo | Variable predictora. |
| `horsepower` | Texto inicialmente, numérico después | Variable predictora con valores faltantes. |
| `weight` | Numérico entero | Variable predictora. |
| `acceleration` | Numérico continuo | Variable predictora. |
| `model year` | Numérico entero | Variable predictora. |
| `origin` | Numérico entero | Variable predictora. |
| `car name` | Texto | Se elimina por ser un identificador textual de alta cardinalidad. |

## Organización del proyecto

La lógica sigue la estructura modular presentada en clase:

```text
src/
├── clases_po/                       # Archivos de modelos y reportes
│   ├── hf_loader.py                 # Carga y preprocesamiento
│   ├── regresion_lineal.py          # Modelo y métricas de regresión
│   ├── clasificacion_riesgo.py      # Modelo y métricas de clasificación
│   ├── Reporte_regresion.py         # Presentación de resultados de regresión
│   ├── Reporte_clasificacion.py     # Presentación de resultados de clasificación
│   ├── reportes.py                  # Generación de resumen_modelos.txt
│   └── reportes/
│       └── actividad_modelos.ipynb   # Ejecución y evidencia visible
```

El notebook importa estos módulos y conserva las salidas de cada etapa para
que la evidencia pueda revisarse directamente. El workflow de GitHub Actions
ejecuta el notebook desde la raíz del repositorio. No se incluyen los CSV ni
los reportes del proyecto de otro grupo.

## Verificación frente a la rúbrica

| Componente | Evidencia |
|---|---|
| 0. DataFrame automático | Carga desde Hugging Face y dimensiones, columnas, tipos y muestra en el notebook. |
| 1. Preprocesamiento | Nulos, duplicados, tipos, columnas finales, etiqueta, división train/test, imputación y escalado. |
| 2. Regresión | `regresion_lineal.py`, métricas MAE/RMSE/R², coeficientes, intercepto e interpretación. |
| 3. Clasificación | `clasificacion_riesgo.py`, métricas por clase y matriz de confusión. |
| 4. Resumen `.txt` | `reportes.py` genera `resumen_modelos.txt`; Actions lo publica como artefacto sin versionarlo. |
| 5. Automatización | `ml_pipeline.yml` ejecuta el notebook y publica el artefacto en GitHub Actions. |

## Etapas del notebook

El notebook [`actividad_modelos.ipynb`](./src/clases_po/reportes/actividad_modelos.ipynb) contiene
todo el flujo reproducible:

### 0. Construcción automática del DataFrame

- Descarga `scikit-learn/auto-mpg` directamente desde Hugging Face.
- Convierte el dataset a un `DataFrame` de Pandas.
- Muestra dimensiones, columnas, tipos de datos y primeras filas.

### 1. Preprocesamiento

- Reemplaza el carácter `?` por `NaN`.
- Detecta 6 valores faltantes en `horsepower`.
- Verifica duplicados.
- Elimina `car name`.
- Convierte los predictores a valores numéricos.
- Imputa los faltantes con la mediana.
- Crea `efficiency` usando la mediana de `mpg`:
  - `High`: `mpg >= 23.00`.
  - `Low`: `mpg < 23.00`.
- Excluye `mpg` de los predictores de clasificación para evitar fuga de
  información.
- Divide los datos en 80 % de entrenamiento y 20 % de prueba.
- Ajusta el imputador y el escalador únicamente con entrenamiento.
- Usa `random_state=42`.

### 2. Regresión lineal

- **Objetivo:** predecir `mpg`.
- **Algoritmo:** `LinearRegression` con parámetros predeterminados.
- **Evaluación:** MAE, RMSE y R².
- **Interpretación:** coeficientes e intercepto considerando que los
  predictores fueron estandarizados.

Resultados de la ejecución local:

```text
MAE:  2.2556
RMSE: 2.8632
R²:   0.8475
```

El coeficiente más negativo corresponde a `weight`, mientras que el más
positivo corresponde a `model year`. Al estar estandarizadas las variables,
cada coeficiente representa el cambio esperado en `mpg` ante un aumento de una
desviación estándar en el predictor, manteniendo constantes los demás.

### 3. Clasificación

- **Objetivo:** clasificar `efficiency` como `High` o `Low`.
- **Algoritmo:** `RandomForestClassifier`.
- **Parámetros:** `n_estimators=100`, `max_depth=5`,
  `random_state=42`.
- **Evaluación:** accuracy, precision, recall, F1 y matriz de confusión.

Random Forest es adecuado porque puede capturar relaciones no lineales e
interacciones entre las características numéricas del vehículo.

Resultados de la ejecución local:

```text
Accuracy:  0.9750
Precision: 0.9750
Recall:    0.9750
F1:        0.9750
```

Matriz de confusión, con filas reales y columnas predichas en el orden
`[Low, High]`:

```text
[[39  1]
 [ 1 39]]
```

El modelo clasificó correctamente 39 vehículos `Low` y 39 vehículos `High`.
Confundió un vehículo `Low` como `High` y un vehículo `High` como `Low`.

### 4. Resumen automático

La última sección del notebook genera `resumen_modelos.txt`, que contiene:

- Fecha de ejecución.
- Identificación y URL del dataset.
- Split y dimensiones.
- Estado inicial y operaciones de preprocesamiento.
- Variables, parámetros y configuración de cada modelo.
- Métricas de prueba.
- Coeficientes e intercepto de la regresión.
- Matriz de confusión y errores de clasificación.
- Interpretaciones y limitaciones.

El archivo no se redacta manualmente: se sobrescribe al ejecutar el notebook.

## Ejecución local

```bash
git clone https://github.com/nickend12/ML_2026Q4_2.git
cd ML_2026Q4_2

python3.12 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 \
  src/clases_po/reportes/actividad_modelos.ipynb
```

Al finalizar, el archivo `resumen_modelos.txt` se genera localmente y queda
ignorado por Git. En GitHub Actions se publica como artefacto descargable.

## Ejecución automática en GitHub Actions

El workflow [`ml_pipeline.yml`](./.github/workflows/ml_pipeline.yml) se
ejecuta automáticamente en:

- `push` a `main` o `master`.
- `pull_request` hacia `main` o `master`.
- Ejecución manual mediante `workflow_dispatch`.

El workflow:

1. Descarga el repositorio.
2. Configura Python 3.12.
3. Instala `requirements.txt`.
4. Ejecuta `src/clases_po/reportes/actividad_modelos.ipynb` con `nbconvert`.
5. Publica `resumen_modelos.txt` como artefacto descargable.

## Historial de la actividad

Los commits principales de esta entrega son:

```text
045df9c feat: agregar notebook reproducible de modelos
b3c8990 docs: documentar flujo y resultados de la actividad
bf47e9c chore: limpiar archivos de ejercicios anteriores
```

## Limitaciones

- La etiqueta `efficiency` depende del umbral de la mediana de `mpg`.
- La regresión lineal puede no capturar relaciones no lineales o interacciones.
- Los resultados corresponden a una partición concreta con semilla 42.
- El rendimiento puede cambiar con vehículos de otras épocas o distribuciones.
- Los resultados de los modelos no implican causalidad.

## Enlaces de entrega

- [Repositorio](https://github.com/nickend12/ML_2026Q4_2)
- [Commits de la rama `main`](https://github.com/nickend12/ML_2026Q4_2/commits/main)
- [Ejecuciones de GitHub Actions](https://github.com/nickend12/ML_2026Q4_2/actions)
