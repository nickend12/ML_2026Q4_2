# EA2 Unidad 1: Modelos de predicción y clasificación

Proyecto de Machine Learning para predecir la calificación final de estudiantes
mediante regresión lineal y clasificar su nivel de rendimiento académico
mediante Random Forest.

## Integrantes

- Rober Andrés Castillo
- Oscar Javier García

## Alcance de esta entrega

La entrega actual corresponde exclusivamente a estos archivos:

- [`actividad_modelos.ipynb`](./src/clases_po/reportes/actividad_modelos.ipynb):
  punto de entrada y evidencia reproducible del flujo completo.
- [`src/`](./src/): módulos de carga, preprocesamiento, modelos y generación
  del resumen.
- `resumen_modelos.txt`: resultado generado durante la ejecución y publicado
  únicamente como artefacto de GitHub Actions; no se versiona en el repositorio.
- [`requirements.txt`](./requirements.txt): dependencias del proyecto.
- [`.github/workflows/ml_pipeline.yml`](./.github/workflows/ml_pipeline.yml):
  ejecución automática.
- `README.md`: documentación y guía de ejecución.

Los archivos de ejercicios anteriores fueron retirados del estado final del
repositorio en el commit `bf47e9c`. El historial anterior se conserva para
mostrar la evolución del repositorio, pero no forma parte de esta entrega.

## Dataset y objetivo

- **Dataset:** [`scikit-learn/student-alcohol-consumption`](https://huggingface.co/datasets/scikit-learn/student-alcohol-consumption)
- **Split utilizado:** `train`
- **Carga:** automática mediante `datasets.load_dataset`.

El dataset contiene información demográfica, social y relacionada con el ámbito
escolar de estudiantes. Se responden dos preguntas utilizando los mismos datos:

1. ¿Se puede predecir la calificación final continua `G3` a partir de las
   características del estudiante y sus hábitos?
2. ¿Se puede clasificar el estudiante con un rendimiento (`performance`) `High`
   o `Low`?

### Variables principales

| Columna | Tipo | Uso |
|---|---|---|
| `G3` | Numérico continuo (0-20) | Objetivo de regresión (Nota final). |
| `performance` | Categórico (High/Low) | Objetivo de clasificación (derivado de G3). |
| `age`, `absences`, `studytime` | Numérico | Variables predictoras numéricas. |
| `sex`, `address`, `famsize` | Texto / Categórico | Variables predictoras (se codifican con One-Hot Encoding). |
| `id`, `student_id` | Texto | Se eliminan por ser identificadores de alta cardinalidad (si existen). |

## Organización del proyecto

La lógica sigue la estructura modular presentada en clase:

```text
src/
└── clases_po/
    ├── datos/
    │   └── hf_loader.py              # Carga, preprocesamiento, OHE y escalado
    ├── entrenamiento/
    │   ├── regresion_lineal.py       # Modelo y métricas de regresión
    │   └── clasificacion_riesgo.py   # Modelo y métricas de clasificación
    └── reportes/
        ├── Reporte_regresion.py      # Resultados de regresión
        ├── Reporte_clasificacion.py  # Resultados de clasificación
        ├── reportes.py               # Resumen automático
        └── actividad_modelos.ipynb   # Evidencia visible del flujo
```

El notebook importa estos módulos y conserva las salidas de cada etapa para
que la evidencia pueda revisarse directamente. El workflow de GitHub Actions
ejecuta el notebook desde la raíz del repositorio.

## Verificación frente a la rúbrica

| Componente | Evidencia |
|---|---|
| 0. DataFrame automático | Carga desde Hugging Face y dimensiones, columnas, tipos y muestra en el notebook. |
| 1. Preprocesamiento | Nulos, duplicados, One-Hot Encoding para categóricas, escalado, división train/test y etiqueta. |
| 2. Regresión | `entrenamiento/regresion_lineal.py`, métricas MAE/RMSE/R², coeficientes e interpretación. |
| 3. Clasificación | `entrenamiento/clasificacion_riesgo.py`, métricas por clase y matriz de confusión. |
| 4. Resumen `.txt` | `reportes/reportes.py` genera `resumen_modelos.txt`; Actions lo publica como artefacto. |
| 5. Automatización | `ml_pipeline.yml` ejecuta el notebook y publica el artefacto en GitHub Actions. |

## Etapas del notebook

El notebook [`actividad_modelos.ipynb`](./src/clases_po/reportes/actividad_modelos.ipynb)
contiene todo el flujo reproducible:

### 0. Construcción automática del DataFrame

- Descarga `scikit-learn/student-alcohol-consumption` directamente desde
  Hugging Face.
- Convierte el dataset a un `DataFrame` de Pandas.
- Muestra dimensiones, columnas, tipos de datos y primeras filas.

### 1. Preprocesamiento

- Reemplaza el carácter `?` por `NaN` (si existe).
- Verifica nulos y duplicados.
- Elimina identificadores textuales (ej. `id`).
- Aplica `OneHotEncoder` a las variables categóricas (texto) y
  `StandardScaler` a las numéricas.
- Crea la variable `performance` usando la mediana de `G3`:
  - `High`: `G3 >= mediana`.
  - `Low`: `G3 < mediana`.
- Excluye `G3` y `performance` de los predictores correspondientes para evitar
  fuga de información.
- Divide los datos en 80 % de entrenamiento y 20 % de prueba.
- Usa `random_state=42`.

### 2. Regresión lineal

- **Objetivo:** predecir `G3` (nota final).
- **Algoritmo:** `LinearRegression` con parámetros predeterminados.
- **Evaluación:** MAE, RMSE y R².
- **Interpretación:** coeficientes e intercepto analizados con base en variables
  estandarizadas y codificadas.

### 3. Clasificación

- **Objetivo:** clasificar `performance` como `High` o `Low`.
- **Algoritmo:** `RandomForestClassifier`.
- **Parámetros:** `n_estimators=100`, `max_depth=5`, `random_state=42`.
- **Evaluación:** accuracy, precision, recall, F1 y matriz de confusión.

Random Forest es adecuado porque puede capturar relaciones no lineales e
interacciones entre características demográficas, hábitos de estudio y la
calificación del estudiante.

### 4. Resumen automático

La última sección del notebook genera `resumen_modelos.txt`, que contiene:

- Fecha de ejecución.
- Identificación y URL del dataset.
- Operaciones de preprocesamiento (escalado y codificación de variables).
- Variables, parámetros y configuración de cada modelo.
- Métricas de prueba.
- Coeficientes de la regresión.
- Matriz de confusión.

El archivo se sobrescribe automáticamente al ejecutar el notebook.

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

El workflow [`ml_pipeline.yml`](./.github/workflows/ml_pipeline.yml) se ejecuta
automáticamente en `push`, `pull_request` o `workflow_dispatch`.

El workflow:

1. Descarga el repositorio.
2. Configura Python 3.12.
3. Instala `requirements.txt`.
4. Ejecuta `src/clases_po/reportes/actividad_modelos.ipynb` con `nbconvert`.
5. Publica `resumen_modelos.txt` como artefacto descargable.

## Historial de la actividad

Los commits principales que evidencian la evolución de esta entrega son:

```text
(NUEVO) feat: migración completa y exitosa a dataset student-alcohol-consumption
045df9c feat: agregar notebook reproducible de modelos
b3c8990 docs: documentar flujo y resultados de la actividad
bf47e9c chore: limpiar archivos de ejercicios anteriores
26ae4ac refactor: mover notebook a carpeta de reportes
eb61529 refactor: ordenar modulos por responsabilidad
fb5c6ab chore: actualizar evidencia ejecutada del notebook
```

## Limitaciones

- La etiqueta `performance` depende del umbral de la mediana de `G3`.
- La regresión lineal puede no capturar relaciones no lineales en
  comportamientos sociales complejos.
- Los resultados corresponden a una partición concreta con semilla 42.
- Los resultados de los modelos no implican causalidad directa.
