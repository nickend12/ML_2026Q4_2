# S30 - EA2 Unidad 1: Modelos de Predicción y Clasificación en Machine Learning

## 📌 Información General del Proyecto
- **Asignatura:** Machine Learning
- **Docente:** Andrés Callejas
- **Integrantes:**
  - Rober Andrés Castillo
  - Oscar Javier García
- **Repositorio:** `ML_2026Q4_2`

---

## 📊 0. Identificación y Carga Automática del Dataset
- **Identificador en Hugging Face:** [`scikit-learn/auto-mpg`](https://huggingface.co/datasets/scikit-learn/auto-mpg)
- **Configuración y Split:** Split de entrenamiento (`train`) cargado mediante la librería `datasets`.
- **Dimensiones:** 398 registros y 9 columnas originales.
- **Descripción del problema:** Análisis y estimación de la eficiencia del consumo de combustible en automóviles (medido en millas por galón - `mpg`) en función de sus características físicas y mecánicas.

| Columna | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `mpg` | Numérico Continuo | Millas por galón (Variable objetivo para Regresión). |
| `cylinders` | Numérico Entero | Número de cilindros del motor. |
| `displacement` | Numérico Continuo | Desplazamiento del motor (pulgadas cúbicas). |
| `horsepower` | Numérico Continuo | Caballos de fuerza del vehículo. |
| `weight` | Numérico Continuo | Peso total del automóvil (libras). |
| `acceleration` | Numérico Continuo | Tiempo de aceleración de 0 a 60 mph (segundos). |
| `model_year` | Numérico Entero | Año del modelo del vehículo. |
| `origin` | Numérico Entero | Origen geográfico del vehículo (1: EE.UU., 2: Europa, 3: Japón). |
| `car_name` | Clic/Texto | Nombre del modelo (Eliminado en preprocesamiento). |

---

## 🧹 1. Preprocesamiento de Datos y Fuga de Información (Data Leakage)
Para garantizar la integridad y calidad de las entradas a los modelos, se aplicaron las siguientes transformaciones:

1. **Limpieza de Valores Nulos y Caracteres Especiales:**
   - Se detectaron valores perdidos codificados como el carácter `'?'` en la variable `horsepower`. Estos se reemplazaron por `NaN` y se convirtieron a formato numérico continuo (`float`).
   - Se aplicó un imputador basado en la **mediana** (`SimpleImputer(strategy='median')`).

2. **Eliminación de Variables no Predictivas:**
   - La columna `car_name` fue removida del espacio de características ($X$) por tratarse de un identificador de texto con alta cardinalidad sin capacidad de generalización predictiva.

3. **Construcción de la Variable de Clasificación (`efficiency`):**
   - Se creó la etiqueta categórica dicotómica `efficiency`:
     - **`High`**: Vehículos con un valor de `mpg` mayor o igual a la mediana del dataset ($23.00$ mpg).
     - **`Low`**: Vehículos con un valor de `mpg` inferior a la mediana.

4. **Prevención de Fuga de Información (Data Leakage):**
   - **Para Clasificación:** Se eliminó explícitamente la columna continua `mpg` de las variables predictoras $X$ para evitar que el modelo acceda directamente a la respuesta original.
   - **En Transformaciones:** Los transformadores (`SimpleImputer` y `StandardScaler`) ejecutaron la fase de ajuste (`.fit()`) **únicamente sobre el conjunto de entrenamiento (Train)**. El conjunto de prueba (Test) se transformó en estricto aislamiento utilizando únicamente las estadísticas previamente aprendidas.

---

## 📈 2. Modelo de Regresión Lineal e Interpretación
- **Objetivo:** Predecir cuantitativamente el rendimiento continuo de combustible (`mpg`).
- **Algoritmo:** Regresión Lineal por Mínimos Cuadrados Ordinarios (`LinearRegression`).
- **Configuración del Experimento:**
  - **División:** 80% Entrenamiento / 20% Prueba.
  - **Semilla Aleatoria (`random_state`):** 42.
  - **Escalado:** `StandardScaler` aplicado en Train.

### Métricas de Evaluación en Test:
- **Error Absoluto Medio (MAE):** Métricas obtenidas directamente en escala de $mpg$.
- **Raíz del Error Cuadrático Medio (RMSE):** Evalúa la magnitud de las desviaciones ponderando errores mayores.
- **Coeficiente de Determinación ($R^2$):** Indica la proporción de la varianza explicada por el modelo.

### Interpretación de Coeficientes e Intercepto:
Los coeficientes estandarizados muestran que la variable **`weight` (peso)** presenta una relación fuertemente negativa con el consumo; a mayor peso del vehículo, menor es la cantidad de millas recorridas por galón. Por el contrario, **`model_year` (año del modelo)** muestra una contribución positiva, reflejando avances tecnológicos en eficiencia de combustible con el paso de los años.

---

## 🌲 3. Modelo de Clasificación e Interpretación
- **Objetivo:** Categorizar los vehículos en clases de eficiencia **`High`** vs **`Low`**.
- **Algoritmo Seleccionado:** **Random Forest Classifier** (`RandomForestClassifier`).
- **Justificación del Algoritmo:** Los conjuntos de árboles de decisión capturan relaciones no lineales e interacciones complejas entre variables físicas (ej. relación entre cilindrada, peso y aceleración) sin requerir supuestos de normalidad.
- **Parámetros:** `n_estimators=100`, `max_depth=5`, `random_state=42`.

### Métricas de Evaluación en Test:
- **Exactitud (Accuracy):** Porcentaje de clasificaciones correctas sobre el total.
- **Precisión (Precision):** Proporción de positivos predichos que fueron correctos.
- **Exhaustividad (Recall):** Capacidad del modelo para identificar la totalidad de los vehículos de alta eficiencia.
- **F1-Score:** Media armónica entre Precisión y Recall.
- **Matriz de Confusión:** Permite analizar los falsos positivos y falsos negativos respecto a las categorías real vs predicha.

---

## 📄 4. Resumen Automático (`resumen_modelos.txt`)
El notebook ejecutable `actividad_modelos.ipynb` genera automáticamente al finalizar su corrida un archivo de texto plano en la raíz del proyecto titulado **`resumen_modelos.txt`**.

Este archivo recopila:
- Estampa de tiempo (fecha y hora exacta de ejecución).
- Metadatos del dataset y transformaciones aplicadas.
- Métricas, parámetros y resumen de coeficientes/matriz de confusión de ambos modelos.

---

## ⚙️ 5. Ejecución Automática en GitHub Actions (CI/CD)
El proyecto incluye un flujo de integración continua ubicado en `.github/workflows/ml_pipeline.yml`.

### Características del Workflow:
- **Disparador (`on`):** Se activa automáticamente con cada evento `push` o `pull_request` a la rama principal (`main`).
- **Pasos de Ejecución:**
  1. Descarga el repositorio mediante `actions/checkout@v4`.
  2. Configura el entorno Python 3.12.
  3. Instala las dependencias declaradas en `requirements.txt`.
  4. Ejecuta el notebook completo:
     ```bash
     jupyter nbconvert --to notebook --execute --inplace actividad_modelos.ipynb
     ```
  5. Empaqueta el archivo `resumen_modelos.txt` y lo publica como un **Artefacto Descargable** (`upload-artifact@v4`).

---

## 💻 6. Instrucciones de Ejecución Local

### Pasos para replicar el proyecto:
```bash
# 1. Clonar el repositorio
git clone [https://github.com/nickend12/ML_2026Q4_2.git](https://github.com/nickend12/ML_2026Q4_2.git)
cd ML_2026Q4_2

# 2. Crear y activar el entorno virtual
python3.12 -m venv venv
source venv/bin/activate  # En Windows use: venv\Scripts\activate

# 3. Instalar las dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Ejecutar el notebook y generar el resumen
jupyter nbconvert --to notebook --execute --inplace actividad_modelos.ipynb
```