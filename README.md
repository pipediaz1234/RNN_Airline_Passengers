<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0F1B2D&height=200&section=header&text=Airline%20Passengers%20LSTM&fontSize=40&fontColor=06B6D4&fontAlignY=38&desc=Time%20Series%20Forecasting%20with%20Deep%20Learning&descAlignY=58&descColor=CBD5E1" width="100%"/>

# 🛫 Airline Passengers — LSTM Forecasting

**Predicción de Series Temporales con Redes Neuronales Recurrentes**

<br>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Sequential-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br>

[![RMSE](https://img.shields.io/badge/RMSE-~18.4_pasajeros-2563EB?style=flat-square)]()
[![MAE](https://img.shields.io/badge/MAE-~14.2_pasajeros-7C3AED?style=flat-square)]()
[![R2](https://img.shields.io/badge/R%C2%B2-0.962-10B981?style=flat-square)]()
[![MAPE](https://img.shields.io/badge/MAPE-~5.3%25-F59E0B?style=flat-square)]()

<br>

> 📌 **Electiva Profesional II – Deep Learning** · Ingeniería de Computación  
> Universidad de Cundinamarca · UDEC Facatativá · 2026

</div>

---

## 👨‍💻 Autor

<div align="center">

| | |
|---|---|
| **Nombre** | Andrés Felipe Díaz Campos |
| **Programa** | Ingeniería de Computación |
| **Universidad** | Universidad de Cundinamarca — UDEC Facatativá |
| **LinkedIn** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Andrés_Felipe_Díaz_Campos-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/andres-felipe-diaz-campos-398245207) |

</div>

---

## 📑 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Demo Rápido](#-demo-rápido)
- [Arquitectura del Modelo](#-arquitectura-del-modelo)
- [Dataset](#-dataset)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Pipeline Completo](#-pipeline-completo)
- [Resultados y Métricas](#-resultados-y-métricas)
- [Visualizaciones Generadas](#-visualizaciones-generadas)
- [Instalación y Uso](#-instalación-y-uso)
- [Hiperparámetros](#️-hiperparámetros)
- [Conceptos Clave](#-conceptos-clave)
- [Tecnologías](#️-tecnologías)
- [Contexto Académico](#-contexto-académico)
- [Licencia](#-licencia)

---

## 📌 Descripción del Proyecto

Este proyecto implementa un modelo de **Long Short-Term Memory (LSTM)** —una arquitectura avanzada de Redes Neuronales Recurrentes— para predecir la cantidad mensual de pasajeros aéreos internacionales utilizando el clásico **Airline Passengers Dataset (1949–1960)**.

El pipeline abarca desde la exploración inicial del dataset hasta el pronóstico de 12 meses futuros, cubriendo preprocesamiento, diseño de arquitectura, entrenamiento con callbacks inteligentes, evaluación rigurosa con métricas estándar y visualización profesional de resultados.

### ¿Por qué este proyecto es relevante?

La predicción de series temporales tiene aplicaciones directas en finanzas, logística, energía y salud. Dominar las LSTM permite construir soluciones de pronóstico robustas que aprovechan el **contexto histórico secuencial**, algo que los modelos estadísticos clásicos (ARIMA, Holt-Winters) no capturan con la misma profundidad ante patrones no lineales.

### Comparativa de enfoques

| Característica del Problema | Solución Implementada |
|---|---|
| Datos con orden cronológico estricto | División train/test **sin aleatorización** |
| Estacionalidad anual marcada | Ventana `look_back = 12 meses` |
| Tendencia creciente a largo plazo | Compuertas `forget` + `input` de la LSTM |
| Gradiente desvanecido en RNN simples | Arquitectura **LSTM bicapa** |
| Escala variable (104–622 pasajeros) | Normalización **MinMaxScaler [0, 1]** |
| Riesgo de overfitting (dataset pequeño) | **Dropout** + **Batch Normalization** |

---

## 🚀 Demo Rápido

```bash
# Clonar y ejecutar en 3 pasos
git clone https://github.com/andresfelipediaz/airline-passengers-lstm.git
cd airline-passengers-lstm
jupyter notebook RNN_Airline_Passengers.ipynb
```

O ejecutar directamente en Google Colab sin instalación local:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andresfelipediaz/airline-passengers-lstm/blob/main/RNN_Airline_Passengers.ipynb)

---

## 🏗️ Arquitectura del Modelo

```
╔══════════════════════════════════════════════════════╗
║              ARQUITECTURA LSTM — AIRLINE             ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║   Input Layer        Shape: (12, 1)                  ║
║        │             Look-back: 12 meses             ║
║        ▼                                             ║
║   LSTM  128          return_sequences = True         ║
║        │             Captura patrones generales      ║
║   BatchNorm                                          ║
║   Dropout (0.30)     Regularización fuerte           ║
║        │                                             ║
║        ▼                                             ║
║   LSTM  64           return_sequences = False        ║
║        │             Refinamiento de representación  ║
║   BatchNorm                                          ║
║   Dropout (0.20)     Segunda regularización          ║
║        │                                             ║
║        ▼                                             ║
║   Dense (32, ReLU)   Transformación no lineal        ║
║        │                                             ║
║        ▼                                             ║
║   Dense (1)          Predicción escalar              ║
║                      pasajeros del mes siguiente     ║
╚══════════════════════════════════════════════════════╝
```

### Compuertas LSTM — Mecanismo interno

Cada celda LSTM regula el flujo de información a través de **tres compuertas** que resuelven el problema del gradiente desvanecido:

```
Forget Gate  →  ft = σ(Wf · [ht-1, xt] + bf)
                Decide qué información OLVIDAR del estado anterior

Input Gate   →  it = σ(Wi · [ht-1, xt] + bi)
                C~t = tanh(Wc · [ht-1, xt] + bc)
                Decide qué NUEVA información guardar en la celda

Output Gate  →  ot = σ(Wo · [ht-1, xt] + bo)
                ht = ot ⊙ tanh(Ct)
                Decide qué parte de la memoria exponer como SALIDA

Cell State   →  Ct = ft ⊙ Ct-1 + it ⊙ C~t
                ( ⊙ = producto elemento a elemento )
```

---

## 📦 Dataset

**Airline Passengers Dataset** — Box & Jenkins (1976)

| Atributo | Detalle |
|---|---|
| **Fuente** | [Jason Brownlee – Datasets (GitHub)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv) |
| **Período** | Enero 1949 — Diciembre 1960 |
| **Frecuencia** | Mensual |
| **Registros totales** | 144 observaciones |
| **Variable objetivo** | Miles de pasajeros aéreos internacionales |
| **Rango de valores** | 104 — 622 miles de pasajeros |
| **Características** | Tendencia creciente + estacionalidad anual pronunciada |

### Estadísticas descriptivas

```
Métrica           Valor
────────────────────────────────
count             144 observaciones
mean              280.30 pasajeros
std               119.97 pasajeros
min               104.00 pasajeros
25%               180.00 pasajeros
50% (mediana)     265.50 pasajeros
75%               360.50 pasajeros
max               622.00 pasajeros
```

El dataset presenta dos características que lo convierten en un **benchmark clásico**:

- **Tendencia multiplicativa creciente**: el volumen de pasajeros crece año tras año.
- **Estacionalidad anual**: pico en julio–agosto (verano) y mínimo en enero–febrero.

---

## 📁 Estructura del Repositorio

```
airline-passengers-lstm/
│
├── 📓 RNN_Airline_Passengers.ipynb      # Notebook principal — 10 bloques
│
├── 📊 outputs/
│   ├── exploracion_dataset.png          # EDA: serie, tendencia y estacionalidad
│   ├── curvas_entrenamiento.png         # Train/Val Loss y MAE por época
│   ├── predicciones_vs_reales.png       # Predicciones vs valores reales
│   ├── analisis_errores.png             # Scatter + distribución de residuos
│   └── pronostico_futuro.png            # Pronóstico 12 meses — año 1961
│
├── 💾 mejor_modelo_lstm.keras           # Mejor checkpoint del modelo
│
├── 📄 requirements.txt
└── 📄 README.md
```

### Estructura del Notebook

| Bloque | Sección | Descripción |
|---|---|---|
| 1 | Instalación e Importaciones | TensorFlow, Keras, NumPy, Pandas, Plotly, scikit-learn |
| 2 | Carga y EDA | Descarga del dataset, estadísticas y 3 visualizaciones exploratorias |
| 3 | Preprocesamiento | MinMaxScaler → escala [0, 1] |
| 4 | Secuencias Temporales | Función `create_sequences` con ventana deslizante |
| 5 | División Train/Test | Partición cronológica 80/20 con visualización |
| 6 | Arquitectura LSTM | `build_lstm_model()` — función modular y documentada |
| 7 | Entrenamiento | `model.fit()` con 3 callbacks inteligentes |
| 8 | Evaluación y Métricas | RMSE, MAE, MAPE, R² en train y test |
| 9 | Visualizaciones | 5 gráficos profesionales exportados a PNG |
| 10 | Pronóstico Futuro | Predicción autorregresiva de los 12 meses de 1961 |

---

## 🔄 Pipeline Completo

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PIPELINE DEL PROYECTO                          │
└─────────────────────────────────────────────────────────────────────┘

  ① CARGA DEL DATASET
     URL pública → pd.read_csv → índice datetime (formato YYYY-MM)
              │
              ▼
  ② ANÁLISIS EXPLORATORIO (EDA)
     Serie completa  /  Media móvil 12m  /  Estacionalidad por mes
              │
              ▼
  ③ PREPROCESAMIENTO
     MinMaxScaler → escala [0, 1]
     (evita saturación de tanh/sigmoid en las compuertas LSTM)
              │
              ▼
  ④ GENERACIÓN DE SECUENCIAS
     Ventana deslizante look_back = 12
     X.shape = (132, 12, 1)   ·   y.shape = (132,)
              │
              ▼
  ⑤ DIVISIÓN CRONOLÓGICA
     Train 80%  →  105 muestras  (1950–1958 aprox.)
     Test  20%  →   27 muestras  (1959–1960)
              │
              ▼
  ⑥ CONSTRUCCIÓN DEL MODELO
     LSTM(128) → BN → Dropout(0.3) → LSTM(64) → BN → Dropout(0.2)
     → Dense(32, ReLU) → Dense(1)
     Compilado con Adam lr=0.001 y loss=MSE
              │
              ▼
  ⑦ ENTRENAMIENTO CON CALLBACKS
     EarlyStopping      (patience=25, restore_best_weights=True)
     ReduceLROnPlateau  (factor=0.5, patience=10, min_lr=1e-6)
     ModelCheckpoint    (save_best_only=True → mejor_modelo_lstm.keras)
              │
              ▼
  ⑧ EVALUACIÓN
     Desnormalización → scaler.inverse_transform()
     Métricas: RMSE / MAE / MAPE / R²  (train y test)
              │
              ▼
  ⑨ VISUALIZACIONES
     Curvas de pérdida  /  Predicciones vs Real
     Scatter residuos   /  Distribución de errores
              │
              ▼
  ⑩ PRONÓSTICO FUTURO
     Predicción autorregresiva multistep → 12 meses de 1961
     Ventana deslizante sobre predicciones anteriores
```

---

## 📊 Resultados y Métricas

### Métricas en conjunto de prueba — 20% de los datos (1959–1960)

| Métrica | Valor | Descripción |
|---|---|---|
| **RMSE** | ~18.4 pasajeros | Raíz del error cuadrático medio — penaliza errores grandes |
| **MAE** | ~14.2 pasajeros | Error absoluto medio — interpretable en unidades originales |
| **R²** | ~0.962 | El modelo explica el **96.2%** de la varianza del conjunto de prueba |
| **MAPE** | ~5.3% | Error porcentual absoluto medio — alta precisión relativa |

> Un **R² de 0.962** indica que el modelo LSTM captura con alta fidelidad tanto la tendencia creciente como los ciclos estacionales, generalizando correctamente a datos no vistos durante el entrenamiento.

### Pronóstico 1961 — 12 meses futuros

| Mes | Pasajeros est. | | Mes | Pasajeros est. |
|---|---:|---|---|---:|
| Enero 1961 | **445** | | Julio 1961 | **648** |
| Febrero 1961 | **420** | | Agosto 1961 | **638** |
| Marzo 1961 | **472** | | Septiembre 1961 | **532** |
| Abril 1961 | **491** | | Octubre 1961 | **478** |
| Mayo 1961 | **535** | | Noviembre 1961 | **430** |
| Junio 1961 | **582** | | Diciembre 1961 | **460** |

> El pronóstico respeta el patrón estacional esperado: **pico en julio–agosto** y **mínimo en febrero**, validando que el modelo aprendió correctamente la estructura cíclica anual.

---

## 🖼️ Visualizaciones Generadas

El notebook exporta automáticamente **5 figuras** de alta resolución (150 DPI):

| # | Archivo | Contenido |
|---|---|---|
| 1 | `exploracion_dataset.png` | Serie temporal completa · Media móvil 12m ± desv. estándar · Promedio mensual por estacionalidad |
| 2 | `curvas_entrenamiento.png` | Train/Val Loss (MSE en escala log) y Train/Val MAE por época |
| 3 | `predicciones_vs_reales.png` | Predicciones de train y test superpuestas a la serie real, con métricas anotadas |
| 4 | `analisis_errores.png` | Scatter de predicho vs real + histograma de distribución de residuos |
| 5 | `pronostico_futuro.png` | Pronóstico 1961 con intervalo de confianza ±8% y valores anotados en cada punto |

---

## ⚙️ Instalación y Uso

### Requisitos previos

- Python 3.9 o superior
- pip
- Jupyter Notebook / JupyterLab

### 1. Clonar el repositorio

```bash
git clone https://github.com/pipediaz1234/RNN_Airline_Passengers/blob/main/RNN_Airline_Passengers.ipynb
cd airline-passengers-lstm
```

### 2. Crear y activar entorno virtual

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Lanzar el notebook

```bash
jupyter notebook RNN_Airline_Passengers.ipynb
```

### 5. Ejecutar en Google Colab (sin instalación)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andresfelipediaz/airline-passengers-lstm/blob/main/RNN_Airline_Passengers.ipynb)

> En Colab, la primera celda instala automáticamente `plotly` y `scikit-learn`. El dataset se descarga desde una URL pública sin necesidad de subir archivos.

### `requirements.txt`

```txt
tensorflow>=2.10.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.1.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0
jupyter>=1.0.0
```

---

## 🎛️ Hiperparámetros

| Parámetro | Valor | Justificación |
|---|---|---|
| `LOOK_BACK` | 12 meses | Captura un ciclo estacional anual completo |
| `LSTM_1 units` | 128 | Capacidad suficiente para extraer patrones generales |
| `LSTM_2 units` | 64 | Refinamiento con dimensionalidad reducida |
| `Dropout 1` | 0.30 | Regularización moderada post-LSTM 1 |
| `Dropout 2` | 0.20 | Regularización leve post-LSTM 2 |
| `Dense units` | 32 | Transformación no lineal antes de la salida |
| `Épocas máx.` | 150 | Límite superior con EarlyStopping activo |
| `Batch size` | 16 | Balance óptimo velocidad / estabilidad del gradiente |
| `Optimizador` | Adam | Adaptativo, robusto para RNN con gradientes variables |
| `Learning rate` | 0.001 | Valor estándar y probado para Adam |
| `Val split` | 15% | Monitoreo de generalización durante entrenamiento |
| `Patience ES` | 25 épocas | Margen para superar mesetas de aprendizaje |
| `ReduceLR factor` | 0.5 | Reduce el lr a la mitad al estancarse |
| `ReduceLR patience` | 10 épocas | Reacción rápida ante mesetas intermedias |
| `Loss function` | MSE | Penaliza errores grandes — estándar en regresión |

---

## 🧠 Conceptos Clave

<details>
<summary><strong>¿Qué es una RNN y por qué es ideal para series temporales?</strong></summary>

<br>

Una **Red Neuronal Recurrente (RNN)** procesa datos secuenciales manteniendo un **estado oculto** `ht` que actúa como memoria del historial. A diferencia de una red densa tradicional, la RNN comparte parámetros a lo largo del tiempo y su salida en cada paso `t` depende tanto de la entrada actual `xt` como del estado previo `ht-1`:

```
ht = tanh(Wh · ht-1 + Wx · xt + b)
yt = Wy · ht
```

Esto la hace ideal para series temporales, donde el **orden de las observaciones importa** y los valores pasados contienen información predictiva que no puede ignorarse.

</details>

<details>
<summary><strong>¿Por qué LSTM en lugar de RNN simple?</strong></summary>

<br>

Las RNN simples sufren el problema del **gradiente desvanecido**: durante el entrenamiento mediante BPTT (*Backpropagation Through Time*), los gradientes se multiplican en cada paso temporal. Con pesos menores que 1, el gradiente decrece exponencialmente hacia las capas tempranas, que dejan de aprender.

La **LSTM** resuelve esto mediante tres compuertas que regulan explícitamente qué información se conserva, actualiza o expone en cada paso. Esto le permite capturar **dependencias a largo plazo** esenciales para series con estacionalidad anual.

</details>

<details>
<summary><strong>¿Cómo funciona la ventana deslizante (look-back)?</strong></summary>

<br>

La serie se transforma en pares de entrenamiento supervisado mediante una **ventana deslizante**:

```
[Mes 1 … Mes 12]  →  predice  →  Mes 13
[Mes 2 … Mes 13]  →  predice  →  Mes 14
[Mes 3 … Mes 14]  →  predice  →  Mes 15
        ·
        ·
[Mes 132 … Mes 143]  →  predice  →  Mes 144
```

Con `LOOK_BACK = 12`, la LSTM recibe un año completo de historia en cada paso, capturando automáticamente el **patrón estacional anual** de la serie.

</details>

<details>
<summary><strong>¿Por qué normalizar con MinMaxScaler?</strong></summary>

<br>

Las funciones de activación internas de la LSTM —`tanh` y `sigmoid`— operan en rangos acotados. Con valores grandes (104–622 pasajeros) estas funciones se **saturan**, el gradiente desaparece y el modelo deja de aprender.

Al escalar todos los valores al rango `[0, 1]`, los datos permanecen en la zona activa de estas funciones, facilitando la **convergencia del gradiente** y mejorando la calidad de las predicciones.

</details>

<details>
<summary><strong>¿Para qué sirven los callbacks EarlyStopping y ReduceLROnPlateau?</strong></summary>

<br>

- **EarlyStopping** (`patience=25`): monitorea la pérdida de validación y detiene el entrenamiento si no mejora durante 25 épocas, restaurando los pesos del mejor epoch. Evita el **overfitting**.

- **ReduceLROnPlateau** (`factor=0.5`, `patience=10`): cuando el modelo se estanca, reduce el learning rate a la mitad, permitiendo salir de mínimos locales sin sobrepasar el óptimo.

- **ModelCheckpoint**: guarda automáticamente el modelo con menor `val_loss`, garantizando que siempre se conserve la mejor versión encontrada.

</details>

---

## 🛠️ Tecnologías

<div align="center">

| Librería | Versión | Rol en el proyecto |
|---|---|---|
| **Python** | 3.9+ | Lenguaje base del proyecto |
| **TensorFlow / Keras** | 2.x | Construcción, entrenamiento y exportación del modelo LSTM |
| **NumPy** | 1.23+ | Operaciones matriciales, reshape de arrays 3D para LSTM |
| **Pandas** | 1.5+ | Carga del CSV, indexado temporal, manipulación del DataFrame |
| **scikit-learn** | 1.1+ | MinMaxScaler, MSE, R², MAE |
| **Matplotlib** | 3.6+ | Gráficos estáticos de serie, curvas de entrenamiento y errores |
| **Seaborn** | 0.12+ | Estilos visuales y paleta de colores |
| **Plotly** | 5.x | Visualizaciones interactivas opcionales |

</div>

---

## 🎓 Contexto Académico

Este proyecto fue desarrollado por **Andrés Felipe Díaz Campos** como práctica de laboratorio para la asignatura **Electiva Profesional II – Deep Learning** del programa de **Ingeniería de Computación** de la **Universidad de Cundinamarca, Sede Facatativá (2024)**.

### Competencias demostradas

| Área | Competencias |
|---|---|
| **Arquitecturas DL** | Diseño e implementación de LSTM bicapa con regularización |
| **Series Temporales** | Preprocesamiento, ventana deslizante, división cronológica |
| **Ingeniería de Features** | Normalización MinMax, secuencias 3D para LSTM |
| **Entrenamiento** | Callbacks inteligentes, ajuste de hiperparámetros, checkpoint |
| **Evaluación** | RMSE, MAE, MAPE, R², análisis de residuos |
| **Visualización** | 5 gráficos profesionales con Matplotlib y Seaborn |
| **Código limpio** | Funciones documentadas con docstrings y comentarios técnicos |

### Temas del curso cubiertos

- ✅ Arquitectura y funcionamiento de las Redes Neuronales Recurrentes (RNN)
- ✅ Mecanismo de compuertas de la LSTM (Forget, Input, Output Gate)
- ✅ Problema del gradiente desvanecido y sus soluciones modernas
- ✅ Procesamiento de series temporales con ventana deslizante
- ✅ Regularización: Dropout y Batch Normalization
- ✅ Evaluación de modelos de regresión con métricas estándar
- ✅ Pronóstico autorregresivo multistep

---

## 📄 Licencia

Este proyecto está distribuido bajo la licencia **MIT**. Puedes usarlo, modificarlo y distribuirlo libremente citando al autor original.

```
MIT License — Copyright (c) 2024 Andrés Felipe Díaz Campos
```

Consulta el archivo [LICENSE](LICENSE) para los términos completos.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0F1B2D&height=130&section=footer&text=Andrés%20Felipe%20Díaz%20Campos&fontSize=22&fontColor=06B6D4&fontAlignY=65" width="100%"/>

**Desarrollado con ❤️ por Andrés Felipe Díaz Campos**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/andres-felipe-diaz-campos-398245207)

*Ingeniería de Computación · Universidad de Cundinamarca · UDEC Facatativá · 2026*

<br>

⭐ **Si este proyecto te fue útil, considera darle una estrella en GitHub.**

</div>
