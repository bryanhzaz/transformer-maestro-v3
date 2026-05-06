# Generación Musical Predictiva con Transformers

Este repositorio contiene una suite completa de algoritmos diseñados para la generación automática de música (formato MIDI) mediante el análisis y modelado de series de tiempo. Utilizando el conjunto de datos *MAESTRO v3.0.0, el proyecto explora modelos con arquitecturas profundas basadas en **Transformers Multi-Característica**.

## Arquitectura del Proyecto

El repositorio está estructurado en tres fases principales:

### 1. Exploración y Análisis Estadístico (/analisis_SerieDeTiempo)
Contiene cuadernos de Jupyter enfocados en el Análisis Exploratorio de Datos (EDA) de las series de tiempo musicales.
- Visualización de secuencias de notas (pitch) y sus medias globales/segmentadas.
- Evaluación de estacionariedad en la música utilizando la *Prueba Aumentada de Dickey-Fuller (ADF)*.
- Comparación visual y estadística (Boxplots/Stripplots) entre diversas pistas del dataset MAESTRO.

### 2. Entrenamiento Distribuido en Clúster (/entrenamientoCluster)
Implementación de un *Transformer* para la generación musical, optimizado para entrenamiento en entornos con GPU.
* **entrenamientoBase.py**: Script de entrenamiento que extrae y procesa **8 características musicales* por nota:
  * Discretas: Pitch (mediante Embeddings).
  * Continuas: Step, Duration, Velocity, Sustain, Chord Size, Tempo, y Beat Position (mediante capas densas).
  * Evalúa combinaciones de funciones de pérdida, incluyendo Error Cuadrático Medio (MSE) con presión positiva para tiempos, y entropía cruzada para variables discretas.
* *entrenamientoContinuacion.py*: Utilidad para reanudar el entrenamiento a partir de checkpoints, ajustar de forma dinámica la tasa de aprendizaje y concatenar históricos de pérdida sin romper la secuencia de métricas.

### 3. Evaluación y Validación de Hipótesis (/resultadosAnalisisEstadistico)
Un riguroso motor de validación para comparar matemáticamente las composiciones generadas por el modelo contra la música compuesta por humanos (Test Set).
* *analisisEstad.py*: Realiza inferencia masiva y evalúa la divergencia de las secuencias generadas utilizando:
  * Prueba de Kolmogorov-Smirnov (KS).
  * Distancia de Wasserstein.
  * Divergencia de Kullback-Leibler (KL) para métricas de tono.
* *resultadosGenerales*: Artefactos de salida resultantes, incluyendo gráficas combinadas de pérdida, consolidados de métricas NLP (como *Perplexity y BLEU Scores), y audios MIDI de demostración generados de manera autónoma.

## Tecnologías y Librerías Utilizadas

- *Core & Data:* Python 3, Pandas, NumPy
- *Machine Learning & Deep Learning:* TensorFlow / Keras (Multi-GPU), Scikit-learn, Statsmodels
- *Análisis Matemático y NLP:* SciPy (Estadística de distribuciones), NLTK (BLEU Score)
- *Audio Processing:* pretty_midi, mido
- *Visualización:* Matplotlib, Seaborn
  
## Uso y Reproducción

### Requisitos Previos
Asegúrate de descargar y descomprimir el dataset [MAESTRO v3.0.0](https://magenta.withgoogle.com/datasets/maestro) en el directorio raíz del proyecto:
```bash
.
├── maestro-v3.0.0/
│   ├── maestro-v3.0.0.csv
│   └── 2004/ ...
├── entrenamientoCluster/
└── ...
```
## Resultados Destacados
Tras 100 épocas de entrenamiento con la arquitectura extendida del Transformer, el modelo logró optimizar el aprendizaje de características melódicas y rítmicas, obteniendo:

*Pitch Accuracy: 0.5387
BLEU-1 Score: 0.6073
Un empate casi perfecto en las medias distribucionales, por ejemplo, la velocidad de las teclas (dinámica de presión) entre humano y máquina (Original: 0.5035 vs Generado: 0.5038)*.
(El archivo .keras conteniendo los pesos del modelo final ha sido omitido del control de versiones por limitaciones de almacenamiento, pero la arquitectura es 100% reproducible).
