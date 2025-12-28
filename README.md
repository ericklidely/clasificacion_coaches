# Segmentación de Coaches (Clustering no supervisado)

Este proyecto implementa un pipeline de preparación de datos y clustering no supervisado para segmentar coaches en grupos (ej. Junior / Mid-level / Senior) a partir de variables numéricas, categóricas y embeddings de texto.

## Objetivo
- Comprender la población de coaches y detectar segmentos naturales.
- Generar clusters interpretables y accionables (sin usar un target).
- Dejar un pipeline reproducible (limpieza → features → embeddings → PCA → clustering → reportes/figuras).

## Enfoque general
1) **Limpieza**
- Normalización de nombres de columnas (minúsculas, sin acentos, con `_`)
- Limpieza de texto (lowercase, strip, unidecode)
- Conversión a numéricos / fechas (incluye serial de Excel)
- Imputación de nulos (mediana para numéricas, `sin_dato` para categóricas)
- Escalado de variables numéricas (StandardScaler)

2) **Features categóricas**
- One Hot Encoding para columnas seleccionadas (se eliminan las originales)

3) **Embeddings**
- Construcción de un texto de perfil (ej. “Tipos de industria…”)
- Generación de embeddings con:
  - `sentence_transformers` (local) **o**
  - `openai` embeddings

4) **Reducción dimensional**
- PCA para reducir embeddings y/o el set completo antes de KMeans

5) **Clustering**
- KMeans con selección de K (silhouette / elbow)
- Interpretación de clusters con estadísticas descriptivas

6) **Visualizaciones**
- PCA 2D/3D de clusters
- Heatmaps de distribución por industria u otras variables
- Export a PNG/HTML (sin exponer identificadores sensibles)

## Estructura sugerida del repo
