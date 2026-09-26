---
title: Riesgo de incumplimiento en préstamos de Lending Club
subtitle: scikit-learn frente a PySpark — modelado, comparación estadística e interpretabilidad
short_title: Presentación
---

Proyecto integrador de aprendizaje automático · Machine Learning UN 202330

## Resumen

Este trabajo construye y compara modelos de clasificación supervisada para predecir si un préstamo otorgado por la plataforma Lending Club terminará en incumplimiento (`Charged Off`) o será pagado por completo (`Fully Paid`). Se utiliza el conjunto de datos completo de préstamos aceptados entre 2007 y 2018, sin ningún tipo de muestreo: 2,26 millones de registros, de los cuales 1,35 millones tienen un desenlace conocido.

Se implementan seis modelos en dos entornos, con espacios de búsqueda equivalentes y una misma partición de entrenamiento y prueba: regresión logística, árbol de decisión, bosque aleatorio, gradient boosting, máquina de vectores de soporte lineal y Naive Bayes, en **scikit-learn** y en **PySpark**. Las diferencias de desempeño se contrastan con la prueba de DeLong para AUC correlacionados, en su versión rápida de Sun y Xu (2014), y se verifican con las pruebas de McNemar y bootstrap pareado, con corrección de Holm por comparaciones múltiples. Por último, las predicciones erróneas se interpretan con LIME.

El mejor modelo en ambos entornos es **gradient boosting** (AUC de 0,720), con una ventaja significativa y relevante sobre la regresión logística. Los dos entornos alcanzan desempeños estadísticamente equivalentes en los modelos bien especificados, y scikit-learn resulta en general más rápido en un único equipo. La excepción es la máquina de vectores de soporte, cuyo resultado depende críticamente de la formulación del optimizador.

## Datos y objetivo

| Aspecto | Descripción |
|---|---|
| Fuente | [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club), archivo `accepted_2007_to_2018Q4.csv` (1,7 GB) |
| Universo | 1.345.310 préstamos con desenlace conocido (sin muestreo) |
| Variable objetivo | `default` = 1 si `loan_status` es `Charged Off` (19,96 %), 0 si es `Fully Paid` |
| Partición | 80/20 estratificada, semilla 42, común a ambos entornos (1.076.248 / 269.062) |
| Predictores | 16 numéricos, 3 indicadores y 7 categóricos disponibles al originar el préstamo (91 columnas tras la codificación) |

## Resultados principales

| Modelo | AUC scikit-learn | AUC PySpark | Entrenamiento scikit-learn | Entrenamiento PySpark |
|---|---|---|---|---|
| **Gradient boosting** | **0,7198** | **0,7195** | 125,1 min | 80,8 min |
| Bosque aleatorio | 0,7151 | 0,7137 | 25,7 min | 105,1 min |
| Regresión logística | 0,7135 | 0,7135 | 0,4 min | 11,0 min |
| Árbol de decisión | 0,7029 | 0,7002 | 2,3 min | 3,7 min |
| Naive Bayes | 0,6390 | 0,6395 | 0,2 min | 2,9 min |
| SVM lineal (hinge) | 0,4611 | 0,6300 | 23,5 min | 21,1 min |

AUC calculado en el conjunto de prueba común. El tiempo de entrenamiento incluye la validación cruzada de 3 pliegues y el reajuste final.

:::{note} Conclusiones
1. Gradient boosting es el mejor modelo en ambos entornos y con los mismos hiperparámetros. Entre sus dos implementaciones no hay una diferencia significativa de AUC ($p_{Holm} = 0{,}33$).
2. Entre entornos, las diferencias de los modelos de árboles son estadísticamente significativas pero inferiores al margen de relevancia de 0,005. La única diferencia relevante es la de la SVM lineal, atribuible al optimizador: el dual de liblinear frente al primal de Spark.
3. DeLong, el bootstrap pareado y McNemar coinciden en todas las comparaciones de AUC. Las discrepancias aparecen solo en el AUC-PR y el F1, las métricas centradas en la clase minoritaria.
4. scikit-learn completó el flujo en 178 minutos y PySpark en 229. Con datos que caben en memoria, Spark solo aventaja a scikit-learn en los algoritmos basados en árboles profundos.
:::

## Estructura del libro

1. **Exploración de datos.** Análisis univariado, bivariado y de valores faltantes; identificación de variables con fuga de información y de redundancias.
2. **Preprocesamiento.** Partición común y transformaciones equivalentes en scikit-learn y PySpark, con verificación numérica de la equivalencia.
3. **Modelado con scikit-learn.** `GridSearchCV` con validación cruzada de 3 pliegues, métricas en entrenamiento y prueba, y diagnóstico de LinearSVC.
4. **Modelado con PySpark.** `CrossValidator` con `ParamGridBuilder` y efecto de las condiciones de ejecución en el rendimiento.
5. **Comparación estadística.** DeLong entre entornos y entre modelos, McNemar y bootstrap pareado.
6. **Interpretabilidad con LIME.** Explicación de préstamos mal clasificados en ambos entornos.
7. **Comparación de resultados y reflexión crítica.** Tablas comparativas, experimento de escalabilidad, modelo ganador y respuesta a las preguntas del enunciado.

## Reproducibilidad

Todos los cálculos se ejecutaron en un único equipo con procesador Intel de 12.ª generación (10 núcleos físicos, 12 lógicos), 16 GB de RAM y Windows 11 (build 26200; Python lo reporta como "Windows 10"). Se usaron Python 3.11.16, scikit-learn 1.9.1, pandas 2.2.3 y PySpark 3.5.5 en modo `local[*]`. El código compartido por los capítulos está en `lc_utils.py` (configuración, preprocesamiento y métricas) y `lc_stats.py` (pruebas estadísticas), y el repositorio incluye un notebook compilado con todos los capítulos y sus salidas (`proyecto_lending_club.ipynb`).
