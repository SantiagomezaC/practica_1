# Riesgo de incumplimiento en préstamos de Lending Club

Proyecto integrador de aprendizaje automático (sección 9.10 del curso). Se predice si un préstamo de Lending Club termina en *default* (`Charged Off`) o se paga por completo (`Fully Paid`) con seis modelos implementados en **scikit-learn** y en **PySpark**. Las diferencias de desempeño se contrastan con las pruebas de DeLong, McNemar y bootstrap pareado, y las predicciones se interpretan con LIME.

- **Jupyter Book publicado:** https://santiagomezac.github.io/practica_1/
- **Notebook compilado (todos los capítulos, con salidas):** [`proyecto_lending_club.ipynb`](proyecto_lending_club.ipynb)

## Estructura

| Archivo | Contenido |
|---|---|
| `docs/intro.md` | Portada del libro: objetivo, datos, metodología y resultados principales |
| `docs/01_eda.ipynb` | Exploración de datos univariada, bivariada y de valores faltantes |
| `docs/02_preprocesamiento.ipynb` | Partición común 80/20 y preprocesamiento equivalente en scikit-learn y PySpark |
| `docs/03_modelado_sklearn.ipynb` | `GridSearchCV` (3 pliegues, ROC AUC) de los seis modelos en scikit-learn |
| `docs/04_modelado_pyspark.ipynb` | `CrossValidator` + `ParamGridBuilder` de los seis modelos en PySpark |
| `docs/05_comparacion_estadistica.ipynb` | DeLong rápido (Sun y Xu, 2014), McNemar, bootstrap pareado y corrección de Holm |
| `docs/06_lime.ipynb` | Interpretabilidad local con LIME en ambos entornos |
| `docs/07_comparacion_resultados.ipynb` | Tablas comparativas, escalabilidad y reflexión crítica |
| `docs/lc_utils.py` | Configuración, preprocesamiento (scikit-learn y Spark) y métricas compartidas |
| `docs/lc_stats.py` | Implementación de las pruebas estadísticas |
| `docs/myst.yml` | Configuración e índice del libro (MyST) |

El CSV original (`accepted_2007_to_2018Q4.csv`, 1,7 GB) no se incluye en el repositorio. Los notebooks lo buscan en la carpeta `Tarea 1`, al mismo nivel que este repositorio. Los artefactos intermedios (partición común, puntuaciones y resultados) se escriben en `data/`, que git ignora.

## Reproducción

```powershell
conda env create -f environment.yml
conda activate ml-lending
python -m ipykernel install --user --name ml-lending
```

Los capítulos deben ejecutarse en orden, cada uno en su propio kernel. Los capítulos 3 y 4 entrenan con validación cruzada sobre el dataset completo y tardan varias horas en un equipo de 12 núcleos lógicos y 16 GB de RAM. PySpark requiere Java 11 o 17.

Para compilar el libro:

```powershell
cd docs
myst build --html     # sitio estático en docs/_build/html
myst start            # vista previa local
```

## Criterios metodológicos

- Partición estratificada 80/20 con semilla 42, común a ambos entornos (`data/split.parquet`).
- Transformadores ajustados solo con el conjunto de entrenamiento.
- Selección de hiperparámetros por ROC AUC con validación cruzada de 3 pliegues.
- Umbral de decisión fijado con datos de entrenamiento (cuantil 1 − π de las puntuaciones).
- Corrección de Holm por familia de comparaciones, con α = 0,05.
- Margen de relevancia práctica definido de antemano: |ΔAUC| ≥ 0,005.
- En PySpark no se transfiere el dataset al driver; solo se transfieren las puntuaciones del conjunto de prueba.
