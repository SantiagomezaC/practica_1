"""Configuración y utilidades compartidas por los notebooks del proyecto Lending Club.

Todos los notebooks importan este módulo para garantizar que usan las mismas rutas,
la misma semilla, las mismas variables y las mismas transformaciones.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
DATA_DIR = REPO_DIR / "data"  # artefactos intermedios (ignorado por git)
DATA_DIR.mkdir(exist_ok=True)

RAW_CSV = REPO_DIR.parent / "Tarea 1" / "accepted_2007_to_2018Q4.csv"
EDA_PARQUET = DATA_DIR / "lc_eda.parquet"      # filas Fully Paid / Charged Off + variables analizadas
SPLIT_PARQUET = DATA_DIR / "split.parquet"     # partición común: columnas id, split

# Spark usa '\' como escape por defecto; el CSV sigue RFC 4180 (comillas dobladas ""),
# igual que pandas. Sin esta opción Spark desplaza las columnas del préstamo 61400928.
SPARK_CSV_OPTIONS = {"header": True, "escape": '"'}

SEED = 42
TARGET = "default"
STATUS_KEEP = {"Fully Paid": 0, "Charged Off": 1}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
# Columnas del CSV original que se cargan para el análisis (disponibles al originar el préstamo).
RAW_COLS = [
    "id", "loan_status", "issue_d",
    "loan_amnt", "term", "int_rate", "installment", "grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "purpose", "addr_state",
    "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "initial_list_status", "application_type",
    "mort_acc", "pub_rec_bankruptcies",
]

NUM_COLS = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "fico_range_low", "fico_range_high", "emp_length", "credit_hist_years",
    "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
]

CAT_COLS = [
    "term", "grade", "home_ownership", "verification_status", "purpose",
    "addr_state", "initial_list_status", "application_type",
]

# Columnas con información posterior a la originación del préstamo (fuga de información).
LEAKAGE_COLS = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low", "pymnt_plan", "debt_settlement_flag",
    "debt_settlement_flag_date", "settlement_status", "settlement_date",
    "settlement_amount", "settlement_percentage", "settlement_term",
]

# ---------------------------------------------------------------------------
# Variables del modelo (decisiones del resumen ejecutivo del EDA, sección 1.6)
# ---------------------------------------------------------------------------
MODEL_NUM = [
    "loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_high", "emp_length",
    "credit_hist_years", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
]
INDICATOR_COLS = ["has_delinq_history", "emp_length_missing", "mort_acc_missing"]
MODEL_CAT = [
    "term", "home_ownership", "verification_status", "purpose", "addr_state",
    "initial_list_status", "application_type",
]
CLIP_COLS = ["dti", "revol_util"]
CLIP_Q = (0.001, 0.999)
LOG_COLS = ["annual_inc", "revol_bal"]
IMPUTE_COLS = ["emp_length", "mort_acc", "revol_util", "pub_rec_bankruptcies", "dti", "inq_last_6mths"]
HOME_OTHER = ["ANY", "NONE", "OTHER"]
RARE_PURPOSE = ["moving", "vacation", "house", "wedding", "renewable_energy", "educational"]
SCALED_COLS = MODEL_NUM + INDICATOR_COLS  # entran al StandardScaler; las dummies no se escalan

# Modelos: nombre corto usado en tablas, archivos y columnas de puntuaciones.
MODELS = ["LogReg", "DecisionTree", "RandomForest", "GradBoost", "LinearSVC", "NaiveBayes"]
REG_PARAMS = [1e-6, 1e-5, 1e-4]  # regParam de PySpark; en scikit-learn C = 1 / (regParam * n_train)
SCORES_SK = DATA_DIR / "scores_sklearn.parquet"
SCORES_SPARK = DATA_DIR / "scores_pyspark.parquet"
RESULTS_SK = DATA_DIR / "results_sklearn.json"
RESULTS_SPARK = DATA_DIR / "results_pyspark.json"
MODELS_SK_DIR = DATA_DIR / "models_sklearn"
SPARK_PARTIAL_DIR = DATA_DIR / "pyspark_by_model"   # resultados de PySpark guardados modelo a modelo

# Particiones de los DataFrames de modelado en PySpark: una por núcleo lógico (ver sección 4.6).
# La unión con la partición común hereda spark.sql.shuffle.partitions=400; con 12 núcleos, 400
# tareas diminutas por pasada multiplican el costo de los árboles y del CrossValidator.
SPARK_MODEL_PARTITIONS = 12


# ---------------------------------------------------------------------------
# Transformaciones
# ---------------------------------------------------------------------------
def parse_emp_length(s: pd.Series) -> pd.Series:
    """'< 1 year' -> 0, '10+ years' -> 10, 'n years' -> n; faltantes quedan NaN."""
    s = s.astype("string").str.strip()
    out = s.str.extract(r"(\d+)")[0].astype("float")
    return out.where(~s.str.startswith("<", na=False), 0.0)


def build_eda_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Filtra Fully Paid / Charged Off, crea el target y las variables derivadas."""
    df = raw[raw["loan_status"].isin(list(STATUS_KEEP))].copy()
    df[TARGET] = df["loan_status"].apply(lambda x: 1 if x == "Charged Off" else 0).astype("int8")
    df["id"] = pd.to_numeric(df["id"]).astype("int64")
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["credit_hist_years"] = (df["issue_d"] - earliest).dt.days / 365.25
    df["emp_length"] = parse_emp_length(df["emp_length"])
    df["term"] = df["term"].str.strip()
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    cols = ["id", "issue_d", TARGET] + NUM_COLS + CAT_COLS
    return df[cols].reset_index(drop=True)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér con corrección de sesgo (Bergsma, 2013)."""
    from scipy.stats import chi2_contingency

    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c = r - (r - 1) ** 2 / (n - 1)
    k_c = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(k_c - 1, r_c - 1))))


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def prevalence_threshold(train_scores, prevalence: float) -> float:
    """Umbral con el que la tasa de positivos predichos en train iguala la prevalencia de default en train."""
    return float(np.quantile(train_scores, 1 - prevalence))


def binary_metrics(y, score, threshold: float) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    pred = (np.asarray(score) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
    }


# ---------------------------------------------------------------------------
# Partición común y preprocesamiento en pandas / scikit-learn
# ---------------------------------------------------------------------------
def load_model_frame() -> pd.DataFrame:
    """Lee el CSV completo con pandas y devuelve id, target, variables del modelo e indicadores."""
    raw = pd.read_csv(RAW_CSV, usecols=RAW_COLS, low_memory=False)
    df = build_eda_frame(raw)
    del raw
    out = df[["id", TARGET] + MODEL_NUM].copy()
    out["has_delinq_history"] = df["mths_since_last_delinq"].notna().astype("float64")
    out["emp_length_missing"] = df["emp_length"].isna().astype("float64")
    out["mort_acc_missing"] = df["mort_acc"].isna().astype("float64")
    for c in MODEL_CAT:
        out[c] = df[c].astype(str)
    out.loc[out["home_ownership"].isin(HOME_OTHER), "home_ownership"] = "OTHER"
    out.loc[out["purpose"].isin(RARE_PURPOSE), "purpose"] = "other"
    return out


def make_split(df: pd.DataFrame) -> pd.DataFrame:
    """Partición 80/20 estratificada por clase, con semilla fija. Devuelve columnas id, split."""
    from sklearn.model_selection import train_test_split

    id_train, id_test = train_test_split(
        df["id"], test_size=0.2, stratify=df[TARGET], random_state=SEED)
    return pd.concat([
        pd.DataFrame({"id": id_train.to_numpy(), "split": "train"}),
        pd.DataFrame({"id": id_test.to_numpy(), "split": "test"}),
    ], ignore_index=True)


def exact_quantile(x: pd.Series, q):
    """Cuantil como elemento de la muestra (inverted CDF), igual que approxQuantile de Spark con error 0."""
    return np.quantile(x.dropna().to_numpy(), q, method="inverted_cdf")


class SkPreprocessor:
    """Recorte -> log1p -> imputación por mediana -> StandardScaler (numéricas) + OneHotEncoder.

    Todos los parámetros se aprenden en `fit` con el conjunto de entrenamiento.
    """

    def fit(self, df: pd.DataFrame):
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        self.clip_ = {c: tuple(exact_quantile(df[c], CLIP_Q)) for c in CLIP_COLS}
        num = self._clip_log(df)
        self.median_ = {c: float(exact_quantile(num[c], 0.5)) for c in IMPUTE_COLS}
        self.scaler_ = StandardScaler().fit(num.fillna(self.median_))
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.ohe_.fit(df[MODEL_CAT])
        self.feature_names_ = SCALED_COLS + list(self.ohe_.get_feature_names_out(MODEL_CAT))
        return self

    def _clip_log(self, df: pd.DataFrame) -> pd.DataFrame:
        num = df[SCALED_COLS].astype("float64")
        for c, (lo, hi) in self.clip_.items():
            num[c] = num[c].clip(lo, hi)
        for c in LOG_COLS:
            num[c] = np.log1p(num[c])
        return num

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = self.scaler_.transform(self._clip_log(df).fillna(self.median_)).astype(np.float32)
        return np.hstack([num, self.ohe_.transform(df[MODEL_CAT])])


def sk_train_test() -> dict:
    """CSV -> partición común (split.parquet) -> SkPreprocessor ajustado en train -> matrices train/test."""
    df = load_model_frame().merge(pd.read_parquet(SPLIT_PARQUET), on="id", validate="one_to_one")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    prep = SkPreprocessor().fit(train)
    return {
        "X_train": prep.transform(train), "X_test": prep.transform(test),
        "y_train": train[TARGET].to_numpy(), "y_test": test[TARGET].to_numpy(),
        "id_train": train["id"].to_numpy(), "id_test": test["id"].to_numpy(),
        "prep": prep, "raw_train": train, "raw_test": test,
    }


# ---------------------------------------------------------------------------
# Preprocesamiento en PySpark (equivalente a SkPreprocessor)
# ---------------------------------------------------------------------------
def spark_session(app: str = "LendingClub_Optimized"):
    """SparkSession con la configuración obligatoria del enunciado (sección 9.10.4.5)."""
    import os
    import sys

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    from pyspark.sql import SparkSession

    spark = (SparkSession.builder.appName(app)
             .config("spark.sql.shuffle.partitions", "400")
             .config("spark.default.parallelism", "400")
             .config("spark.executor.memory", "8g")
             .config("spark.driver.memory", "8g")
             .config("spark.memory.fraction", 0.8)
             .config("spark.memory.storageFraction", 0.3)
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def spark_shutdown(spark):
    """Detiene la sesión y termina el proceso de la JVM, para que la siguiente `spark_session()` arranque
    una JVM nueva (con toda su memoria libre y con `spark.driver.memory` aplicado de nuevo).

    Cerrar solo el gateway de Py4J deja viva la JVM (con sus ~9 GB de memoria comprometida) hasta que
    termina el proceso de Python; con varias sesiones seguidas, Windows se queda sin memoria virtual.
    """
    import subprocess
    import sys

    from pyspark import SparkContext

    spark.stop()
    gateway = SparkContext._gateway
    if gateway is not None:
        proc = getattr(gateway, "proc", None)   # Popen de spark-submit creado por launch_gateway
        gateway.shutdown()
        if proc is not None and proc.poll() is None:
            if sys.platform == "win32":          # spark-submit.cmd -> cmd -> java: hay que matar el árbol
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            else:
                proc.kill()
            proc.wait(timeout=60)
        SparkContext._gateway = None
        SparkContext._jvm = None


def spark_load_model_frame(spark):
    """Lee el CSV completo con Spark y replica `load_model_frame`, unido a la partición común."""
    from pyspark.sql import functions as F

    raw = spark.read.csv(str(RAW_CSV), **SPARK_CSV_OPTIONS)
    df = raw.filter(F.col("loan_status").isin(list(STATUS_KEEP)))

    emp = F.trim(F.col("emp_length"))
    emp_years = F.when(emp.startswith("<"), F.lit(0.0)).otherwise(
        F.regexp_extract(emp, r"(\d+)", 1).cast("double"))
    issue = F.to_date(F.col("issue_d"), "MMM-yyyy")
    earliest = F.to_date(F.col("earliest_cr_line"), "MMM-yyyy")
    plain_num = [c for c in MODEL_NUM if c not in ("emp_length", "credit_hist_years")]

    df = df.select(
        F.col("id").cast("long").alias("id"),
        F.when(F.col("loan_status") == "Charged Off", 1).otherwise(0).alias(TARGET),
        *[F.col(c).cast("double").alias(c) for c in plain_num],
        emp_years.alias("emp_length"),
        (F.datediff(issue, earliest) / 365.25).alias("credit_hist_years"),
        F.col("mths_since_last_delinq").isNotNull().cast("double").alias("has_delinq_history"),
        F.trim(F.col("term")).alias("term"),
        F.when(F.col("home_ownership").isin(HOME_OTHER), "OTHER")
         .otherwise(F.col("home_ownership")).alias("home_ownership"),
        F.when(F.col("purpose").isin(RARE_PURPOSE), "other")
         .otherwise(F.col("purpose")).alias("purpose"),
        *[F.col(c) for c in ["verification_status", "addr_state", "initial_list_status", "application_type"]],
    )
    df = (df.withColumn("emp_length_missing", F.col("emp_length").isNull().cast("double"))
            .withColumn("mort_acc_missing", F.col("mort_acc").isNull().cast("double")))
    split = spark.read.parquet(str(SPLIT_PARQUET))
    return df.join(split, on="id", how="inner")


def spark_clip_log(df, bounds: dict):
    from pyspark.sql import functions as F

    for c, (lo, hi) in bounds.items():
        col = F.col(c)
        df = df.withColumn(c, F.when(col < lo, lo).when(col > hi, hi).otherwise(col))
    for c in LOG_COLS:
        df = df.withColumn(c, F.log1p(F.col(c)))
    return df


def spark_fit_preprocessor(train):
    """Ajusta con el conjunto de entrenamiento el Pipeline equivalente a SkPreprocessor."""
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler

    bounds = dict(zip(CLIP_COLS, [tuple(b) for b in train.approxQuantile(CLIP_COLS, list(CLIP_Q), 0.0)]))
    imputed = [c + "_imp" for c in IMPUTE_COLS]
    scaled_in = [c + "_imp" if c in IMPUTE_COLS else c for c in SCALED_COLS]
    stages = [
        Imputer(strategy="median", relativeError=0.0, inputCols=IMPUTE_COLS, outputCols=imputed),
        # 'keep' + dropLast=True: cada categoría vista tiene su dummy y una categoría nueva queda
        # en ceros, igual que OneHotEncoder(handle_unknown="ignore") de scikit-learn.
        *[StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep",
                        stringOrderType="alphabetAsc") for c in MODEL_CAT],
        OneHotEncoder(inputCols=[c + "_idx" for c in MODEL_CAT],
                      outputCols=[c + "_ohe" for c in MODEL_CAT], dropLast=True),
        VectorAssembler(inputCols=scaled_in, outputCol="num_vec"),
        StandardScaler(inputCol="num_vec", outputCol="num_scaled", withMean=True, withStd=True),
        VectorAssembler(inputCols=["num_scaled"] + [c + "_ohe" for c in MODEL_CAT], outputCol="features"),
    ]
    model = Pipeline(stages=stages).fit(spark_clip_log(train, bounds))
    return model, bounds


def spark_preprocess(spark, n_partitions: int | None = None):
    """CSV -> partición común -> Pipeline ajustado en train -> (train, test) cacheados con id, default, features.

    Con `n_partitions`, los DataFrames finales se reducen con `coalesce` (sin barajar las filas, de modo
    que el particionado es determinista) antes de cachearlos.
    """
    from pyspark import StorageLevel
    from pyspark.sql import functions as F

    base = spark_load_model_frame(spark).persist(StorageLevel.MEMORY_AND_DISK)
    train = base.filter(F.col("split") == "train")
    test = base.filter(F.col("split") == "test")
    model, bounds = spark_fit_preprocessor(train)

    def apply(d):
        out = model.transform(spark_clip_log(d, bounds)).select("id", TARGET, "features")
        if n_partitions:
            out = out.coalesce(n_partitions)
        return out.persist(StorageLevel.MEMORY_AND_DISK)  # caché obligatorio tras el VectorAssembler

    train_f, test_f = apply(train), apply(test)
    n_train, n_test = train_f.count(), test_f.count()  # materializa el caché
    base.unpersist()
    return {"train": train_f, "test": test_f, "model": model, "bounds": bounds,
            "n_train": n_train, "n_test": n_test}


def spark_feature_names(df, col: str = "features") -> list:
    """Nombres de las posiciones del vector `features` en el formato de scikit-learn."""
    attrs = df.schema[col].metadata["ml_attr"]["attrs"]
    # Las primeras posiciones son las numéricas escaladas (en el orden de SCALED_COLS); el
    # StandardScaler no propaga sus nombres, así que solo se leen los de las dummies.
    ohe = sorted((a["idx"], a["name"]) for group in attrs.values() for a in group
                 if a["idx"] >= len(SCALED_COLS))
    return list(SCALED_COLS) + [name.replace("_ohe_", "_", 1) for _, name in ohe]
