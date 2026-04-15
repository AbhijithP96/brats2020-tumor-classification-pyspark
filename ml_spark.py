"""
BraTS2020 — Random Forest vs GBT Classifier with PySpark ML
============================================================
Predicts tumour grade (HGG=1, LGG=0) from the extracted features.
Both models are evaluated with 5-fold cross-validation for an honest
performance estimate, then compared side-by-side.
"""

from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from rich import box
from rich.console import Console
from rich.table import Table

import pandas as pd

console = Console()

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

spark = (
    SparkSession.builder.appName("BraTS2020_RF_vs_GBT")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = spark.read.csv("./brats2020_features_cleaned.csv", inferSchema=True, header=True)

console.rule("[bold cyan]Dataset")
console.print(f"  Rows    : [bold]{df.count()}[/]")
console.print(f"  Columns : [bold]{len(df.columns)}[/]")

# ---------------------------------------------------------------------------
# Assemble features
# ---------------------------------------------------------------------------

feature_cols = [
    c
    for c, t in df.dtypes
    if t in ("int", "float", "double") and c not in ("volume_id", "grade")
]

assembler = VectorAssembler(
    inputCols=feature_cols, outputCol="features", handleInvalid="skip"
)

df_ml = assembler.transform(df).select(
    "features", col("grade").cast("double").alias("label")
)

# ---------------------------------------------------------------------------
# 80 / 20 hold-out split
# CV trains on the 80% portion; we keep the 20% hold-out completely unseen
# until final evaluation so we get an unbiased test score.
# ---------------------------------------------------------------------------

df_train, df_test = df_ml.randomSplit([0.8, 0.2], seed=42)

console.rule("[bold cyan]Split + Class Distribution")
console.print(f"  Train rows : [bold]{df_train.count()}[/]")
console.print(f"  Test rows  : [bold]{df_test.count()}[/]")

# Check class balance in the test set — important for imbalanced datasets
console.print("\n  Test set class distribution:")
df_test.groupBy("label").count().orderBy("label").show()

# ---------------------------------------------------------------------------
# Evaluator (AUC-ROC used as the CV scoring metric)
# ---------------------------------------------------------------------------

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

# ---------------------------------------------------------------------------
# Helper: compute all metrics on a predictions DataFrame
# ---------------------------------------------------------------------------


def compute_metrics(predictions):
    return {
        "accuracy": MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        ).evaluate(predictions),
        "auc": auc_evaluator.evaluate(predictions),
        "precision": MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
        ).evaluate(predictions),
        "recall": MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedRecall"
        ).evaluate(predictions),
        "f1": MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        ).evaluate(predictions),
    }


def print_metrics(title, metrics):
    t = Table(
        title=title, box=box.SIMPLE, show_header=True, header_style="bold magenta"
    )
    t.add_column("Metric", style="dim")
    t.add_column("Value", justify="right")
    t.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    t.add_row("AUC-ROC", f"{metrics['auc']:.4f}")
    t.add_row("Weighted Precision", f"{metrics['precision']:.4f}")
    t.add_row("Weighted Recall", f"{metrics['recall']:.4f}")
    t.add_row("F1 Score", f"{metrics['f1']:.4f}")
    console.print(t)


# ---------------------------------------------------------------------------
# Model 1: Random Forest with 5-fold cross-validation
# ParamGridBuilder with a single set of params = pure CV, no grid search.
# Add more param values later to turn this into a hyperparameter search.
# ---------------------------------------------------------------------------

console.rule("[bold cyan]Random Forest with 5-Fold CV")

rf = RandomForestClassifier(
    featuresCol="features", labelCol="label", numTrees=20, maxDepth=5, seed=42
)
rf_param_grid = ParamGridBuilder().build()  # empty grid = just CV, no tuning

rf_cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=rf_param_grid,
    evaluator=auc_evaluator,
    numFolds=5,
    seed=42,
)

rf_cv_model = rf_cv.fit(df_train)
rf_cv_auc = max(rf_cv_model.avgMetrics)
console.print(f"  CV mean AUC-ROC : [bold green]{rf_cv_auc:.4f}[/]")

rf_predictions = rf_cv_model.transform(df_test)
rf_metrics = compute_metrics(rf_predictions)
print_metrics("Random Forest on Hold-out Test", rf_metrics)

# ---------------------------------------------------------------------------
# Model 2: Gradient Boosted Trees with 5-fold cross-validation
# GBT only supports binary classification, which fits perfectly here.
# maxIter=20 keeps training fast; increase for better accuracy.
# ---------------------------------------------------------------------------

console.rule("[bold cyan]GBT Classifier with 5-Fold CV")

gbt = GBTClassifier(
    featuresCol="features", labelCol="label", maxIter=20, maxDepth=5, seed=42
)
gbt_param_grid = ParamGridBuilder().build()

gbt_cv = CrossValidator(
    estimator=gbt,
    estimatorParamMaps=gbt_param_grid,
    evaluator=auc_evaluator,
    numFolds=5,
    seed=42,
)

gbt_cv_model = gbt_cv.fit(df_train)
gbt_cv_auc = max(gbt_cv_model.avgMetrics)
console.print(f"  CV mean AUC-ROC : [bold green]{gbt_cv_auc:.4f}[/]")

gbt_predictions = gbt_cv_model.transform(df_test)
gbt_metrics = compute_metrics(gbt_predictions)
print_metrics("GBT on Hold-out Test", gbt_metrics)

# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

console.rule("[bold cyan]Model Comparison")
cmp = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
cmp.add_column("Metric", style="dim")
cmp.add_column("Random Forest", justify="right")
cmp.add_column("GBT", justify="right")
cmp.add_column("Winner", justify="center")

for metric, label in [
    ("accuracy", "Accuracy"),
    ("auc", "AUC-ROC"),
    ("precision", "Weighted Precision"),
    ("recall", "Weighted Recall"),
    ("f1", "F1 Score"),
]:
    rf_val = rf_metrics[metric]
    gbt_val = gbt_metrics[metric]
    winner = "[green]RF[/]" if rf_val >= gbt_val else "[green]GBT[/]"
    cmp.add_row(label, f"{rf_val:.4f}", f"{gbt_val:.4f}", winner)
console.print(cmp)

# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------

console.rule("[bold cyan]Top 10 Feature Importances (GBT)")
gbt_best = gbt_cv_model.bestModel
importances_gbt = sorted(
    zip(feature_cols, gbt_best.featureImportances.toArray()),
    key=lambda x: x[1],
    reverse=True,
)

imp_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
imp_table.add_column("Rank", style="dim", justify="right")
imp_table.add_column("Feature", style="cyan")
imp_table.add_column("Importance", justify="right")
for rank, (name, score) in enumerate(importances_gbt[:10], start=1):
    imp_table.add_row(str(rank), name, f"{score:.4f}")
console.print(imp_table)


console.rule("[bold cyan]Top 10 Feature Importances (RF)")
rf_best = rf_cv_model.bestModel
importances_rf = sorted(
    zip(feature_cols, rf_best.featureImportances.toArray()),
    key=lambda x: x[1],
    reverse=True,
)

imp_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
imp_table.add_column("Rank", style="dim", justify="right")
imp_table.add_column("Feature", style="cyan")
imp_table.add_column("Importance", justify="right")
for rank, (name, score) in enumerate(importances_rf[:10], start=1):
    imp_table.add_row(str(rank), name, f"{score:.4f}")
console.print(imp_table)

# ---------------------------------------------------------------------------
# Export for Power BI
# ---------------------------------------------------------------------------

console.rule("[bold cyan]Export for Power BI")

LABEL_MAP = {0.0: "LGG", 1.0: "HGG"}


def save_and_log(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    console.print(
        f"  Saved [bold green]{path}[/]  ({len(df)} rows × {len(df.columns)} cols)"
    )


# --- 1. model_comparison.csv ---
rf_rows = [
    {"model": "Random Forest", "metric": k, "value": round(v, 6)}
    for k, v in rf_metrics.items()
]
gbt_rows = [
    {"model": "GBT", "metric": k, "value": round(v, 6)} for k, v in gbt_metrics.items()
]
save_and_log(pd.DataFrame(rf_rows + gbt_rows), "model_comparison.csv")

# --- 2. rf_feature_importance.csv ---
save_and_log(
    pd.DataFrame(importances_rf, columns=["feature", "importance"]).assign(
        rank=range(1, len(importances_rf) + 1)
    )[["rank", "feature", "importance"]],
    "rf_feature_importance.csv",
)

# --- 3. gbt_feature_importance.csv ---
save_and_log(
    pd.DataFrame(importances_gbt, columns=["feature", "importance"]).assign(
        rank=range(1, len(importances_gbt) + 1)
    )[["rank", "feature", "importance"]],
    "gbt_feature_importance.csv",
)


# --- 4. rf_predictions.csv / gbt_predictions.csv ---
def export_predictions(predictions, path: str) -> None:
    pdf = predictions.select("label", "prediction", "probability").toPandas()
    pdf["actual_label"] = pdf["label"].map(LABEL_MAP)
    pdf["predicted_label"] = pdf["prediction"].map(LABEL_MAP)
    pdf["prob_LGG"] = pdf["probability"].apply(lambda v: round(float(v[0]), 6))
    pdf["prob_HGG"] = pdf["probability"].apply(lambda v: round(float(v[1]), 6))
    pdf.drop(columns=["label", "prediction", "probability"], inplace=True)
    save_and_log(pdf, path)


export_predictions(rf_predictions, "rf_predictions.csv")
export_predictions(gbt_predictions, "gbt_predictions.csv")


# --- 5. class_distribution.csv ---
def class_counts(split_df, split_name: str) -> pd.DataFrame:
    rows = split_df.groupBy("label").count().toPandas()
    rows["split"] = split_name
    rows["grade"] = rows["label"].map(LABEL_MAP)
    return rows[["split", "grade", "count"]]


dist = pd.concat(
    [class_counts(df_train, "train"), class_counts(df_test, "test")], ignore_index=True
)
save_and_log(dist, "class_distribution.csv")


# --- 6. confusion_matrices.csv ---
def build_confusion_matrix(predictions, model_name: str) -> pd.DataFrame:
    rdd = predictions.select(
        col("prediction").cast("double"),
        col("label").cast("double"),
    ).rdd
    cm = MulticlassMetrics(rdd).confusionMatrix().toArray().astype(int)
    # Rows = actual, Cols = predicted; labels ordered 0 (LGG) then 1 (HGG)
    rows = []
    for actual_idx, actual_label in enumerate(["LGG", "HGG"]):
        for pred_idx, pred_label in enumerate(["LGG", "HGG"]):
            rows.append(
                {
                    "model": model_name,
                    "actual": actual_label,
                    "predicted": pred_label,
                    "count": cm[actual_idx][pred_idx],
                }
            )
    return pd.DataFrame(rows)


cm_rf = build_confusion_matrix(rf_predictions, "Random Forest")
cm_gbt = build_confusion_matrix(gbt_predictions, "GBT")
save_and_log(pd.concat([cm_rf, cm_gbt], ignore_index=True), "confusion_matrices.csv")

spark.stop()
