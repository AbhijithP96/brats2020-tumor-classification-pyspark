"""
BraTS2020 — Exploratory Data Analysis with PySpark
====================================================
This script loads the extracted feature dataset into a Spark DataFrame
and performs basic exploratory data analysis (EDA).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    when,
    isnan,
    avg,
    std,
    max,
    min,
    round as spark_round,
)
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Create a Spark session
# ---------------------------------------------------------------------------

spark = (
    SparkSession.builder.appName("BraTS2020")
    .config("spark.sql.shuffle.partitions", "8")  # keep it small for local mode
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")  # suppress INFO noise

# ---------------------------------------------------------------------------
# Define the schema explicitly
# ---------------------------------------------------------------------------

schema = StructType(
    [
        StructField("volume_id", IntegerType()),
        StructField("voxels_ch0", IntegerType()),
        StructField("voxels_ch1", IntegerType()),
        StructField("voxels_ch2", IntegerType()),
        StructField("total_tumour_voxels", IntegerType()),
        StructField("ratio_ch0", FloatType()),
        StructField("ratio_ch1", FloatType()),
        StructField("ratio_ch2", FloatType()),
        StructField("tumour_slice_count", IntegerType()),
        StructField("start_slice", IntegerType()),
        StructField("end_slice", IntegerType()),
        StructField("flair_mean", FloatType()),
        StructField("flair_std", FloatType()),
        StructField("flair_min", FloatType()),
        StructField("flair_max", FloatType()),
        StructField("flair_skewness", FloatType()),
        StructField("t1_mean", FloatType()),
        StructField("t1_std", FloatType()),
        StructField("t1_min", FloatType()),
        StructField("t1_max", FloatType()),
        StructField("t1_skewness", FloatType()),
        StructField("t1ce_mean", FloatType()),
        StructField("t1ce_std", FloatType()),
        StructField("t1ce_min", FloatType()),
        StructField("t1ce_max", FloatType()),
        StructField("t1ce_skewness", FloatType()),
        StructField("t2_mean", FloatType()),
        StructField("t2_std", FloatType()),
        StructField("t2_min", FloatType()),
        StructField("t2_max", FloatType()),
        StructField("t2_skewness", FloatType()),
        StructField("centroid_x", FloatType()),
        StructField("centroid_y", FloatType()),
        StructField("centroid_slice", FloatType()),
        StructField("spread_x", FloatType()),
        StructField("spread_y", FloatType()),
        StructField("spread_slice", FloatType()),
        StructField("brats20_id", StringType()),
        StructField("grade", StringType()),
        StructField("age", FloatType()),
    ]
)

# ---------------------------------------------------------------------------
# Load the CSV
# header=True tells Spark the first row contains column names, not data.
# ---------------------------------------------------------------------------

df = spark.read.csv("./data/brats2020_features.csv", schema=schema, header=True)

# Cache the DataFrame so repeated actions don't re-read the file each time
df.cache()

# -----------------------------------------------------------------------------
# Dataset Shape
# -----------------------------------------------------------------------------
console.rule("[bold cyan]1. Dataset Shape")
n_rows = df.count()
n_columns = len(df.columns)
console.print(f" Rows: {n_rows}")
console.print(f" Columns: {n_columns}")

# -----------------------------------------------------------------------------
# Dataset Schema
# -----------------------------------------------------------------------------
console.rule("[bold cyan]2. Dataset Schema")
df.printSchema()

# -----------------------------------------------------------------------------
# Sample Data
# -----------------------------------------------------------------------------
console.rule("[bold cyan]3. Sample Data (First 5 Rows)")
cols = [
    "volume_id",
    "grade",
    "age",
    "total_tumour_voxels",
    "tumour_slice_count",
]
df.select(cols).show(n=5, truncate=False)

# -----------------------------------------------------------------------------
# Summary Statistics
# -----------------------------------------------------------------------------
console.rule("[bold cyan]4. Summary Statistics")
cols = [
    "age",
    "total_tumour_voxels",
    "tumour_slice_count",
    "flair_mean",
    "t1_mean",
    "t1ce_mean",
    "t2_mean",
]
df.select(cols).describe().show()

# -----------------------------------------------------------------------------
# Missing Values
# -----------------------------------------------------------------------------
console.rule("[bold cyan]5. Missing values")
string_cols = [c for c, t in df.dtypes if t == "string"]

df.select(
    [
        count(
            when(
                col(c).isNull() | (isnan(col(c)) if c not in string_cols else False), c
            )
        ).alias(c)
        for c in df.columns
    ]
).show(vertical=True)

# -----------------------------------------------------------------------------
# HGG/LGG counts
# -----------------------------------------------------------------------------
console.rule("[bold cyan]6. HGG/LGG counts")
total = df.count()
df.groupBy("grade").count().withColumn(
    "percent", spark_round(col("count") / total * 100, 1)
).orderBy("count", ascending=False).show()

# -----------------------------------------------------------------------------
# Stats Grouped by Grade
# -----------------------------------------------------------------------------
console.rule("[bold cyan]7. Stats Grouped by Grade")
numeric_cols = [
    "age",
    "total_tumour_voxels",
    "tumour_slice_count",
    "ratio_ch0",
    "ratio_ch1",
    "ratio_ch2",
    "flair_mean",
    "t1_mean",
    "t1ce_mean",
    "t2_mean",
]
# avg
df.groupBy("grade").agg(
    *[spark_round(avg(col(c)), 2).alias(f"avg_{c}") for c in numeric_cols]
).orderBy("grade").show(truncate=False)
# std
df.groupBy("grade").agg(
    *[spark_round(std(col(c)), 2).alias(f"std_{c}") for c in numeric_cols]
).orderBy("grade").show(truncate=False)
# max
df.groupBy("grade").agg(
    *[spark_round(max(col(c)), 2).alias(f"max_{c}") for c in numeric_cols]
).orderBy("grade").show(truncate=False)
# min
df.groupBy("grade").agg(
    *[spark_round(min(col(c)), 2).alias(f"min_{c}") for c in numeric_cols]
).orderBy("grade").show(truncate=False)

# -----------------------------------------------------------------------------
# Correlation matrix across all numeric features
# -----------------------------------------------------------------------------
console.rule("[bold cyan]8. Correlation matrix.")
# Collect all numeric columns (both int and float) except volume_id
corr_cols = [c for c, t in df.dtypes if t in ("int", "float") and c != "volume_id"]

# VectorAssembler combines multiple columns into a single vector column,
# which is required by Spark's Correlation utility.
assembler = VectorAssembler(
    inputCols=corr_cols, outputCol="features", handleInvalid="skip"
)
df_vec = assembler.transform(df).select("features")

# Compute the Pearson correlation matrix (returns a 1-row DataFrame)
corr_matrix = Correlation.corr(df_vec, "features").head()[0].toArray()

# Convert to pandas and plot as a seaborn heatmap
corr_df = pd.DataFrame(corr_matrix, index=corr_cols, columns=corr_cols).round(2)

plt.figure(figsize=(18, 14))
sns.heatmap(
    corr_df,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    annot_kws={"size": 7},
)
plt.title("Pearson Correlation Matrix of BraTS2020 Features", fontsize=14)
plt.tight_layout()
plt.savefig("Pearson_Correlation_Matrix_BraTS2020_Features.png", dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# Feature correlation with grade label
# Encode HGG=1, LGG=0, then correlate every numeric feature against it.
# Higher absolute value = stronger signal for classification.
# -----------------------------------------------------------------------------
console.rule("[bold cyan]9. Feature Correlation with Grade")

# Encode grade as a numeric label
df_labeled = df.withColumn("grade_label", when(col("grade") == "HGG", 1).otherwise(0))

# Compute Pearson correlation of each feature against grade_label
feature_cols = [c for c, t in df.dtypes if t in ("int", "float") and c != "volume_id"]
correlations = [(c, df_labeled.stat.corr(c, "grade_label")) for c in feature_cols]

# Sort by absolute correlation descending
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

# Display as a table
corr_grade_df = pd.DataFrame(correlations, columns=["feature", "corr_with_grade"])
corr_grade_df["abs_corr"] = corr_grade_df["corr_with_grade"].abs().round(4)
corr_grade_df["corr_with_grade"] = corr_grade_df["corr_with_grade"].round(4)
# print(corr_grade_df.to_string(index=False))

# Bar chart — most predictive features at the top
plt.figure(figsize=(10, len(feature_cols) * 0.35 + 1))
colors = ["#d73027" if v > 0 else "#4575b4" for v in corr_grade_df["corr_with_grade"]]
plt.barh(corr_grade_df["feature"], corr_grade_df["corr_with_grade"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Pearson Correlation with Grade (HGG=1, LGG=0)")
plt.title("Feature Correlation with Grade Label")
plt.gca().invert_yaxis()  # highest abs correlation at the top
plt.tight_layout()
plt.savefig("Pearson_Correlation_with_Grade.png", dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# Build cleaned DataFrame: drop low-signal/ redundant columns, encode grade
# -----------------------------------------------------------------------------
console.rule("[bold cyan]10. Save Cleaned Dataset")

drop_cols = [
    "voxels_ch0",
    "voxels_ch1",
    "voxels_ch2",
    "t2_max",
    "end_slice",
    "t1_max",
    "spread_x",
    "spread_y",
    "t1_skewness",
    "tumour_slice_count",
    "start_slice",
]

df_clean = df.drop(*drop_cols).withColumn(
    "grade", when(col("grade") == "HGG", 1).otherwise(0)
)

console.print(f"  Columns kept : {len(df_clean.columns)}")
console.print(f"  Columns dropped : {len(drop_cols)}")
df_clean.show(5, truncate=False)

df_clean.coalesce(1).write.csv(
    "./brats2020_features_clean.csv", header=True, mode="overwrite"
)
console.print("  Saved -> brats2020_features_clean.csv")

# Stop the spark session
spark.stop()
